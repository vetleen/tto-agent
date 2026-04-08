"""WebSocket consumer for live meeting transcription.

Protocol (see plan in C:\\ClaudeCodeData\\plans\\wobbly-sauteeing-wombat.md
section "Phase 3"):

  C->S text json:    start | chunk_meta | extend_auto_stop | stop
  C->S binary frame: raw audio bytes for the most-recent unfulfilled chunk_meta
  S->C text json:    started | segment.queued | segment.ready |
                     segment.failed | stopped | error

The consumer accepts complete, independently-decodable audio container files
(produced by stopping/restarting MediaRecorder every ~30 s on the client) and
hands each chunk to the existing TranscriptionService via a Celery task.
Audio is never persisted past the chunk task — it's deleted in a finally:.
"""
from __future__ import annotations

import json
import logging
import uuid as uuid_lib
from typing import Any

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


class MeetingTranscribeConsumer(AsyncWebsocketConsumer):
    """Consumer that ingests live audio chunks and proxies transcription results.

    Lifecycle:
      connect    -> auth, ownership check, transition meeting to LIVE_TRANSCRIBING
      receive    -> chunk_meta + binary frame -> temp file -> celery task
      group_send -> segment.ready / segment.failed forwarded to client
      stop       -> finalize meeting (status=READY)
      disconnect -> if not stopped, finalize meeting (status=INTERRUPTED)
    """

    async def connect(self):
        self.user = self.scope.get("user")
        self.meeting_uuid: str | None = None
        self.meeting_id: int | None = None
        self._stopped: bool = False
        self._stop_requested: bool = False
        self._pending_meta: list[dict[str, Any]] = []
        self._segment_index_base: int = 0
        self._segments_total: int = 0
        self._segments_failed: int = 0
        self._model_id: str = ""

        if not self.user or self.user.is_anonymous:
            await self.close(code=4401)
            return

        url_uuid = self.scope["url_route"]["kwargs"].get("meeting_uuid")
        try:
            uuid_lib.UUID(str(url_uuid))
        except (ValueError, TypeError):
            await self.close(code=4400)
            return

        meeting = await self._load_and_lock_meeting(url_uuid)
        if meeting is None:
            await self.close(code=4404)
            return
        if meeting["created_by_id"] != self.user.id:
            await self.close(code=4403)
            return

        self.meeting_id = meeting["id"]
        self.meeting_uuid = str(meeting["uuid"])
        self._segment_index_base = meeting["segment_index_base"]
        self._model_id = meeting["model_id"]
        self._group_name = f"meetings.{self.meeting_uuid}"

        await self.channel_layer.group_add(self._group_name, self.channel_name)
        await self.accept()
        await self.send(text_data=json.dumps({
            "type": "started",
            "meeting_id": self.meeting_uuid,
            "started_at": meeting["started_at"],
            "segment_index_base": self._segment_index_base,
        }))

    async def disconnect(self, close_code):
        if self.meeting_id and not self._stopped and not self._stop_requested:
            try:
                await self._finalize_meeting(interrupted=True)
            except Exception:
                logger.exception("disconnect: finalize_meeting failed")
        if getattr(self, "_group_name", None):
            try:
                await self.channel_layer.group_discard(self._group_name, self.channel_name)
            except Exception:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data is not None:
            await self._handle_binary_frame(bytes_data)
            return
        if not text_data:
            return
        try:
            payload = json.loads(text_data)
        except (json.JSONDecodeError, ValueError):
            await self._send_error("Invalid JSON.")
            return
        msg_type = payload.get("type")
        if msg_type == "start":
            # Optional client hello — we already started in connect(). Echo back if needed.
            return
        if msg_type == "chunk_meta":
            await self._handle_chunk_meta(payload)
            return
        if msg_type == "extend_auto_stop":
            # Auto-stop is enforced client-side. Server just acknowledges.
            return
        if msg_type == "stop":
            await self._handle_stop()
            return
        await self._send_error(f"Unknown message type: {msg_type!r}")

    # ------------------------------------------------------------------ binary

    async def _handle_chunk_meta(self, payload: dict) -> None:
        try:
            segment_index = int(payload["segment_index"])
            byte_length = int(payload["byte_length"])
        except (KeyError, ValueError, TypeError):
            await self._send_error("chunk_meta missing or malformed segment_index/byte_length.")
            return
        if segment_index < 0 or byte_length <= 0:
            await self._send_error("chunk_meta has invalid values.")
            return
        max_bytes = getattr(settings, "MEETING_CHUNK_MAX_BYTES", 20 * 1024 * 1024)
        if byte_length > max_bytes:
            await self._send_error(f"chunk too large: {byte_length} > {max_bytes}")
            return
        # Push onto the pending queue. The next binary frame will consume it.
        self._pending_meta.append({
            "segment_index": segment_index,
            "byte_length": byte_length,
            "mime": str(payload.get("mime") or "audio/webm"),
            "start_offset_seconds": float(payload.get("start_offset_seconds") or 0.0),
        })

    async def _handle_binary_frame(self, raw: bytes) -> None:
        if not self._pending_meta:
            await self._send_error("Binary frame received with no pending chunk_meta.")
            return
        meta = self._pending_meta.pop(0)
        if len(raw) != meta["byte_length"]:
            await self._send_error(
                f"chunk byte_length mismatch: expected {meta['byte_length']}, got {len(raw)}"
            )
            return
        try:
            temp_path = await self._write_chunk(meta, raw)
        except Exception as exc:
            logger.exception("write_chunk_to_temp failed")
            await self._send_error(f"Could not buffer chunk: {exc}")
            return

        self._segments_total += 1
        await self.send(text_data=json.dumps({
            "type": "segment.queued",
            "segment_index": meta["segment_index"],
        }))

        try:
            await self._enqueue_chunk_task(meta, str(temp_path))
        except Exception as exc:
            logger.exception("enqueue chunk task failed")
            await self._send_error(f"Could not start transcription: {exc}")

    async def _handle_stop(self) -> None:
        self._stop_requested = True
        try:
            duration_seconds, segments_total, segments_failed = await self._finalize_meeting(interrupted=False)
        except Exception:
            logger.exception("_handle_stop: finalize failed")
            duration_seconds, segments_total, segments_failed = 0, self._segments_total, self._segments_failed
        await self.send(text_data=json.dumps({
            "type": "stopped",
            "duration_seconds": duration_seconds,
            "segments_total": segments_total,
            "segments_failed": segments_failed,
        }))
        self._stopped = True
        try:
            await self.close()
        except Exception:
            pass

    async def _send_error(self, message: str) -> None:
        try:
            await self.send(text_data=json.dumps({"type": "error", "message": message}))
        except Exception:
            pass

    # -------------------------------------------------- channel-layer handlers

    async def segment_ready(self, event):
        self._segments_total = max(self._segments_total, int(event.get("segment_index", 0)) + 1)
        await self.send(text_data=json.dumps({
            "type": "segment.ready",
            "segment_index": event.get("segment_index"),
            "text": event.get("text", ""),
            "start_offset_seconds": event.get("start_offset_seconds", 0),
            "transcription_model": event.get("transcription_model", ""),
        }))

    async def segment_failed(self, event):
        self._segments_failed += 1
        await self.send(text_data=json.dumps({
            "type": "segment.failed",
            "segment_index": event.get("segment_index"),
            "error": event.get("error", ""),
        }))

    # ----------------------------------------------------- DB-bound (sync)

    @database_sync_to_async
    def _load_and_lock_meeting(self, meeting_uuid):
        from .models import Meeting, MeetingTranscriptSegment

        try:
            meeting = Meeting.objects.get(uuid=meeting_uuid)
        except Meeting.DoesNotExist:
            return None

        max_existing = (
            MeetingTranscriptSegment.objects
            .filter(meeting=meeting)
            .order_by("-segment_index")
            .values_list("segment_index", flat=True)
            .first()
        )
        segment_index_base = (max_existing + 1) if max_existing is not None else 0

        # Pick a transcription model from user prefs, falling back to project default.
        try:
            from core.preferences import get_preferences
            prefs = get_preferences(self.user)
            allowed_models = list(getattr(prefs, "allowed_transcription_models", None) or [])
        except Exception:
            allowed_models = []
        model_id = (
            allowed_models[0]
            if allowed_models
            else getattr(settings, "TRANSCRIPTION_DEFAULT_MODEL", "openai/gpt-4o-mini-transcribe")
        )

        # Transition the meeting to LIVE_TRANSCRIBING (preserve started_at on resume).
        update_fields = ["status", "transcript_source", "updated_at"]
        meeting.status = Meeting.Status.LIVE_TRANSCRIBING
        meeting.transcript_source = Meeting.TranscriptSource.LIVE
        if not meeting.started_at:
            meeting.started_at = timezone.now()
            update_fields.append("started_at")
        meeting.transcription_error = ""
        update_fields.append("transcription_error")
        meeting.save(update_fields=update_fields)

        return {
            "id": meeting.id,
            "uuid": str(meeting.uuid),
            "created_by_id": meeting.created_by_id,
            "started_at": meeting.started_at.isoformat() if meeting.started_at else None,
            "segment_index_base": segment_index_base,
            "model_id": model_id,
        }

    @database_sync_to_async
    def _write_chunk(self, meta: dict, raw: bytes):
        from .services.chunks import write_chunk_to_temp

        return write_chunk_to_temp(
            self.meeting_uuid, meta["segment_index"], raw, meta["mime"]
        )

    @database_sync_to_async
    def _enqueue_chunk_task(self, meta: dict, temp_path: str) -> None:
        from .tasks import transcribe_meeting_chunk_task

        transcribe_meeting_chunk_task.delay(
            meeting_id=self.meeting_id,
            segment_index=meta["segment_index"],
            temp_path=temp_path,
            mime=meta["mime"],
            model_id=self._model_id,
            user_id=self.user.id,
            start_offset_seconds=meta["start_offset_seconds"],
        )

    @database_sync_to_async
    def _finalize_meeting(self, *, interrupted: bool):
        from .models import Meeting, MeetingTranscriptSegment
        from .services.chunks import recompute_meeting_transcript

        ended = timezone.now()
        duration_seconds = 0
        try:
            meeting = Meeting.objects.get(pk=self.meeting_id)
        except Meeting.DoesNotExist:
            return 0, self._segments_total, self._segments_failed

        if meeting.started_at:
            duration_seconds = max(0, int((ended - meeting.started_at).total_seconds()))

        meeting.status = Meeting.Status.INTERRUPTED if interrupted else Meeting.Status.READY
        meeting.ended_at = ended
        meeting.duration_seconds = duration_seconds
        meeting.save(update_fields=["status", "ended_at", "duration_seconds", "updated_at"])

        recompute_meeting_transcript(self.meeting_id)

        total = MeetingTranscriptSegment.objects.filter(meeting_id=self.meeting_id).count()
        failed = MeetingTranscriptSegment.objects.filter(
            meeting_id=self.meeting_id,
            status=MeetingTranscriptSegment.Status.FAILED,
        ).count()
        return duration_seconds, total, failed
