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

import asyncio
import json
import logging
import uuid as uuid_lib
from typing import Any

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# Interval at which the consumer sends a ping frame while idle so the socket
# doesn't go unused long enough for any transport (Heroku router, corporate
# proxy, browser throttle) to decide it's dead. Kept well under Heroku's ~55s
# WebSocket idle window.
MEETING_WS_HEARTBEAT_SECONDS = 20


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
        # Realtime-mode state. None in the default "chunked" path — filled in
        # by _maybe_start_realtime() when the cascading preference asks for
        # the OpenAI Realtime session instead of the per-chunk Celery tasks.
        self._realtime_mode: str = "chunked"
        self._realtime_session = None
        self._pcm_pipe = None
        self._realtime_tasks: list[asyncio.Task] = []
        self._realtime_started: bool = False
        self._realtime_language: str = ""
        self._total_pcm_bytes: int = 0
        self._live_segment_counter: int = 0  # counts utterances persisted in realtime mode

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
        self._realtime_mode = meeting.get("live_mode") or "chunked"
        self._realtime_language = meeting.get("forced_language") or ""
        self._realtime_prompt = meeting.get("prompt") or ""
        self._live_segment_counter = self._segment_index_base
        self._group_name = f"meetings.{self.meeting_uuid}"

        await self.channel_layer.group_add(self._group_name, self.channel_name)
        await self.accept()
        # Hand the effective mode to the client so it can pick the right
        # MediaRecorder strategy: chunked ⇒ stop/restart every 30s producing
        # self-contained WebM files; realtime ⇒ single continuous recorder
        # with a small timeslice so ffmpeg sees one unbroken Matroska stream.
        await self.send(text_data=json.dumps({
            "type": "started",
            "meeting_id": self.meeting_uuid,
            "started_at": meeting["started_at"],
            "segment_index_base": self._segment_index_base,
            "live_mode": self._realtime_mode,
        }))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def disconnect(self, close_code):
        task = getattr(self, "_heartbeat_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        # Teardown realtime first so pending utterances get a chance to persist
        # before we flip the meeting row to INTERRUPTED.
        if self._realtime_started:
            try:
                await self._teardown_realtime()
            except Exception:
                logger.exception("disconnect: realtime teardown failed")
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

    async def _heartbeat_loop(self):
        """Send a lightweight ping frame at a fixed cadence.

        The client ignores `type=ping`. This keeps the WS socket non-idle so
        transports that kill idle connections (Heroku router, proxies) don't
        tear it down during a quiet stretch — e.g. when transcription of a
        chunk takes longer than usual and no other frames are flowing.
        """
        try:
            while True:
                await asyncio.sleep(MEETING_WS_HEARTBEAT_SECONDS)
                try:
                    await self.send(text_data=json.dumps({"type": "ping"}))
                except Exception:
                    return
        except asyncio.CancelledError:
            raise

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
        if msg_type == "set_model":
            await self._handle_set_model(payload)
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
        if self._realtime_mode != "chunked":
            await self._handle_binary_frame_realtime(raw)
            return
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

    # -------------------------------------------------- realtime mode

    async def _handle_binary_frame_realtime(self, raw: bytes) -> None:
        """Forward raw container bytes straight to the ffmpeg pipe.

        chunk_meta is optional in realtime mode — if the client sends it we
        use the first one's ``mime`` to pick ffmpeg's input demuxer; any
        subsequent metadata is consumed but not required to match. The
        byte_length sanity check is kept as DoS protection.
        """
        max_bytes = getattr(settings, "MEETING_CHUNK_MAX_BYTES", 20 * 1024 * 1024)
        if len(raw) > max_bytes:
            await self._send_error(f"chunk too large: {len(raw)} > {max_bytes}")
            return
        mime = "audio/webm"
        if self._pending_meta:
            # Drain the metadata queue so it doesn't grow unbounded. The first
            # entry provides the mime for the pipe on the very first frame.
            meta = self._pending_meta.pop(0)
            mime = meta.get("mime") or mime

        if not self._realtime_started:
            try:
                await self._start_realtime_session(mime)
            except Exception as exc:
                logger.exception("realtime: could not start session")
                await self._send_error(f"Could not start realtime session: {exc}")
                self._realtime_mode = "chunked"  # give up on realtime for this meeting
                return
        try:
            await self._pcm_pipe.write(raw)
        except Exception as exc:
            logger.warning("realtime: pipe write failed (%s)", exc)

    async def _start_realtime_session(self, mime: str) -> None:
        from meetings.services.pcm_pipe import PcmPipe
        from meetings.services.realtime_session import build_realtime_session

        self._pcm_pipe = PcmPipe(mime=mime)
        await self._pcm_pipe.start()

        self._realtime_session = build_realtime_session(
            model_id=self._model_id,
            prompt=self._realtime_prompt or None,
            language=self._realtime_language or None,
        )
        try:
            await self._realtime_session.connect()
        except Exception:
            # Unwind the pipe so we don't leak ffmpeg on connect failure.
            try:
                await self._pcm_pipe.aclose()
            finally:
                self._pcm_pipe = None
            raise

        self._realtime_tasks = [
            asyncio.create_task(self._pump_pcm_to_realtime()),
            asyncio.create_task(self._consume_realtime_events()),
        ]
        self._realtime_started = True
        logger.info(
            "realtime: session started (meeting=%s model=%s mime=%s)",
            self.meeting_uuid, self._model_id, mime,
        )

    async def _pump_pcm_to_realtime(self) -> None:
        """Forward decoded PCM from the ffmpeg pipe to the realtime session."""
        try:
            async for frame in self._pcm_pipe.read_frames():
                self._total_pcm_bytes += len(frame)
                await self._realtime_session.send_pcm(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("realtime: PCM pump crashed")

    async def _consume_realtime_events(self) -> None:
        """Translate structured provider events into client-facing WS messages."""
        from meetings.services.realtime_session import (
            SessionError,
            SessionStatus,
            TranscriptCompleted,
            TranscriptDelta,
        )

        try:
            async for evt in self._realtime_session.events():
                if isinstance(evt, TranscriptDelta):
                    await self.send(text_data=json.dumps({
                        "type": "transcript.delta",
                        "item_id": evt.item_id,
                        "text": evt.text,
                    }))
                elif isinstance(evt, TranscriptCompleted):
                    await self._persist_realtime_utterance(evt)
                elif isinstance(evt, SessionStatus):
                    await self.send(text_data=json.dumps({
                        "type": "session.status",
                        "state": evt.state,
                    }))
                    if evt.state == "disconnected":
                        return
                elif isinstance(evt, SessionError):
                    await self.send(text_data=json.dumps({
                        "type": "error",
                        "message": f"{evt.code}: {evt.message}",
                    }))
                    if evt.fatal:
                        return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("realtime: event consumer crashed")

    async def _persist_realtime_utterance(self, completed) -> None:
        """Create a MeetingTranscriptSegment row + push segment.ready to client."""
        # Compute the start offset from cumulative PCM bytes forwarded before
        # this utterance started. The server's clock is authoritative for the
        # server-VAD turn boundaries; we just need a stable ordering hint for
        # the UI.
        start_offset = self._total_pcm_bytes / (24_000 * 2)
        idx = await self._allocate_and_persist_segment(
            text=completed.text,
            start_offset=start_offset,
            model_id=self._model_id,
            usage=completed.usage,
        )
        if idx is None:
            return
        await self.send(text_data=json.dumps({
            "type": "segment.ready",
            "segment_index": idx,
            "text": completed.text,
            "start_offset_seconds": start_offset,
            "transcription_model": self._model_id,
        }))

    @database_sync_to_async
    def _allocate_and_persist_segment(self, *, text, start_offset, model_id, usage):
        """Persist one realtime utterance as a MeetingTranscriptSegment row.

        Also writes a cost-tracking row to LLMCallLog via log_transcription_streaming
        so billing analytics sees one row per utterance, matching how the chunked
        path produces one row per chunk.
        """
        from django.db import transaction
        from .models import Meeting, MeetingTranscriptSegment
        from .services.chunks import recompute_meeting_transcript

        try:
            with transaction.atomic():
                max_existing = (
                    MeetingTranscriptSegment.objects
                    .select_for_update()
                    .filter(meeting_id=self.meeting_id)
                    .order_by("-segment_index")
                    .values_list("segment_index", flat=True)
                    .first()
                )
                idx = (max_existing + 1) if max_existing is not None else 0
                MeetingTranscriptSegment.objects.create(
                    meeting_id=self.meeting_id,
                    segment_index=idx,
                    text=text,
                    transcription_model=model_id,
                    start_offset_seconds=start_offset,
                    status=MeetingTranscriptSegment.Status.READY,
                    transcribed_at=timezone.now(),
                )
                recompute_meeting_transcript(self.meeting_id)
            self._segments_total += 1

            # Cost logging — never raises. Realtime is officially priced per
            # token on the same rate card as the batch transcription model,
            # so we reuse the existing calculator. When the server doesn't
            # return usage (older API versions, cancelled utterances) we
            # fall through to a None cost_usd rather than guessing.
            try:
                from llm.service.logger import log_transcription_streaming
                from llm.service.pricing import calculate_transcription_cost
                from llm.types.context import RunContext

                input_tokens = (usage or {}).get("input_tokens")
                output_tokens = (usage or {}).get("output_tokens")
                total_tokens = (usage or {}).get("total_tokens")
                cost_usd = calculate_transcription_cost(
                    model_id,
                    audio_duration_seconds=0.0,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                log_transcription_streaming(
                    model=model_id,
                    context=RunContext.create(user_id=self.user.id),
                    kind="realtime_utterance",
                    item_id="",
                    audio_duration_seconds=0.0,
                    transcript_len=len(text),
                    cost_usd=cost_usd,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
            except Exception:
                logger.exception("realtime: log_transcription_streaming failed")
            return idx
        except Exception:
            logger.exception("realtime: persist utterance failed")
            return None

    async def _teardown_realtime(self) -> None:
        """Cancel forwarder tasks, flush the session, close the pipe."""
        if self._realtime_session is not None:
            try:
                await self._realtime_session.finalize(timeout=2.0)
            except Exception:
                pass
        for task in self._realtime_tasks:
            task.cancel()
        for task in self._realtime_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._realtime_tasks = []
        if self._realtime_session is not None:
            try:
                await self._realtime_session.aclose()
            except Exception:
                pass
            self._realtime_session = None
        if self._pcm_pipe is not None:
            try:
                await self._pcm_pipe.aclose()
            except Exception:
                pass
            self._pcm_pipe = None
        self._realtime_started = False

    async def _handle_set_model(self, payload: dict) -> None:
        """Switch the transcription model used for *future* chunks.

        Already-queued chunks keep whatever model they were enqueued with.
        We validate against the user's allowed models so a malicious client
        can't escape the org's allow-list via the WS.
        """
        new_model = str(payload.get("model_id") or "").strip()
        if not new_model:
            await self._send_error("set_model: missing model_id")
            return
        try:
            allowed = await self._get_allowed_transcription_models()
        except Exception:
            allowed = []
        if new_model not in allowed:
            await self._send_error(f"set_model: '{new_model}' is not allowed for this user")
            return
        self._model_id = new_model
        try:
            await self._persist_meeting_model(new_model)
        except Exception:
            logger.exception("set_model: failed to persist meeting.transcription_model")

    async def _handle_stop(self) -> None:
        self._stop_requested = True
        if self._realtime_started:
            try:
                await self._teardown_realtime()
            except Exception:
                logger.exception("_handle_stop: realtime teardown failed")
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

        # Refuse the connect if an upload-path transcription is currently
        # mid-flight on this meeting. The upload orchestrator owns the meeting
        # row's status / transcription_chunks_* fields while it runs; if we
        # let the live path take over here it would clobber those updates and
        # the polling UI would show garbage.
        if (
            meeting.transcript_source == Meeting.TranscriptSource.AUDIO_UPLOAD
            and meeting.status == Meeting.Status.LIVE_TRANSCRIBING
            and meeting.transcription_chunks_done < meeting.transcription_chunks_total
        ):
            logger.info(
                "_load_and_lock_meeting: refused WS connect for meeting %s — "
                "audio upload transcription in progress (%d/%d chunks)",
                meeting.uuid,
                meeting.transcription_chunks_done,
                meeting.transcription_chunks_total,
            )
            return None

        max_existing = (
            MeetingTranscriptSegment.objects
            .filter(meeting=meeting)
            .order_by("-segment_index")
            .values_list("segment_index", flat=True)
            .first()
        )
        segment_index_base = (max_existing + 1) if max_existing is not None else 0

        # Pick the transcription model. Priority order for the *live* path:
        #   1. Meeting-level override (only if it's live-capable)
        #   2. User/org "live" default from preferences
        #   3. First live-capable model in the allow-list
        # This means a user whose general default is a diarize model (batch-only)
        # still gets a working live session — the consumer picks a live-capable
        # model from their allow-list automatically.
        from llm.transcription_registry import get_transcription_model_info

        live_mode = "chunked"
        allowed_models: list[str] = []
        live_default = ""
        try:
            from core.preferences import get_preferences
            prefs = get_preferences(self.user)
            allowed_models = list(getattr(prefs, "allowed_transcription_models", None) or [])
            live_default = getattr(prefs, "transcription_model_live", "") or ""
            live_mode = getattr(prefs, "live_transcription_mode", "chunked") or "chunked"
        except Exception:
            pass

        def _is_live_capable(mid: str) -> bool:
            info = get_transcription_model_info(mid)
            return bool(info and info.supports_live_streaming)

        live_capable_allowed = [m for m in allowed_models if _is_live_capable(m)]

        meeting_model = (meeting.transcription_model or "").strip()
        if meeting_model and meeting_model in live_capable_allowed:
            model_id = meeting_model
        elif live_default and live_default in live_capable_allowed:
            model_id = live_default
        elif live_capable_allowed:
            model_id = live_capable_allowed[0]
        elif allowed_models:
            # No live-capable models in the allow-list. Keep the meeting's
            # choice so the user sees a meaningful error downstream rather
            # than silently switching to something else.
            model_id = meeting_model or allowed_models[0]
        else:
            model_id = getattr(settings, "TRANSCRIPTION_DEFAULT_MODEL", "openai/gpt-4o-mini-transcribe")

        # If no live-capable model is available (e.g. org only allows diarize),
        # force chunked mode — which will also fail, but at least with a clearer
        # error path. Realtime is physically impossible without a capable model.
        info = get_transcription_model_info(model_id)
        if live_mode != "chunked" and (info is None or not info.supports_live_streaming):
            logger.info(
                "_load_and_lock_meeting: selected model %s cannot stream live; "
                "forcing chunked mode for this session",
                model_id,
            )
            live_mode = "chunked"

        # Seed a prompt for the realtime session from meeting metadata so the
        # model biases toward proper nouns. Reuses the same builder the upload
        # path uses so both routes see identical context.
        prompt = ""
        if live_mode != "chunked":
            try:
                from meetings.services.audio_transcription import build_transcription_prompt
                tail = (meeting.transcript or "")[-1200:] if meeting.transcript else None
                prompt = build_transcription_prompt(meeting, prior_tail=tail)
            except Exception:
                prompt = ""

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
            "live_mode": live_mode,
            "forced_language": (meeting.forced_language or "") or None,
            "prompt": prompt,
        }

    @database_sync_to_async
    def _get_allowed_transcription_models(self) -> list[str]:
        from core.preferences import get_preferences
        prefs = get_preferences(self.user)
        return list(getattr(prefs, "allowed_transcription_models", None) or [])

    @database_sync_to_async
    def _persist_meeting_model(self, model_id: str) -> None:
        from .models import Meeting
        Meeting.objects.filter(pk=self.meeting_id).update(
            transcription_model=model_id,
        )

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
