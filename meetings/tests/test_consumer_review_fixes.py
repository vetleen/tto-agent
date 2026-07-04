"""Consumer tests for the review-batch fixes.

Covers, by driving handler methods directly with fakes:
- C14: malformed/non-finite start_offset_seconds is rejected, not crashing.
- C27: _pending_meta is bounded.
- C18: set_model is guarded during an active/realtime session.
- C16: a live connect is refused while an upload transcription is in flight.
- C10: a session-build failure unwinds the ffmpeg pipe.
- C7:  a Stop whose finalize fails does not report success and re-arms disconnect.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from meetings.consumers import MeetingTranscribeConsumer
from meetings.models import Meeting

User = get_user_model()


def _consumer_with_send(user=None):
    consumer = MeetingTranscribeConsumer()
    consumer.user = user
    sent: list[dict] = []
    closed: list[int | None] = []

    async def _send(text_data=None, **kwargs):
        if text_data:
            sent.append(json.loads(text_data))

    async def _close(code=None):
        closed.append(code)

    consumer.send = _send
    consumer.close = _close
    return consumer, sent, closed


class ChunkMetaValidationTests(TransactionTestCase):
    async def _handle(self, payload):
        consumer, sent, _closed = _consumer_with_send()
        consumer._realtime_mode = "chunked"
        consumer._pending_meta = []
        await consumer._handle_chunk_meta(payload)
        return consumer, sent

    async def test_non_numeric_start_offset_emits_error_not_crash(self):
        consumer, sent = await self._handle(
            {"segment_index": 0, "byte_length": 100, "start_offset_seconds": "abc"}
        )
        self.assertTrue(any(m["type"] == "error" for m in sent))
        self.assertEqual(consumer._pending_meta, [])

    async def test_nan_start_offset_rejected(self):
        consumer, sent = await self._handle(
            {"segment_index": 0, "byte_length": 100, "start_offset_seconds": "NaN"}
        )
        self.assertTrue(any(m["type"] == "error" for m in sent))
        self.assertEqual(consumer._pending_meta, [])

    async def test_valid_offset_is_accepted(self):
        consumer, _sent = await self._handle(
            {"segment_index": 0, "byte_length": 100, "start_offset_seconds": 1.5}
        )
        self.assertEqual(len(consumer._pending_meta), 1)
        self.assertEqual(consumer._pending_meta[0]["start_offset_seconds"], 1.5)


class PendingMetaCapTests(TransactionTestCase):
    async def test_overflow_closes_with_4400(self):
        from meetings.consumers import _PENDING_META_MAX

        consumer, sent, closed = _consumer_with_send()
        consumer._realtime_mode = "chunked"
        consumer._pending_meta = [{} for _ in range(_PENDING_META_MAX)]
        await consumer._handle_chunk_meta(
            {"segment_index": 0, "byte_length": 100}
        )
        self.assertTrue(any(m["type"] == "error" for m in sent))
        self.assertIn(4400, closed)


class SetModelGuardTests(TransactionTestCase):
    async def test_rejected_after_realtime_started(self):
        consumer, sent, _closed = _consumer_with_send()
        consumer._realtime_started = True
        consumer._realtime_mode = "realtime"
        await consumer._handle_set_model({"model_id": "openai/gpt-4o-transcribe"})
        self.assertTrue(any(m["type"] == "error" for m in sent))

    async def test_rejects_non_streaming_model_in_realtime_mode(self):
        consumer, sent, _closed = _consumer_with_send()
        consumer._realtime_started = False
        consumer._realtime_mode = "realtime"

        async def _allowed():
            return ["openai/gpt-4o-transcribe-diarize"]

        consumer._get_allowed_transcription_models = _allowed
        await consumer._handle_set_model(
            {"model_id": "openai/gpt-4o-transcribe-diarize"}
        )
        self.assertTrue(any("does not support live streaming" in m.get("message", "") for m in sent))


class ConnectRefusedDuringUploadTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="c16@example.com", password="pw")

    async def test_refuses_live_connect_while_upload_in_flight(self):
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Uploading",
            slug="m-c16",
            created_by=self.user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
            transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
            # chunks_total stays 0 for single-chunk uploads — the old 0<0 guard
            # never fired here.
            transcription_chunks_total=0,
            transcription_chunks_done=0,
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        result = await consumer._load_and_lock_meeting(meeting.uuid)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], 4409)


class StartSessionPipeLeakTests(TransactionTestCase):
    async def test_build_failure_closes_pipe(self):
        from meetings.services import pcm_pipe as pcm_pipe_mod
        from meetings.services import realtime_session as rt_mod

        consumer, _sent, _closed = _consumer_with_send()
        consumer._model_id = "openai/gpt-4o-transcribe"
        consumer._realtime_language = ""

        closed_flag = {"pipe": False}

        class _FakePipe:
            def __init__(self, *a, **k):
                pass

            async def start(self):
                pass

            async def aclose(self):
                closed_flag["pipe"] = True

        def _boom(**kwargs):
            raise rt_mod.UnsupportedModelError("no live")

        with patch.object(pcm_pipe_mod, "PcmPipe", _FakePipe), \
                patch.object(rt_mod, "build_realtime_session", _boom):
            with self.assertRaises(rt_mod.UnsupportedModelError):
                await consumer._start_realtime_session("audio/webm")

        self.assertTrue(closed_flag["pipe"])
        self.assertIsNone(consumer._pcm_pipe)


class StopFinalizeFailureTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="c7@example.com", password="pw")

    async def test_finalize_failure_does_not_report_stopped(self):
        consumer, sent, closed = _consumer_with_send(self.user)
        consumer.meeting_id = 1
        consumer._realtime_started = False
        consumer._segments_total = 0
        consumer._segments_failed = 0

        async def _boom(*, interrupted):
            raise RuntimeError("db down")

        consumer._finalize_meeting = _boom
        await consumer._handle_stop()

        # No success frame; an error frame instead.
        self.assertFalse(any(m["type"] == "stopped" for m in sent))
        self.assertTrue(any(m["type"] == "error" for m in sent))
        # Stop is un-claimed so disconnect() retries finalize.
        self.assertFalse(consumer._stop_requested)
