"""Channels consumer tests for live transcription.

Uses ``channels.testing.WebsocketCommunicator`` to drive the consumer
without an actual browser. The Celery task is mocked so we don't hit
OpenAI; we exercise the WS protocol, lifecycle transitions, and state
preservation on disconnect / interruption.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from meetings import consumers as consumers_module
from meetings.consumers import MeetingTranscribeConsumer
from meetings.models import Meeting, MeetingTranscriptSegment

User = get_user_model()


def _make_communicator(meeting_uuid, user):
    """Build a WebsocketCommunicator with the user injected into scope."""
    application = MeetingTranscribeConsumer.as_asgi()
    communicator = WebsocketCommunicator(
        application, f"/ws/meetings/{meeting_uuid}/transcribe/"
    )
    # The URLRouter normally populates url_route; we must do it manually here.
    communicator.scope["url_route"] = {"kwargs": {"meeting_uuid": str(meeting_uuid)}}
    communicator.scope["user"] = user
    return communicator


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
)
class MeetingTranscribeConsumerTests(TransactionTestCase):
    """TransactionTestCase because we use database_sync_to_async.

    These tests cover the chunked (Celery) live-transcription path. The
    realtime path has its own test module. We pin the live mode to
    "chunked" via a user-level preference so the consumer takes the
    chunked branch regardless of the shipping default.
    """

    def setUp(self):
        from accounts.models import UserSettings
        Meeting.objects.all().delete()
        self.user = User.objects.create_user(email="cons@example.com", password="pw")
        # Opt this user into chunked mode — the shipping default is
        # realtime_with_fallback which would otherwise route binary frames
        # to the ffmpeg/Realtime path that these tests aren't exercising.
        UserSettings.objects.update_or_create(
            user=self.user,
            defaults={"preferences": {"live_transcription_mode": "chunked"}},
        )
        self.meeting = Meeting.objects.create(
            name="M", slug="m-cons", created_by=self.user,
        )

    async def test_anonymous_is_rejected(self):
        from django.contrib.auth.models import AnonymousUser

        comm = _make_communicator(self.meeting.uuid, AnonymousUser())
        connected, code_or_subproto = await comm.connect()
        self.assertFalse(connected)
        await comm.disconnect()

    async def test_owner_connects_and_receives_started(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        connected, _ = await comm.connect()
        self.assertTrue(connected)
        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "started")
        self.assertEqual(msg["meeting_id"], str(self.meeting.uuid))
        self.assertEqual(msg["segment_index_base"], 0)

        # Meeting should now be in LIVE_TRANSCRIBING.
        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.status, Meeting.Status.LIVE_TRANSCRIBING)
        self.assertIsNotNone(meeting.started_at)
        await comm.disconnect()

    async def test_non_owner_is_rejected(self):
        other = await database_sync_to_async(User.objects.create_user)(
            email="not-owner@example.com", password="pw",
        )
        comm = _make_communicator(self.meeting.uuid, other)
        connected, _ = await comm.connect()
        self.assertFalse(connected)
        await comm.disconnect()

    @patch("meetings.tasks.transcribe_meeting_chunk_task")
    async def test_chunk_meta_plus_binary_enqueues_task(self, mock_task):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        chunk = b"\x00\x01\x02\x03"
        await comm.send_to(text_data=json.dumps({
            "type": "chunk_meta",
            "segment_index": 0,
            "byte_length": len(chunk),
            "mime": "audio/webm;codecs=opus",
            "start_offset_seconds": 0.0,
        }))
        await comm.send_to(bytes_data=chunk)

        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "segment.queued")
        self.assertEqual(msg["segment_index"], 0)

        # The consumer dispatches the celery task via database_sync_to_async,
        # which runs in a thread pool. segment.queued is sent BEFORE the
        # dispatch, so we yield briefly to let the thread pool worker run.
        await asyncio.sleep(0.3)

        mock_task.delay.assert_called_once()
        kwargs = mock_task.delay.call_args.kwargs
        self.assertEqual(kwargs["meeting_id"], self.meeting.id)
        self.assertEqual(kwargs["segment_index"], 0)
        self.assertEqual(kwargs["mime"], "audio/webm;codecs=opus")

    @patch("meetings.tasks.transcribe_meeting_chunk_task")
    async def test_byte_length_mismatch_emits_error(self, mock_task):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # started

        await comm.send_to(text_data=json.dumps({
            "type": "chunk_meta",
            "segment_index": 0,
            "byte_length": 10,
            "mime": "audio/webm",
        }))
        await comm.send_to(bytes_data=b"\x00\x01")  # too short

        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "error")
        mock_task.delay.assert_not_called()

    async def test_stop_finalizes_meeting_to_ready(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # started

        await comm.send_to(text_data=json.dumps({"type": "stop"}))
        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "stopped")

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.status, Meeting.Status.READY)
        self.assertIsNotNone(meeting.ended_at)
        await comm.disconnect()

    async def test_disconnect_without_stop_marks_interrupted(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # started

        await comm.disconnect()

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.status, Meeting.Status.INTERRUPTED)

    async def test_set_model_persists_allowed_choice(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "set_model",
            "model_id": "openai/gpt-4o-transcribe",
        }))
        # Give the database_sync_to_async write a tick to complete.
        await asyncio.sleep(0.2)

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.transcription_model, "openai/gpt-4o-transcribe")
        await comm.disconnect()

    async def test_set_model_rejects_unknown_choice(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "set_model",
            "model_id": "evil/not-a-real-model",
        }))
        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "error")
        self.assertIn("not allowed", msg["message"])

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.transcription_model, "")
        await comm.disconnect()

    async def test_heartbeat_ping_on_idle_socket(self):
        """The consumer sends a ping at a fixed cadence so idle sockets don't
        get torn down by an intermediary (Heroku router, proxy, etc.).
        Shortens the interval to keep the test fast.
        """
        with patch.object(consumers_module, "MEETING_WS_HEARTBEAT_SECONDS", 0.1):
            comm = _make_communicator(self.meeting.uuid, self.user)
            await comm.connect()
            msg = json.loads(await comm.receive_from())
            self.assertEqual(msg["type"], "started")

            ping = json.loads(await comm.receive_from(timeout=2))
            self.assertEqual(ping["type"], "ping")
            await comm.disconnect()

    async def test_resume_continues_segment_index_base(self):
        # Pre-populate one segment so the resume reconnect picks up at index 1.
        await database_sync_to_async(MeetingTranscriptSegment.objects.create)(
            meeting=self.meeting,
            segment_index=0,
            text="prior",
            status=MeetingTranscriptSegment.Status.READY,
        )
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "started")
        self.assertEqual(msg["segment_index_base"], 1)
        await comm.disconnect()

    async def test_interruption_marker_inserts_segment_and_emits_ready(self):
        """interruption_marker → marker segment + segment.ready frame."""
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "interruption_marker",
            "gap_seconds": 17,
        }))

        msg = json.loads(await comm.receive_from(timeout=2))
        self.assertEqual(msg["type"], "segment.ready")
        self.assertIn("17 seconds", msg["text"])
        self.assertEqual(msg["transcription_model"], "_interrupt_marker")

        # Segment row was persisted with the marker's distinguishing
        # transcription_model so we can audit/filter these later.
        segs = await database_sync_to_async(list)(
            MeetingTranscriptSegment.objects.filter(meeting_id=self.meeting.id)
            .order_by("segment_index")
        )
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].transcription_model, "_interrupt_marker")
        self.assertEqual(segs[0].status, MeetingTranscriptSegment.Status.READY)
        self.assertIn("17 seconds", segs[0].text)
        await comm.disconnect()

    async def test_interruption_marker_below_threshold_is_ignored(self):
        """Sub-threshold gaps (e.g. 2s) don't clutter the transcript."""
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "interruption_marker",
            "gap_seconds": 2,
        }))
        # Give the consumer a moment to process; no frame should arrive.
        await asyncio.sleep(0.2)
        self.assertTrue(await comm.receive_nothing())

        count = await database_sync_to_async(
            MeetingTranscriptSegment.objects.filter(meeting_id=self.meeting.id).count
        )()
        self.assertEqual(count, 0)
        await comm.disconnect()

    async def test_interruption_marker_rejects_non_numeric_gap(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "interruption_marker",
            "gap_seconds": "soon",
        }))
        msg = json.loads(await comm.receive_from(timeout=2))
        self.assertEqual(msg["type"], "error")
        self.assertIn("gap_seconds", msg["message"])
        await comm.disconnect()

    async def test_fall_back_to_chunked_emits_live_mode_changed(self):
        """First fallback emits live_mode_changed with permanent=False; second
        emits with permanent=True. Bypasses the realtime infra by calling the
        helper directly via a tiny consumer harness — covers the protocol
        contract without needing a working ffmpeg child."""
        from meetings.consumers import (
            MeetingTranscribeConsumer,
            REALTIME_FAILURE_BUDGET,
        )
        sent = []

        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer._realtime_mode = "realtime"
        consumer._realtime_failure_count = 0
        consumer._realtime_permanently_disabled = False
        consumer._interruption_started_at = None

        async def _fake_send(text_data=None, **kwargs):
            sent.append(json.loads(text_data))
        consumer.send = _fake_send

        await consumer._fall_back_to_chunked(reason="realtime_unstable")
        # Re-arm the "was realtime" precondition so a second flip can fire
        # — in production the second failure happens after a successful
        # realtime restart, not here.
        consumer._realtime_mode = "realtime"
        await consumer._fall_back_to_chunked(reason="realtime_unstable")

        events = [m for m in sent if m["type"] == "live_mode_changed"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["live_mode"], "chunked")
        self.assertFalse(events[0]["permanent"])
        # Second strike pins us to chunked.
        self.assertTrue(events[1]["permanent"])
        self.assertEqual(events[1]["reason"], "realtime_unstable_fallback")
        self.assertGreaterEqual(consumer._realtime_failure_count, REALTIME_FAILURE_BUDGET)
        self.assertTrue(consumer._realtime_permanently_disabled)


class RecomputeIncludesMarkerTests(TransactionTestCase):
    """A marker segment with status=READY must show up inline in the joined
    transcript, ordered by segment_index. This guards the contract used by
    the live transcription path."""

    def setUp(self):
        Meeting.objects.all().delete()
        self.user = User.objects.create_user(email="rmark@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="M", slug="m-rmark", created_by=self.user,
        )

    def test_marker_segment_renders_between_real_segments(self):
        from meetings.services.chunks import recompute_meeting_transcript

        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=0, text="Hello world.",
            status=MeetingTranscriptSegment.Status.READY,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=1,
            text="[Transcription was interrupted for 14 seconds]",
            transcription_model="_interrupt_marker",
            status=MeetingTranscriptSegment.Status.READY,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=2, text="Welcome back.",
            status=MeetingTranscriptSegment.Status.READY,
        )

        joined = recompute_meeting_transcript(self.meeting.id)
        self.assertEqual(
            joined,
            "Hello world.\n\n"
            "[Transcription was interrupted for 14 seconds]\n\n"
            "Welcome back.",
        )
