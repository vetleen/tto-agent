"""Channels consumer tests for live transcription.

Uses ``channels.testing.WebsocketCommunicator`` to drive the consumer
without an actual browser. The Celery task is mocked so we don't hit
OpenAI; we exercise the WS protocol, lifecycle transitions, and state
preservation on disconnect / interruption.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
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
    # Pin chunk persistence to the local filesystem: with AWS_STORAGE_BUCKET_NAME
    # in the local .env the default storage is S3, so write_chunk_to_temp would
    # upload real objects to the bucket — and the S3 round-trip blows the
    # communicator's 1s receive timeout. Tests must not depend on (or write to)
    # remote storage.
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        },
    },
    MEETING_CHUNK_TEMP_DIR=tempfile.mkdtemp(prefix="wilfred-test-chunks-"),
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

    async def test_suspended_user_is_rejected(self):
        from accounts.models import Membership, Organization

        org = await database_sync_to_async(Organization.objects.create)(
            name="Acme", slug="acme-cons",
        )
        await database_sync_to_async(Membership.objects.create)(
            user=self.user, org=org, is_suspended=True,
        )
        comm = _make_communicator(self.meeting.uuid, self.user)
        connected, code = await comm.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4403)
        await comm.disconnect()

    async def test_non_owner_connect_does_not_mutate_meeting(self):
        # Regression: the status transition used to run BEFORE the ownership
        # check, letting a non-owner flip a victim's meeting to LIVE_TRANSCRIBING.
        thief = await database_sync_to_async(User.objects.create_user)(
            email="thief@example.com", password="pw",
        )
        comm = _make_communicator(self.meeting.uuid, thief)
        connected, code = await comm.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4403)
        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.status, Meeting.Status.DRAFT)
        self.assertIsNone(meeting.started_at)
        await comm.disconnect()

    async def test_second_active_meeting_rejected_with_4409(self):
        # Another meeting owned by the same user is already transcribing.
        await database_sync_to_async(Meeting.objects.create)(
            name="Busy", slug="m-busy-cons", created_by=self.user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        comm = _make_communicator(self.meeting.uuid, self.user)
        connected, code = await comm.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4409)
        # The meeting we tried to start stays untouched.
        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.status, Meeting.Status.DRAFT)
        await comm.disconnect()

    @patch("core.preferences.get_preferences")
    async def test_transcription_disabled_rejected_with_4402(self, mock_prefs):
        # Empty allow-list = transcription disabled for the user/org.
        mock_prefs.return_value.allowed_transcription_models = []
        mock_prefs.return_value.transcription_model_live = ""
        mock_prefs.return_value.live_transcription_mode = "chunked"
        comm = _make_communicator(self.meeting.uuid, self.user)
        connected, code = await comm.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4402)
        await comm.disconnect()

    @patch("meetings.services.chunks.cleanup_temp")
    @patch("meetings.tasks.transcribe_meeting_chunk_task")
    async def test_enqueue_failure_cleans_up_chunk(self, mock_task, mock_cleanup):
        # If dispatching the Celery task fails, the chunk already written to
        # storage must be cleaned up (it otherwise orphans) and the raw error
        # must not leak to the client.
        mock_task.delay.side_effect = RuntimeError("broker down")
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # started

        chunk = b"\x00\x01\x02\x03"
        await comm.send_to(text_data=json.dumps({
            "type": "chunk_meta",
            "segment_index": 0,
            "byte_length": len(chunk),
            "mime": "audio/webm",
        }))
        await comm.send_to(bytes_data=chunk)

        msg1 = json.loads(await comm.receive_from())
        self.assertEqual(msg1["type"], "segment.queued")
        msg2 = json.loads(await comm.receive_from())
        self.assertEqual(msg2["type"], "error")
        self.assertNotIn("broker down", msg2["message"])
        await asyncio.sleep(0.2)
        self.assertTrue(mock_cleanup.called)
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

    async def test_undersized_chunk_burst_is_dropped_after_first(self):
        """A *run* of tiny frames — 250ms streaming bursts leaking into chunked
        mode from a mis-torn-down realtime recorder — is the flood signature.
        We allow the first (it could be a legit short final chunk) and drop the
        rest before they reach _write_chunk / the Celery queue, so undecodable
        garbage can't storm ffprobe. Built directly so we can inspect internal
        state without a working ffmpeg child."""
        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "chunked"
        consumer._realtime_was_active_this_conn = False
        consumer._post_fallback_drop_armed = False
        consumer._pending_meta = []
        consumer._segments_total = 0
        consumer._undersized_chunk_streak = 0

        sent = []

        async def _fake_send(text_data=None, **kwargs):
            sent.append(json.loads(text_data))
        consumer.send = _fake_send

        write_calls: list[int] = []

        async def _fake_write_chunk(meta, raw):
            write_calls.append(meta["segment_index"])
            return f"/tmp/seg-{meta['segment_index']}"
        consumer._write_chunk = _fake_write_chunk

        enqueue_calls: list[int] = []

        async def _fake_enqueue(meta, temp_path):
            enqueue_calls.append(meta["segment_index"])
        consumer._enqueue_chunk_task = _fake_enqueue

        tiny = b"\x00" * 100  # well under the 8 KB MEETING_CHUNK_MIN_BYTES default
        for i in range(4):
            consumer._pending_meta.append({
                "segment_index": i,
                "byte_length": len(tiny),
                "mime": "audio/webm",
                "start_offset_seconds": 0.0,
            })
            await consumer._handle_binary_frame(tiny)

        # Only the first undersized frame is written/enqueued; frames 2-4 dropped.
        self.assertEqual(write_calls, [0])
        self.assertEqual(enqueue_calls, [0])
        self.assertEqual(consumer._segments_total, 1)
        queued = [m for m in sent if m["type"] == "segment.queued"]
        self.assertEqual(len(queued), 1)
        self.assertEqual(consumer._undersized_chunk_streak, 4)

    async def test_normal_chunk_resets_undersized_streak(self):
        """A legit full-size chunk clears the streak so a later short final
        chunk is still accepted — the guard must not drop normal traffic."""
        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "chunked"
        consumer._realtime_was_active_this_conn = False
        consumer._post_fallback_drop_armed = False
        consumer._pending_meta = []
        consumer._segments_total = 0
        consumer._undersized_chunk_streak = 0

        async def _fake_send(text_data=None, **kwargs):
            pass
        consumer.send = _fake_send

        enqueue_calls: list[int] = []

        async def _fake_enqueue(meta, temp_path):
            enqueue_calls.append(meta["segment_index"])
        consumer._enqueue_chunk_task = _fake_enqueue

        async def _fake_write_chunk(meta, raw):
            return f"/tmp/seg-{meta['segment_index']}"
        consumer._write_chunk = _fake_write_chunk

        big = b"\x00" * (16 * 1024)   # > 8 KB → resets the streak
        tiny = b"\x00" * 100          # legit short tail

        # tiny (streak=1, allowed) → big (resets to 0) → tiny (streak=1, allowed)
        for i, frame in enumerate((tiny, big, tiny)):
            consumer._pending_meta.append({
                "segment_index": i,
                "byte_length": len(frame),
                "mime": "audio/webm",
                "start_offset_seconds": 0.0,
            })
            await consumer._handle_binary_frame(frame)

        # All three are accepted; the big frame reset the streak in between.
        self.assertEqual(enqueue_calls, [0, 1, 2])
        self.assertEqual(consumer._segments_total, 3)
        self.assertEqual(consumer._undersized_chunk_streak, 1)

    async def test_post_fallback_drop_armed_drops_undersized_from_first(self):
        """With _post_fallback_drop_armed=True ALL undersized frames are dropped,
        including the first one.  This prevents a leaked streaming burst (the
        very first burst after a realtime→chunked fallback) from reaching Celery
        and creating an undecodable segment."""
        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "chunked"
        consumer._realtime_was_active_this_conn = True
        consumer._post_fallback_drop_armed = True
        consumer._pending_meta = []
        consumer._segments_total = 0
        consumer._undersized_chunk_streak = 0

        async def _fake_send(text_data=None, **kwargs):
            pass
        consumer.send = _fake_send

        write_calls: list[int] = []

        async def _fake_write_chunk(meta, raw):
            write_calls.append(meta["segment_index"])
            return "/tmp/fake"
        consumer._write_chunk = _fake_write_chunk

        enqueue_calls: list[int] = []

        async def _fake_enqueue(meta, temp_path):
            enqueue_calls.append(meta["segment_index"])
        consumer._enqueue_chunk_task = _fake_enqueue

        tiny = b"\x00" * 100  # well under 8 KB

        # Send two tiny (leaked) frames — both should be dropped.
        for i in range(2):
            consumer._pending_meta.append({
                "segment_index": i + 500,  # inflated client-counted index
                "byte_length": len(tiny),
                "mime": "audio/webm",
                "start_offset_seconds": 0.0,
            })
            await consumer._handle_binary_frame(tiny)

        self.assertEqual(write_calls, [], "no writes expected: both frames should be dropped")
        self.assertEqual(enqueue_calls, [], "no enqueues expected")
        self.assertEqual(consumer._segments_total, 0)

    async def test_post_fallback_chunk_uses_server_allocated_index(self):
        """After realtime was active, chunked frames get a server-allocated index
        rather than trusting the (potentially inflated) client-supplied one.
        This prevents a leaked streaming burst from poisoning the max+1 allocator."""
        # Pre-create two ready segments so the allocator returns 2 next.
        await database_sync_to_async(MeetingTranscriptSegment.objects.create)(
            meeting=self.meeting,
            segment_index=0,
            text="Hello.",
            status=MeetingTranscriptSegment.Status.READY,
        )
        await database_sync_to_async(MeetingTranscriptSegment.objects.create)(
            meeting=self.meeting,
            segment_index=1,
            text="World.",
            status=MeetingTranscriptSegment.Status.READY,
        )

        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "chunked"
        consumer._realtime_was_active_this_conn = True
        consumer._post_fallback_drop_armed = False
        consumer._pending_meta = []
        consumer._segments_total = 0
        consumer._undersized_chunk_streak = 0

        async def _fake_send(text_data=None, **kwargs):
            pass
        consumer.send = _fake_send

        write_calls: list[int] = []

        async def _fake_write_chunk(meta, raw):
            write_calls.append(meta["segment_index"])
            return "/tmp/fake"
        consumer._write_chunk = _fake_write_chunk

        enqueue_calls: list[int] = []

        async def _fake_enqueue(meta, temp_path):
            enqueue_calls.append(meta["segment_index"])
        consumer._enqueue_chunk_task = _fake_enqueue

        # Client sends an inflated index (271) — server should allocate 2 instead.
        big = b"\x00" * (16 * 1024)
        consumer._pending_meta.append({
            "segment_index": 271,
            "byte_length": len(big),
            "mime": "audio/webm",
            "start_offset_seconds": 0.0,
        })
        await consumer._handle_binary_frame(big)

        self.assertEqual(write_calls, [2], "write should use server-allocated index 2, not 271")
        self.assertEqual(enqueue_calls, [2], "enqueue should use server-allocated index 2, not 271")

    async def test_pure_chunked_session_uses_client_index_unchanged(self):
        """A pure chunked session (_realtime_was_active_this_conn=False) keeps the
        existing behaviour: client-supplied index is used, first undersized allowed."""
        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "chunked"
        consumer._realtime_was_active_this_conn = False
        consumer._post_fallback_drop_armed = False
        consumer._pending_meta = []
        consumer._segments_total = 0
        consumer._undersized_chunk_streak = 0

        async def _fake_send(text_data=None, **kwargs):
            pass
        consumer.send = _fake_send

        enqueue_calls: list[int] = []

        async def _fake_enqueue(meta, temp_path):
            enqueue_calls.append(meta["segment_index"])
        consumer._enqueue_chunk_task = _fake_enqueue

        async def _fake_write_chunk(meta, raw):
            return "/tmp/fake"
        consumer._write_chunk = _fake_write_chunk

        tiny = b"\x00" * 100  # undersized — should be allowed as first
        consumer._pending_meta.append({
            "segment_index": 7,  # arbitrary client index
            "byte_length": len(tiny),
            "mime": "audio/webm",
            "start_offset_seconds": 0.0,
        })
        await consumer._handle_binary_frame(tiny)

        # First undersized frame is allowed in pure chunked mode (original behaviour).
        self.assertEqual(enqueue_calls, [7], "first undersized frame should pass with client index 7")
        self.assertEqual(consumer._undersized_chunk_streak, 1)

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
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "realtime"
        consumer._realtime_failure_count = 0
        consumer._realtime_permanently_disabled = False
        consumer._realtime_was_active_this_conn = False
        consumer._post_fallback_drop_armed = False
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

    async def test_fatal_error_tears_down_pipe_and_session(self):
        """A fatal upstream SessionError must release the ffmpeg pipe + pump
        task + session (not leak them until disconnect). _teardown_realtime is
        invoked from inside the consumer task, so it must skip its own task and
        still cancel the pump + close the pipe/session."""
        from meetings.services.realtime_session import SessionError, SessionStatus

        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "realtime"
        consumer._realtime_failure_count = 0
        consumer._realtime_permanently_disabled = False
        consumer._realtime_was_active_this_conn = True
        consumer._post_fallback_drop_armed = False
        consumer._interruption_started_at = None
        consumer._realtime_started = True
        consumer._total_pcm_bytes = 0

        sent = []

        async def _fake_send(text_data=None, **kwargs):
            sent.append(json.loads(text_data))
        consumer.send = _fake_send

        calls = {"session_finalize": 0, "session_aclose": 0, "pipe_aclose": 0}

        class _FakeSession:
            async def events(self):
                yield SessionError(code="invalid_api_key", message="bad key", fatal=True)
                yield SessionStatus(state="disconnected")  # never reached

            async def finalize(self, timeout=2.0):
                calls["session_finalize"] += 1

            async def aclose(self):
                calls["session_aclose"] += 1

        class _FakePipe:
            stats = None

            async def aclose(self):
                calls["pipe_aclose"] += 1

        consumer._realtime_session = _FakeSession()
        consumer._pcm_pipe = _FakePipe()

        async def _park():
            await asyncio.sleep(3600)
        pump_task = asyncio.create_task(_park())
        consume_task = asyncio.create_task(consumer._consume_realtime_events())
        # The consumer's own task is in the realtime task list — teardown must
        # not try to cancel/await itself.
        consumer._realtime_tasks = [pump_task, consume_task]

        await asyncio.wait_for(consume_task, timeout=2.0)

        # Resources released, not leaked.
        self.assertEqual(calls["session_aclose"], 1)
        self.assertEqual(calls["pipe_aclose"], 1)
        self.assertIsNone(consumer._realtime_session)
        self.assertIsNone(consumer._pcm_pipe)
        self.assertFalse(consumer._realtime_started)
        self.assertEqual(consumer._realtime_tasks, [])
        # The parked pump task was cancelled, not left running.
        self.assertTrue(pump_task.cancelled() or pump_task.done())
        # Client was told to fall back to chunked.
        self.assertTrue(any(m["type"] == "live_mode_changed" for m in sent))


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
