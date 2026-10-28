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


async def _rejected_code(comm, timeout=1):
    """Connect and return the application close code of a refused handshake.

    Refusals accept first and then close: a close *before* accept reaches a
    real browser as a bare HTTP 403 handshake failure (CloseEvent 1006), so
    the client could never show its per-code message. The communicator
    therefore reports a successful connect followed by a ``websocket.close``
    frame carrying the code.
    """
    connected, _ = await comm.connect(timeout=timeout)
    assert connected, "refusal must accept before closing (pre-accept close = HTTP 403)"
    close = await comm.receive_output(timeout)
    assert close["type"] == "websocket.close", close
    return close.get("code")


def _patch_chunked_mode(test_case):
    """Pin the resolved live mode to ``chunked`` for the duration of a test.

    The live-transcription-path picker was removed — resolution is always
    ``realtime_with_fallback``, so these chunked-path tests can no longer force
    the branch via a user preference. Instead we wrap the real ``get_preferences``
    (keeping every other resolved value intact) and override just the mode, so
    the consumer takes the chunked branch and doesn't reach for the ffmpeg /
    Realtime path these tests don't exercise. The patcher is stopped on cleanup.
    """
    import core.preferences as prefs_mod

    real_get_preferences = prefs_mod.get_preferences

    def _chunked(user):
        prefs = real_get_preferences(user)
        prefs.live_transcription_mode = "chunked"
        return prefs

    patcher = patch("core.preferences.get_preferences", side_effect=_chunked)
    patcher.start()
    test_case.addCleanup(patcher.stop)


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
    realtime path has its own test module. We pin the resolved live mode to
    "chunked" (see _patch_chunked_mode) so the consumer takes the chunked
    branch regardless of the shipping default (realtime_with_fallback).
    """

    def setUp(self):
        Meeting.objects.all().delete()
        self.user = User.objects.create_user(email="cons@example.com", password="pw")
        # Force chunked routing — the shipping default is realtime_with_fallback,
        # which would otherwise route binary frames to the ffmpeg/Realtime path
        # that these tests aren't exercising.
        _patch_chunked_mode(self)
        self.meeting = Meeting.objects.create(
            name="M", slug="m-cons", created_by=self.user,
        )

    async def test_anonymous_is_rejected(self):
        from django.contrib.auth.models import AnonymousUser

        comm = _make_communicator(self.meeting.uuid, AnonymousUser())
        self.assertEqual(await _rejected_code(comm), 4401)
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
        self.assertEqual(await _rejected_code(comm), 4403)
        await comm.disconnect()

    async def test_refusal_accepts_then_closes_with_app_code(self):
        # Regression guard for the browser-visible close code: Daphne turns a
        # close() issued before accept() into an HTTP 403 handshake rejection,
        # so the browser's CloseEvent carried 1006 and TERMINAL_CLOSE_MESSAGES
        # never showed. The handshake must complete and the close frame must
        # carry the application code.
        other = await database_sync_to_async(User.objects.create_user)(
            email="not-owner-2@example.com", password="pw",
        )
        comm = _make_communicator(self.meeting.uuid, other)
        connected, subprotocol = await comm.connect()
        self.assertTrue(connected)
        self.assertIsNone(subprotocol)
        close = await comm.receive_output(1)
        self.assertEqual(close["type"], "websocket.close")
        self.assertEqual(close["code"], 4403)
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
        self.assertEqual(await _rejected_code(comm), 4403)
        await comm.disconnect()

    async def test_non_owner_connect_does_not_mutate_meeting(self):
        # Regression: the status transition used to run BEFORE the ownership
        # check, letting a non-owner flip a victim's meeting to LIVE_TRANSCRIBING.
        thief = await database_sync_to_async(User.objects.create_user)(
            email="thief@example.com", password="pw",
        )
        comm = _make_communicator(self.meeting.uuid, thief)
        self.assertEqual(await _rejected_code(comm), 4403)
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
        self.assertEqual(await _rejected_code(comm), 4409)
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
        self.assertEqual(await _rejected_code(comm), 4402)
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

    async def test_set_language_persists_and_acks(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "set_language",
            "language": "no",
        }))
        msg = json.loads(await comm.receive_from(timeout=2))
        self.assertEqual(msg["type"], "language_changed")
        self.assertEqual(msg["language"], "no")
        self.assertEqual(msg["effective"], "no")

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.forced_language, "no")
        await comm.disconnect()

    async def test_set_language_auto_acks_with_empty_effective(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "set_language",
            "language": "auto",
        }))
        msg = json.loads(await comm.receive_from(timeout=2))
        self.assertEqual(msg["type"], "language_changed")
        self.assertEqual(msg["language"], "auto")
        self.assertEqual(msg["effective"], "")

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.forced_language, "auto")
        await comm.disconnect()

    async def test_set_language_rejects_unknown_code(self):
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "set_language",
            "language": "xx",
        }))
        msg = json.loads(await comm.receive_from(timeout=2))
        self.assertEqual(msg["type"], "error")
        self.assertIn("unsupported", msg["message"])

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=self.meeting.pk)
        self.assertEqual(meeting.forced_language, "")
        await comm.disconnect()

    async def test_extend_auto_stop_is_acknowledged_silently(self):
        # Auto-stop is client-enforced; the server only logs the new limit.
        comm = _make_communicator(self.meeting.uuid, self.user)
        await comm.connect()
        await comm.receive_from()  # consume "started"

        await comm.send_to(text_data=json.dumps({
            "type": "extend_auto_stop",
            "minutes": 30,
        }))
        await asyncio.sleep(0.2)
        self.assertTrue(await comm.receive_nothing())
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

    async def test_session_ceiling_force_stops(self):
        """The server-side session ceiling (MEETING_AUTO_STOP_MAX_SECONDS)
        force-stops a session that outlives it — the cost backstop for a hung
        client whose socket (kept alive by our own heartbeat) would otherwise
        stream audio forever. Finalizes like a user Stop: error + stopped
        frames, meeting READY.
        """
        with patch.object(consumers_module, "MEETING_WS_HEARTBEAT_SECONDS", 0.05):
            with override_settings(MEETING_AUTO_STOP_MAX_SECONDS=0.01):
                comm = _make_communicator(self.meeting.uuid, self.user)
                connected, _ = await comm.connect()
                self.assertTrue(connected)
                msg = json.loads(await comm.receive_from())
                self.assertEqual(msg["type"], "started")

                # First heartbeat tick finds the ceiling already passed: the
                # deterministic sequence is error then stopped (no ping — the
                # ceiling check runs before the ping send).
                msg = json.loads(await comm.receive_from(timeout=2))
                self.assertEqual(msg["type"], "error")
                self.assertIn("Maximum session length", msg["message"])
                msg = json.loads(await comm.receive_from(timeout=2))
                self.assertEqual(msg["type"], "stopped")

                meeting = await database_sync_to_async(Meeting.objects.get)(
                    pk=self.meeting.pk
                )
                self.assertEqual(meeting.status, Meeting.Status.READY)
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

    async def test_session_activity_is_forwarded_to_client(self):
        """Upstream VAD lifecycle reaches the client as ``session.activity`` so
        its stall watchdog can tell a long open utterance (no text yet, by
        design) from a dead pipeline."""
        from meetings.services.realtime_session import SessionActivity, SessionStatus

        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = self.meeting.id
        consumer.meeting_uuid = str(self.meeting.uuid)
        consumer._realtime_mode = "realtime"
        consumer._realtime_started = True
        consumer._total_pcm_bytes = 0
        consumer._realtime_tasks = []
        consumer._realtime_consume_task = None

        sent = []

        async def _fake_send(text_data=None, **kwargs):
            sent.append(json.loads(text_data))
        consumer.send = _fake_send

        class _FakeSession:
            async def events(self):
                yield SessionActivity(kind="speech_started")
                yield SessionActivity(kind="committed")
                yield SessionStatus(state="disconnected")

            async def finalize(self, timeout=2.0):
                pass

            def pending_event_count(self):
                return 0

            async def aclose(self):
                pass

        class _FakePipe:
            stats = None

            async def aclose(self):
                pass

        consumer._realtime_session = _FakeSession()
        consumer._pcm_pipe = _FakePipe()

        await asyncio.wait_for(consumer._consume_realtime_events(), timeout=2.0)

        kinds = [m["kind"] for m in sent if m["type"] == "session.activity"]
        self.assertEqual(kinds, ["speech_started", "committed"])


class RealtimePromptSeedingTests(TransactionTestCase):
    """The realtime session must be seeded with NO prompt.

    gpt-4o-transcribe treats the realtime ``prompt`` as a token-biasing hint,
    not an instruction, so on short/quiet VAD utterances it regurgitates the
    prompt verbatim as fake "utterances". 5b1b241 dropped the prior-transcript
    tail for this reason but kept the meeting metadata; prod evidence (meeting
    da1c8903, 2026-06-30) showed the metadata line itself leaks — e.g.
    "Meeting: <name>" and "<name>" appearing repeatedly in the transcript. So
    the realtime path now seeds no prompt at all. The chunked/upload path still
    uses a metadata+tail prompt; full ~30s chunks anchor the model so it
    transcribes rather than echoes.
    """

    def setUp(self):
        self.user = User.objects.create_user(email="rtprompt@example.com", password="pw")

    async def _seed_prompt_for(self, *, transcript: str) -> dict:
        import types

        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Innovasjon Norge Sync",
            slug="rt-prompt-seed",
            created_by=self.user,
            transcript=transcript,
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user

        prefs = types.SimpleNamespace(
            allowed_transcription_models=["openai/gpt-4o-transcribe"],
            transcription_model_live="openai/gpt-4o-transcribe",
            live_transcription_mode="realtime",
        )
        with patch("core.preferences.get_preferences", return_value=prefs):
            return await consumer._load_and_lock_meeting(meeting.uuid)

    async def test_load_and_lock_carries_no_prompt(self):
        # The dead prompt plumbing was removed: the result dict no longer carries
        # a prompt key at all (there is nothing to seed).
        result = await self._seed_prompt_for(
            transcript="...REGURGITATION_CANARY and then the deadline would slip.",
        )
        self.assertTrue(result["ok"])
        self.assertNotEqual(result["live_mode"], "chunked")
        self.assertNotIn("prompt", result)

    async def _load_with_participants(self, slug):
        import types

        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Innovasjon Norge Sync", slug=slug, created_by=self.user,
            participants="Vetle Engesbak, Kari Nordmann", agenda="Review the EPO opposition.",
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        prefs = types.SimpleNamespace(
            allowed_transcription_models=["openai/gpt-transcribe", "openai/gpt-live-transcribe"],
            transcription_model_live="openai/gpt-live-transcribe",
            live_transcription_mode="realtime",
        )
        with patch("core.preferences.get_preferences", return_value=prefs):
            return await consumer._load_and_lock_meeting(meeting.uuid)

    async def test_live_keywords_seeded_from_participants_never_the_name(self):
        result = await self._load_with_participants("rt-keywords")
        self.assertEqual(result["model_id"], "openai/gpt-live-transcribe")
        self.assertEqual(result["keywords"], ["Vetle Engesbak", "Kari Nordmann", "EPO"])

    @override_settings(MEETING_REALTIME_KEYWORDS_ENABLED=False)
    async def test_live_keywords_kill_switch(self):
        result = await self._load_with_participants("rt-keywords-off")
        self.assertEqual(result["keywords"], [])

    async def test_realtime_session_built_without_prompt(self):
        # gpt-4o-transcribe regurgitates a realtime `prompt` verbatim on short
        # utterances, so the session must be built with NO prompt. (Regression
        # guard for meeting da1c8903, 2026-06-30.)
        from meetings.services import pcm_pipe as pcm_pipe_mod
        from meetings.services import realtime_session as rt_mod

        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Innovasjon Norge Sync",
            slug="rt-prompt-build",
            created_by=self.user,
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        consumer.meeting_id = meeting.id
        consumer.meeting_uuid = str(meeting.uuid)
        consumer._model_id = "openai/gpt-4o-transcribe"
        consumer._realtime_language = "no"
        consumer._realtime_keywords = ["Vetle Engesbak"]
        consumer._total_pcm_bytes = 0  # normally set in connect()

        captured: dict = {}

        class _FakePipe:
            stats = None

            def __init__(self, *a, **k):
                pass

            async def start(self):
                pass

            async def read_frames(self):
                if False:
                    yield b""

            async def aclose(self):
                pass

        class _FakeSession:
            async def connect(self):
                pass

            async def events(self):
                await asyncio.sleep(3600)
                if False:
                    yield None

            async def send_pcm(self, frame):
                pass

            def pending_event_count(self):
                return 0

            async def finalize(self, timeout=2.0):
                pass

            async def aclose(self):
                pass

        def _fake_build(**kwargs):
            captured.update(kwargs)
            return _FakeSession()

        with patch.object(pcm_pipe_mod, "PcmPipe", _FakePipe), \
                patch.object(rt_mod, "build_realtime_session", _fake_build):
            await consumer._start_realtime_session("audio/webm")
            try:
                # No prompt is passed, so the session-layer default (None) is used.
                self.assertIsNone(captured.get("prompt"))
                self.assertEqual(captured.get("model_id"), "openai/gpt-4o-transcribe")
                self.assertEqual(captured.get("language"), "no")
                self.assertEqual(captured.get("keywords"), ["Vetle Engesbak"])
            finally:
                await consumer._teardown_realtime()


class RealtimeLanguageResolutionTests(TransactionTestCase):
    """``_load_and_lock_meeting`` resolves the realtime language hint: the
    meeting's own ``forced_language`` if set, else the resolved default-language
    preference. Covers both the detail-page Start and the overview auto-start
    since both use this WS path. ``None`` means auto-detect."""

    def setUp(self):
        self.user = User.objects.create_user(email="rtlang@example.com", password="pw")

    async def _resolve(self, *, meeting_forced, pref_language):
        import types

        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="RT Lang",
            slug=f"rt-lang-{meeting_forced or 'blank'}-{pref_language}",
            created_by=self.user,
            forced_language=meeting_forced,
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        prefs = types.SimpleNamespace(
            allowed_transcription_models=["openai/gpt-4o-transcribe"],
            transcription_model_live="openai/gpt-4o-transcribe",
            live_transcription_mode="realtime",
            transcription_language=pref_language,
        )
        with patch("core.preferences.get_preferences", return_value=prefs):
            return await consumer._load_and_lock_meeting(meeting.uuid)

    async def test_meeting_blank_uses_pref_default(self):
        result = await self._resolve(meeting_forced="", pref_language="no")
        self.assertEqual(result["forced_language"], "no")

    async def test_meeting_language_overrides_pref(self):
        result = await self._resolve(meeting_forced="en", pref_language="no")
        self.assertEqual(result["forced_language"], "en")

    async def test_meeting_auto_forces_detection(self):
        result = await self._resolve(meeting_forced="auto", pref_language="no")
        self.assertIsNone(result["forced_language"])

    async def test_auto_default_is_none(self):
        result = await self._resolve(meeting_forced="", pref_language="auto")
        self.assertIsNone(result["forced_language"])

    async def test_persist_meeting_language_resolves_effective(self):
        """The live-switch helper stores the dropdown value and returns the
        API language through the same cascade as a fresh connect."""
        import types

        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="RT Lang",
            slug="rt-lang-persist",
            created_by=self.user,
            forced_language="en",
        )
        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        consumer.meeting_id = meeting.id
        prefs = types.SimpleNamespace(transcription_language="no")
        with patch("core.preferences.get_preferences", return_value=prefs):
            self.assertIsNone(await consumer._persist_meeting_language("auto"))
            self.assertEqual(await consumer._persist_meeting_language(""), "no")
            self.assertEqual(await consumer._persist_meeting_language("en"), "en")

        meeting = await database_sync_to_async(Meeting.objects.get)(pk=meeting.pk)
        self.assertEqual(meeting.forced_language, "en")


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


class RealtimeUtteranceCostTests(TransactionTestCase):
    """The live model bills per minute: a completed turn is priced from the
    seconds it reports, else from the audio its commit covered."""

    def setUp(self):
        Meeting.objects.all().delete()
        self.user = User.objects.create_user(email="rtcost@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-rtcost", created_by=self.user)

    async def _persist(self, usage, duration_seconds):
        from meetings.consumers import MeetingTranscribeConsumer

        consumer = MeetingTranscribeConsumer()
        consumer.user = self.user
        consumer.meeting_id = self.meeting.id
        consumer._segments_total = 0  # set by connect()
        with patch("llm.service.logger.log_transcription_streaming") as mock_log:
            await consumer._allocate_and_persist_segment(
                text="hello", start_offset=0.0, model_id="openai/gpt-live-transcribe",
                usage=usage, duration_seconds=duration_seconds,
            )
        return mock_log.call_args.kwargs

    async def test_reported_seconds_price_the_turn(self):
        kwargs = await self._persist({"type": "duration", "seconds": 60}, 58.2)
        self.assertAlmostEqual(float(kwargs["cost_usd"]), 0.017, places=6)
        self.assertEqual(kwargs["audio_duration_seconds"], 60.0)

    async def test_commit_duration_is_the_fallback(self):
        kwargs = await self._persist(None, 30.0)
        self.assertAlmostEqual(float(kwargs["cost_usd"]), 0.0085, places=6)
        self.assertEqual(kwargs["audio_duration_seconds"], 30.0)
