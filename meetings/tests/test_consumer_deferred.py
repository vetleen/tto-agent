"""Consumer tests for the deferred-batch fixes: C1 (seed transcript), C4c
(per-meeting presence lock), and S2 (duration = recorded time).

These use a locmem cache override so the C4c presence lock is deterministic and
doesn't depend on (or pollute) the real Redis the default test cache uses.
"""
from __future__ import annotations

import json
import tempfile
from unittest.mock import patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from meetings.consumers import MeetingTranscribeConsumer
from meetings.models import Meeting, MeetingTranscriptSegment
from meetings.tests.test_consumer import _make_communicator, _patch_chunked_mode

User = get_user_model()

_CONSUMER_OVERRIDES = dict(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
        },
    },
    MEETING_CHUNK_TEMP_DIR=tempfile.mkdtemp(prefix="wilfred-deferred-chunks-"),
    CACHES={
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "meetings-deferred-tests",
        }
    },
)


async def _chunked_user(email):
    # Chunked routing is pinned via _patch_chunked_mode in each class's setUp
    # (the live-mode picker was removed), so this just creates a plain user.
    return await database_sync_to_async(User.objects.create_user)(
        email=email, password="pw",
    )


@override_settings(**_CONSUMER_OVERRIDES)
class SeedTranscriptOnConnectTests(TransactionTestCase):
    def setUp(self):
        Meeting.objects.all().delete()
        _patch_chunked_mode(self)

    async def test_seeds_existing_transcript_as_segment_zero(self):
        user = await _chunked_user("c1-seed@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Up", slug="m-c1-seed", created_by=user,
            transcript="Existing uploaded transcript.",
            transcript_source=Meeting.TranscriptSource.TEXT_UPLOAD,
            status=Meeting.Status.READY,
        )
        comm = _make_communicator(meeting.uuid, user)
        connected, _ = await comm.connect()
        self.assertTrue(connected)
        msg = json.loads(await comm.receive_from())
        self.assertEqual(msg["type"], "started")
        # Seeded segment 0 pushes the base to 1.
        self.assertEqual(msg["segment_index_base"], 1)

        seg = await database_sync_to_async(
            MeetingTranscriptSegment.objects.get
        )(meeting=meeting, segment_index=0)
        self.assertEqual(seg.text, "Existing uploaded transcript.")
        self.assertEqual(seg.status, MeetingTranscriptSegment.Status.READY)
        await comm.disconnect()

    async def test_seed_is_idempotent_across_reconnect(self):
        user = await _chunked_user("c1-idem@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Up", slug="m-c1-idem", created_by=user,
            transcript="Some transcript.",
            transcript_source=Meeting.TranscriptSource.TEXT_UPLOAD,
            status=Meeting.Status.READY,
        )
        comm1 = _make_communicator(meeting.uuid, user)
        await comm1.connect()
        await comm1.receive_from()
        await comm1.disconnect()

        comm2 = _make_communicator(meeting.uuid, user)
        await comm2.connect()
        await comm2.receive_from()
        await comm2.disconnect()

        count = await database_sync_to_async(
            MeetingTranscriptSegment.objects.filter(meeting=meeting, segment_index=0).count
        )()
        self.assertEqual(count, 1)


@override_settings(**_CONSUMER_OVERRIDES)
class PresenceLockTests(TransactionTestCase):
    def setUp(self):
        from django.core.cache import cache
        Meeting.objects.all().delete()
        _patch_chunked_mode(self)
        cache.clear()

    async def test_second_tab_same_meeting_rejected_4409(self):
        user = await _chunked_user("c4c-2tab@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="M", slug="m-c4c", created_by=user,
        )
        comm1 = _make_communicator(meeting.uuid, user)
        connected1, _ = await comm1.connect()
        self.assertTrue(connected1)
        await comm1.receive_from()  # started

        comm2 = _make_communicator(meeting.uuid, user)
        connected2, code = await comm2.connect()
        self.assertFalse(connected2)
        self.assertEqual(code, 4409)
        await comm2.disconnect()
        await comm1.disconnect()

    async def test_reconnect_after_disconnect_succeeds(self):
        user = await _chunked_user("c4c-recon@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="M", slug="m-c4c-recon", created_by=user,
        )
        comm1 = _make_communicator(meeting.uuid, user)
        await comm1.connect()
        await comm1.receive_from()
        await comm1.disconnect()  # releases the lock

        comm2 = _make_communicator(meeting.uuid, user)
        connected, _ = await comm2.connect()
        self.assertTrue(connected)
        await comm2.disconnect()

    async def test_foreign_stale_lock_refuses_then_clears(self):
        from django.core.cache import cache

        user = await _chunked_user("c4c-stale@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="M", slug="m-c4c-stale", created_by=user,
        )
        key = f"meeting_live_session:{meeting.uuid}"
        await database_sync_to_async(cache.set)(key, "another-connection", 45)

        comm = _make_communicator(meeting.uuid, user)
        connected, code = await comm.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4409)
        await comm.disconnect()

        await database_sync_to_async(cache.delete)(key)
        comm2 = _make_communicator(meeting.uuid, user)
        connected2, _ = await comm2.connect()
        self.assertTrue(connected2)
        await comm2.disconnect()


@override_settings(**_CONSUMER_OVERRIDES)
class FinalizeDurationTests(TransactionTestCase):
    def setUp(self):
        Meeting.objects.all().delete()
        _patch_chunked_mode(self)

    def _consumer(self, meeting_id, connected_monotonic):
        consumer = MeetingTranscribeConsumer()
        consumer.meeting_id = meeting_id
        consumer._segments_total = 0
        consumer._segments_failed = 0
        consumer._duration_committed = False
        consumer._connected_monotonic = connected_monotonic
        return consumer

    async def test_duration_accumulates_across_sessions(self):
        user = await _chunked_user("s2-accum@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Dur", slug="m-s2-accum", created_by=user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        # Session 1: 60s of recorded wall time.
        c1 = self._consumer(meeting.id, connected_monotonic=100.0)
        with patch("meetings.consumers.time.monotonic", return_value=160.0):
            d1, _, _ = await c1._finalize_meeting(interrupted=True)
        self.assertEqual(d1, 60)

        # Session 2 (resume): a 30s span adds to the stored 60 (idle gap excluded).
        await database_sync_to_async(
            Meeting.objects.filter(pk=meeting.id).update
        )(status=Meeting.Status.LIVE_TRANSCRIBING)
        c2 = self._consumer(meeting.id, connected_monotonic=200.0)
        with patch("meetings.consumers.time.monotonic", return_value=230.0):
            d2, _, _ = await c2._finalize_meeting(interrupted=False)
        self.assertEqual(d2, 90)

    async def test_duration_not_double_counted_on_retry(self):
        user = await _chunked_user("s2-retry@example.com")
        meeting = await database_sync_to_async(Meeting.objects.create)(
            name="Dur", slug="m-s2-retry", created_by=user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        c = self._consumer(meeting.id, connected_monotonic=100.0)
        with patch("meetings.consumers.time.monotonic", return_value=150.0):
            d1, _, _ = await c._finalize_meeting(interrupted=False)
        self.assertEqual(d1, 50)
        # A retry (stop-failure → disconnect) must add 0, not another 50.
        with patch("meetings.consumers.time.monotonic", return_value=999.0):
            d2, _, _ = await c._finalize_meeting(interrupted=True)
        self.assertEqual(d2, 50)
