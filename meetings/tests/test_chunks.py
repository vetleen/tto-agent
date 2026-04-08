"""Tests for meetings.services.chunks helpers."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from meetings.models import Meeting, MeetingTranscriptSegment
from meetings.services.chunks import (
    cleanup_temp,
    recompute_meeting_transcript,
    write_chunk_to_temp,
)

User = get_user_model()


class WriteChunkTempTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="wt@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-chunk", created_by=self.user)
        self.tmpdir = tempfile.mkdtemp(prefix="meeting_chunks_test_")
        self.addCleanup(self._wipe_tmpdir)

    def _wipe_tmpdir(self):
        for root, _dirs, files in os.walk(self.tmpdir):
            for name in files:
                try: os.unlink(os.path.join(root, name))
                except OSError: pass
        try:
            for root, dirs, _ in os.walk(self.tmpdir, topdown=False):
                for d in dirs:
                    try: os.rmdir(os.path.join(root, d))
                    except OSError: pass
            os.rmdir(self.tmpdir)
        except OSError:
            pass

    def test_write_chunk_creates_file_with_correct_extension(self):
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            path = write_chunk_to_temp(self.meeting.uuid, 0, b"\x00\x01\x02", "audio/webm;codecs=opus")
        self.assertTrue(Path(path).exists())
        self.assertTrue(str(path).endswith(".webm"))

    def test_cleanup_temp_removes_file(self):
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            path = write_chunk_to_temp(self.meeting.uuid, 1, b"\xff", "audio/ogg")
            self.assertTrue(Path(path).exists())
            cleanup_temp(path)
            self.assertFalse(Path(path).exists())

    def test_cleanup_temp_idempotent(self):
        cleanup_temp("/tmp/this/does/not/exist.webm")  # should not raise

    def test_write_chunk_rejects_invalid_uuid(self):
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            with self.assertRaises(ValueError):
                write_chunk_to_temp("not-a-uuid; rm -rf /", 0, b"x", "audio/webm")


class RecomputeTranscriptTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="rt@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-recompute", created_by=self.user)

    def test_join_segments_in_order(self):
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=2, text="third",
            status=MeetingTranscriptSegment.Status.READY,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=0, text="first",
            status=MeetingTranscriptSegment.Status.READY,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=1, text="second",
            status=MeetingTranscriptSegment.Status.READY,
        )
        result = recompute_meeting_transcript(self.meeting.id)
        self.meeting.refresh_from_db()
        self.assertEqual(result, "first\n\nsecond\n\nthird")
        self.assertEqual(self.meeting.transcript, "first\n\nsecond\n\nthird")

    def test_skips_pending_and_failed_segments(self):
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=0, text="ready",
            status=MeetingTranscriptSegment.Status.READY,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=1, text="pending",
            status=MeetingTranscriptSegment.Status.PENDING,
        )
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=2, text="failed",
            status=MeetingTranscriptSegment.Status.FAILED,
        )
        recompute_meeting_transcript(self.meeting.id)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "ready")
