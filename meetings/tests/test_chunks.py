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
    write_chunk_stream_to_temp,
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


class WriteChunkStreamTempTests(TestCase):
    """Streaming variant used by the audio upload view — must never load the
    full payload into memory."""

    def setUp(self):
        self.user = User.objects.create_user(email="ws@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-chunk-stream", created_by=self.user)
        self.tmpdir = tempfile.mkdtemp(prefix="meeting_chunks_stream_test_")
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

    def test_streams_uploaded_file_to_local_disk(self):
        """Written file content must match the uploaded bytes end-to-end."""
        from django.core.files.uploadedfile import SimpleUploadedFile

        payload = (b"\xde\xad\xbe\xef" * 1024)  # 4 KB; small but exercises .chunks()
        uf = SimpleUploadedFile("audio.webm", payload, content_type="audio/webm")
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            path = write_chunk_stream_to_temp(
                self.meeting.uuid, 0, uf, "audio/webm;codecs=opus",
            )
        self.assertTrue(Path(path).exists())
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), payload)

    def test_streaming_does_not_consume_whole_file_at_once(self):
        """Assert we call .chunks() on the file-like rather than .read() with
        no size — if we regress to .read(), memory doubles for large uploads."""

        class _Tracker:
            def __init__(self, data):
                self._data = data
                self.read_calls = 0
                self.chunks_calls = 0

            def seek(self, pos):
                return None

            def read(self, size=-1):
                self.read_calls += 1
                if size in (-1, None):
                    raise AssertionError("unbounded .read() not allowed on upload path")
                data = self._data[:size]
                self._data = self._data[size:]
                return data

            def chunks(self, chunk_size=1024 * 1024):
                self.chunks_calls += 1
                while self._data:
                    out = self._data[:chunk_size]
                    self._data = self._data[chunk_size:]
                    yield out

        tracker = _Tracker(b"\x01" * 2048)
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            path = write_chunk_stream_to_temp(
                self.meeting.uuid, 5, tracker, "audio/mpeg",
            )
        self.assertTrue(Path(path).exists())
        self.assertEqual(tracker.chunks_calls, 1)
        self.assertEqual(tracker.read_calls, 0)

    def test_rejects_invalid_uuid(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        uf = SimpleUploadedFile("a.webm", b"x", content_type="audio/webm")
        with override_settings(MEETING_CHUNK_TEMP_DIR=self.tmpdir):
            with self.assertRaises(ValueError):
                write_chunk_stream_to_temp("not-a-uuid; rm -rf /", 0, uf, "audio/webm")


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
