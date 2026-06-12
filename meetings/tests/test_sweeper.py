"""Tests for meetings.tasks.expire_stale_transcriptions."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from meetings.models import Meeting, MeetingTranscriptSegment
from meetings.tasks import expire_stale_transcriptions

User = get_user_model()


class ExpireStaleTranscriptionsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sweep@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="Swept meeting", slug="m-sweeper", created_by=self.user,
        )

    def _make_segment(self, status=MeetingTranscriptSegment.Status.PENDING,
                      minutes_old=0, segment_index=0):
        segment = MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=segment_index, status=status,
        )
        # created_at is auto_now_add — backdate via queryset update.
        MeetingTranscriptSegment.objects.filter(pk=segment.pk).update(
            created_at=timezone.now() - timedelta(minutes=minutes_old),
        )
        segment.refresh_from_db()
        return segment

    def _make_upload_meeting(self, minutes_old=0, status=Meeting.Status.LIVE_TRANSCRIBING,
                             source=Meeting.TranscriptSource.AUDIO_UPLOAD, slug="m-upload"):
        meeting = Meeting.objects.create(
            name="Upload", slug=slug, created_by=self.user,
            status=status, transcript_source=source,
        )
        Meeting.objects.filter(pk=meeting.pk).update(
            updated_at=timezone.now() - timedelta(minutes=minutes_old),
        )
        meeting.refresh_from_db()
        return meeting

    @patch("meetings.tasks._push_to_ws")
    def test_stale_pending_segment_marked_failed(self, mock_push):
        segment = self._make_segment(minutes_old=20)

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 1)
        segment.refresh_from_db()
        self.assertEqual(segment.status, MeetingTranscriptSegment.Status.FAILED)
        self.assertIn("interrupted", segment.error)
        self.assertIsNotNone(segment.transcribed_at)
        mock_push.assert_called_once()
        group_arg, payload = mock_push.call_args.args
        self.assertEqual(group_arg, str(self.meeting.uuid))
        self.assertEqual(payload["type"], "segment.failed")
        self.assertEqual(payload["segment_index"], segment.segment_index)

    @patch("meetings.tasks._push_to_ws")
    def test_fresh_pending_segment_untouched(self, mock_push):
        segment = self._make_segment(minutes_old=5)

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 0)
        mock_push.assert_not_called()
        segment.refresh_from_db()
        self.assertEqual(segment.status, MeetingTranscriptSegment.Status.PENDING)

    @patch("meetings.tasks._push_to_ws")
    def test_old_terminal_segments_untouched(self, mock_push):
        ready = self._make_segment(
            status=MeetingTranscriptSegment.Status.READY, minutes_old=120, segment_index=0,
        )
        failed = self._make_segment(
            status=MeetingTranscriptSegment.Status.FAILED, minutes_old=120, segment_index=1,
        )

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 0)
        ready.refresh_from_db()
        failed.refresh_from_db()
        self.assertEqual(ready.status, MeetingTranscriptSegment.Status.READY)
        self.assertEqual(failed.status, MeetingTranscriptSegment.Status.FAILED)

    def test_stale_upload_meeting_marked_failed(self):
        meeting = self._make_upload_meeting(minutes_old=90)

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 1)
        meeting.refresh_from_db()
        self.assertEqual(meeting.status, Meeting.Status.FAILED)
        self.assertIn("interrupted", meeting.transcription_error)
        self.assertIsNotNone(meeting.ended_at)

    def test_fresh_upload_meeting_untouched(self):
        meeting = self._make_upload_meeting(minutes_old=30)

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 0)
        meeting.refresh_from_db()
        self.assertEqual(meeting.status, Meeting.Status.LIVE_TRANSCRIBING)

    def test_live_meeting_never_swept(self):
        """Live-path LIVE_TRANSCRIBING is WS-managed (Stop/disconnect handle
        it) — only the upload path goes through Celery and can be stranded."""
        live = self._make_upload_meeting(
            minutes_old=180, source=Meeting.TranscriptSource.LIVE, slug="m-live",
        )
        unset = self._make_upload_meeting(minutes_old=180, source="", slug="m-unset")

        handled = expire_stale_transcriptions()

        self.assertEqual(handled, 0)
        live.refresh_from_db()
        unset.refresh_from_db()
        self.assertEqual(live.status, Meeting.Status.LIVE_TRANSCRIBING)
        self.assertEqual(unset.status, Meeting.Status.LIVE_TRANSCRIBING)

    def test_swallows_transient_db_error(self):
        from django.db.utils import OperationalError

        with patch.object(
            MeetingTranscriptSegment.objects,
            "filter",
            side_effect=OperationalError("the database system is starting up"),
        ):
            result = expire_stale_transcriptions()

        self.assertEqual(result, 0)

    def test_propagates_non_transient_db_error(self):
        from django.db.utils import ProgrammingError

        with patch.object(
            MeetingTranscriptSegment.objects,
            "filter",
            side_effect=ProgrammingError("column does not exist"),
        ):
            with self.assertRaises(ProgrammingError):
                expire_stale_transcriptions()
