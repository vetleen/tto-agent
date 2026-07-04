"""Model tests for the meetings app."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.test import TestCase

from meetings.models import (
    Meeting,
    MeetingAttachment,
    MeetingTranscriptSegment,
)

User = get_user_model()


class MeetingModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="m@example.com", password="pw")

    def test_create_meeting_defaults(self):
        m = Meeting.objects.create(name="Acme call", slug="acme-call", created_by=self.user)
        self.assertEqual(m.status, Meeting.Status.DRAFT)
        self.assertEqual(m.transcript, "")
        self.assertEqual(m.transcript_source, "")
        self.assertFalse(m.is_archived)
        self.assertIn("Acme call", str(m))

    def test_status_transitions_are_string_choices(self):
        m = Meeting.objects.create(name="X", slug="x", created_by=self.user)
        m.status = Meeting.Status.LIVE_TRANSCRIBING
        m.save(update_fields=["status"])
        m.refresh_from_db()
        self.assertEqual(m.status, "live_transcribing")

    def test_slug_unique(self):
        Meeting.objects.create(name="A", slug="dup", created_by=self.user)
        with self.assertRaises(IntegrityError):
            Meeting.objects.create(name="B", slug="dup", created_by=self.user)

    def test_one_live_transcription_per_user(self):
        Meeting.objects.create(
            name="A", slug="live-a", created_by=self.user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Meeting.objects.create(
                    name="B", slug="live-b", created_by=self.user,
                    status=Meeting.Status.LIVE_TRANSCRIBING,
                )

    def test_two_users_each_live_allowed(self):
        other = User.objects.create_user(email="m2@example.com", password="pw")
        Meeting.objects.create(
            name="A", slug="live-u1", created_by=self.user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        # Different user → no conflict.
        Meeting.objects.create(
            name="B", slug="live-u2", created_by=other,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )

    def test_live_plus_non_live_allowed(self):
        # The constraint is partial (only LIVE_TRANSCRIBING rows), so a user can
        # have one live meeting alongside any number of READY/etc. meetings.
        Meeting.objects.create(
            name="A", slug="live-only", created_by=self.user,
            status=Meeting.Status.LIVE_TRANSCRIBING,
        )
        Meeting.objects.create(
            name="B", slug="ready-1", created_by=self.user,
            status=Meeting.Status.READY,
        )
        Meeting.objects.create(
            name="C", slug="ready-2", created_by=self.user,
            status=Meeting.Status.READY,
        )


class MeetingTranscriptSegmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="seg@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-seg", created_by=self.user)

    def test_segment_unique_per_meeting(self):
        MeetingTranscriptSegment.objects.create(
            meeting=self.meeting, segment_index=0, text="hello",
            status=MeetingTranscriptSegment.Status.READY,
        )
        with self.assertRaises(IntegrityError):
            MeetingTranscriptSegment.objects.create(
                meeting=self.meeting, segment_index=0, text="dup",
                status=MeetingTranscriptSegment.Status.READY,
            )

    def test_segment_default_status_pending(self):
        seg = MeetingTranscriptSegment.objects.create(meeting=self.meeting, segment_index=1)
        self.assertEqual(seg.status, MeetingTranscriptSegment.Status.PENDING)


class MeetingAttachmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="att@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-att", created_by=self.user)

    def test_create_attachment(self):
        from django.core.files.base import ContentFile

        att = MeetingAttachment.objects.create(
            meeting=self.meeting,
            uploaded_by=self.user,
            file=ContentFile(b"hello", name="agenda.txt"),
            original_filename="agenda.txt",
            content_type="text/plain",
            size_bytes=5,
        )
        self.assertIn("agenda.txt", str(att))
