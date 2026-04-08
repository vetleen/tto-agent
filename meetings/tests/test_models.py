"""Model tests for the meetings app."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from documents.models import DataRoom
from meetings.models import (
    Meeting,
    MeetingArtifact,
    MeetingAttachment,
    MeetingDataRoom,
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


class MeetingDataRoomLinkTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="link@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-link", created_by=self.user)
        self.room = DataRoom.objects.create(name="Room", slug="room-link", created_by=self.user)

    def test_link_is_unique(self):
        MeetingDataRoom.objects.create(meeting=self.meeting, data_room=self.room)
        with self.assertRaises(IntegrityError):
            MeetingDataRoom.objects.create(meeting=self.meeting, data_room=self.room)

    def test_m2m_accessor(self):
        MeetingDataRoom.objects.create(meeting=self.meeting, data_room=self.room)
        self.assertIn(self.room, list(self.meeting.data_rooms.all()))
        self.assertIn(self.meeting, list(self.room.meetings.all()))


class MeetingArtifactTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="art@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-art", created_by=self.user)

    def test_kind_default_minutes(self):
        art = MeetingArtifact.objects.create(
            meeting=self.meeting, content_md="# minutes", created_by=self.user,
        )
        self.assertEqual(art.kind, MeetingArtifact.Kind.MINUTES)
        self.assertIn("Minutes", str(art))


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
