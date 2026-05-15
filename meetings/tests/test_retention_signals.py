"""Tests for meetings retention signals and file cleanup."""
from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone

from meetings.models import Meeting, MeetingAttachment, MeetingTranscriptSegment

User = get_user_model()


class MeetingRetainUntilTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="meet@example.com", password="testpass")

    def test_save_sets_retain_until(self):
        before = timezone.now()
        m = Meeting.objects.create(name="M", slug="m-ret", created_by=self.user)
        self.assertIsNotNone(m.retain_until)
        self.assertGreaterEqual(m.retain_until, before + timedelta(days=89))

    def test_segment_extends_meeting_retain(self):
        m = Meeting.objects.create(name="M", slug="m-seg-ret", created_by=self.user)
        Meeting.objects.filter(pk=m.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        MeetingTranscriptSegment.objects.create(
            meeting=m,
            segment_index=0,
            text="hello world",
        )
        m.refresh_from_db()

        self.assertGreater(m.retain_until, timezone.now() + timedelta(days=88))

    def test_attachment_extends_meeting_retain(self):
        m = Meeting.objects.create(name="M", slug="m-att-ret", created_by=self.user)
        Meeting.objects.filter(pk=m.pk).update(
            retain_until=timezone.now() - timedelta(days=1),
        )

        MeetingAttachment.objects.create(
            meeting=m,
            uploaded_by=self.user,
            file=SimpleUploadedFile("a.txt", b"data", content_type="text/plain"),
            original_filename="a.txt",
        )
        m.refresh_from_db()

        self.assertGreater(m.retain_until, timezone.now() + timedelta(days=88))


class MeetingAttachmentFileCleanupTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._override = override_settings(MEDIA_ROOT=cls._tmpdir.name)
        cls._override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._override.disable()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        self.user = User.objects.create_user(email="matt@example.com", password="testpass")
        self.meeting = Meeting.objects.create(name="M", slug="m-file", created_by=self.user)

    def test_delete_attachment_removes_file(self):
        upload = SimpleUploadedFile("slide.pdf", b"pdf-data", content_type="application/pdf")
        att = MeetingAttachment.objects.create(
            meeting=self.meeting,
            uploaded_by=self.user,
            file=upload,
            original_filename="slide.pdf",
        )
        stored_path = Path(att.file.path)
        self.assertTrue(stored_path.exists())

        att.delete()
        self.assertFalse(stored_path.exists())

    def test_meeting_delete_cascades_to_file_removal(self):
        upload = SimpleUploadedFile("agenda.pdf", b"data", content_type="application/pdf")
        att = MeetingAttachment.objects.create(
            meeting=self.meeting,
            uploaded_by=self.user,
            file=upload,
            original_filename="agenda.pdf",
        )
        stored_path = Path(att.file.path)
        self.assertTrue(stored_path.exists())

        self.meeting.delete()
        self.assertFalse(stored_path.exists())
