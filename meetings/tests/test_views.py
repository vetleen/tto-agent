"""View tests for the meetings app."""
from __future__ import annotations

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import DataRoom
from meetings.models import (
    Meeting,
    MeetingArtifact,
    MeetingAttachment,
    MeetingDataRoom,
)

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingListViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="list@example.com", password="pw")

    def test_requires_login(self):
        response = self.client.get(reverse("meeting_list"))
        self.assertEqual(response.status_code, 302)

    def test_list_renders_for_authed_user(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("meeting_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Meetings")

    def test_post_creates_meeting_and_redirects_to_detail(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse("meeting_list"), {"name": "Acme call"})
        self.assertEqual(response.status_code, 302)
        meeting = Meeting.objects.get(created_by=self.user, name="Acme call")
        self.assertEqual(response["Location"], reverse("meeting_detail", args=[meeting.uuid]))

    def test_post_with_existing_slug_retries(self):
        Meeting.objects.create(name="Acme call", slug="acme-call", created_by=self.user)
        self.client.force_login(self.user)
        response = self.client.post(reverse("meeting_list"), {"name": "Acme call"})
        self.assertEqual(response.status_code, 302)
        # Two meetings now exist with the same name but different slugs.
        self.assertEqual(Meeting.objects.filter(created_by=self.user, name="Acme call").count(), 2)


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingDetailViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="detail@example.com", password="pw")
        self.other = User.objects.create_user(email="other@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-detail", created_by=self.user)

    def test_owner_can_view(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("meeting_detail", args=[self.meeting.uuid]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "M")

    def test_non_owner_redirected(self):
        self.client.force_login(self.other)
        response = self.client.get(reverse("meeting_detail", args=[self.meeting.uuid]))
        self.assertEqual(response.status_code, 302)


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingCRUDTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="crud@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="Old name", slug="m-crud", created_by=self.user)
        self.client.force_login(self.user)

    def test_rename(self):
        response = self.client.post(reverse("meeting_rename", args=[self.meeting.uuid]), {"name": "New name"})
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.name, "New name")

    def test_archive_toggle(self):
        self.client.post(reverse("meeting_archive", args=[self.meeting.uuid]))
        self.meeting.refresh_from_db()
        self.assertTrue(self.meeting.is_archived)
        self.client.post(reverse("meeting_archive", args=[self.meeting.uuid]))
        self.meeting.refresh_from_db()
        self.assertFalse(self.meeting.is_archived)

    def test_delete(self):
        response = self.client.post(reverse("meeting_delete", args=[self.meeting.uuid]))
        self.assertEqual(response.status_code, 302)
        self.assertFalse(Meeting.objects.filter(pk=self.meeting.pk).exists())


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingDataRoomLinkViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="dr@example.com", password="pw")
        self.other = User.objects.create_user(email="dr-other@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-link-view", created_by=self.user)
        self.my_room = DataRoom.objects.create(name="Mine", slug="mine-room", created_by=self.user)
        self.foreign_room = DataRoom.objects.create(name="Theirs", slug="theirs-room", created_by=self.other)
        self.client.force_login(self.user)

    def test_link_owned_room(self):
        response = self.client.post(
            reverse("meeting_link_data_room", args=[self.meeting.uuid]),
            {"data_room_id": str(self.my_room.uuid)},
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(MeetingDataRoom.objects.filter(meeting=self.meeting, data_room=self.my_room).exists())

    def test_cannot_link_foreign_room(self):
        self.client.post(
            reverse("meeting_link_data_room", args=[self.meeting.uuid]),
            {"data_room_id": str(self.foreign_room.uuid)},
        )
        self.assertFalse(MeetingDataRoom.objects.filter(meeting=self.meeting, data_room=self.foreign_room).exists())

    def test_unlink_room(self):
        MeetingDataRoom.objects.create(meeting=self.meeting, data_room=self.my_room)
        response = self.client.post(
            reverse("meeting_unlink_data_room", args=[self.meeting.uuid, self.my_room.uuid])
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(MeetingDataRoom.objects.filter(meeting=self.meeting, data_room=self.my_room).exists())


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingTranscriptUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="up@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-up", created_by=self.user)
        self.client.force_login(self.user)

    def test_upload_text_transcript(self):
        f = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        response = self.client.post(
            reverse("meeting_upload_transcript", args=[self.meeting.uuid]),
            {"transcript": f},
        )
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "hello world")
        self.assertEqual(self.meeting.transcript_source, Meeting.TranscriptSource.TEXT_UPLOAD)
        self.assertEqual(self.meeting.status, Meeting.Status.READY)

    def test_upload_rejects_unsupported_extension(self):
        f = SimpleUploadedFile("notes.docx", b"data", content_type="application/octet-stream")
        response = self.client.post(
            reverse("meeting_upload_transcript", args=[self.meeting.uuid]),
            {"transcript": f},
        )
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "")

    @override_settings(MEETING_TRANSCRIPT_UPLOAD_MAX_BYTES=10)
    def test_upload_rejects_oversized_transcript(self):
        f = SimpleUploadedFile("notes.txt", b"way too many bytes here", content_type="text/plain")
        self.client.post(
            reverse("meeting_upload_transcript", args=[self.meeting.uuid]),
            {"transcript": f},
        )
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "")


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingAudioUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="aup@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-aup", created_by=self.user)
        self.client.force_login(self.user)

    @patch("meetings.tasks.transcribe_uploaded_audio_task.delay")
    def test_upload_audio_enqueues_task(self, mock_delay):
        f = SimpleUploadedFile("clip.mp3", b"\x00" * 32, content_type="audio/mpeg")
        response = self.client.post(
            reverse("meeting_upload_audio", args=[self.meeting.uuid]),
            {"audio": f},
        )
        self.assertEqual(response.status_code, 302)
        mock_delay.assert_called_once()
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript_source, Meeting.TranscriptSource.AUDIO_UPLOAD)
        self.assertEqual(self.meeting.status, Meeting.Status.LIVE_TRANSCRIBING)

    @patch("meetings.tasks.transcribe_uploaded_audio_task.delay")
    def test_upload_audio_rejects_unsupported_extension(self, mock_delay):
        f = SimpleUploadedFile("notes.txt", b"text", content_type="text/plain")
        self.client.post(
            reverse("meeting_upload_audio", args=[self.meeting.uuid]),
            {"audio": f},
        )
        mock_delay.assert_not_called()


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingAttachmentViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="att@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-att-v", created_by=self.user)
        self.client.force_login(self.user)

    def test_attachment_upload(self):
        f = SimpleUploadedFile("slides.pdf", b"%PDF-1.4 fake", content_type="application/pdf")
        response = self.client.post(
            reverse("meeting_upload_attachment", args=[self.meeting.uuid]),
            {"file": f},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(MeetingAttachment.objects.filter(meeting=self.meeting).count(), 1)

    def test_attachment_delete(self):
        from django.core.files.base import ContentFile

        att = MeetingAttachment.objects.create(
            meeting=self.meeting, uploaded_by=self.user,
            file=ContentFile(b"x", name="x.pdf"),
            original_filename="x.pdf", content_type="application/pdf", size_bytes=1,
        )
        self.client.post(reverse("meeting_delete_attachment", args=[self.meeting.uuid, att.id]))
        self.assertFalse(MeetingAttachment.objects.filter(pk=att.pk).exists())


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingMetadataUpdateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="meta@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-meta", created_by=self.user)
        self.client.force_login(self.user)

    def test_update_metadata(self):
        response = self.client.post(
            reverse("meeting_update_metadata", args=[self.meeting.uuid]),
            {"agenda": "Q1 plan", "participants": "A, B"},
        )
        self.assertEqual(response.status_code, 200)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.agenda, "Q1 plan")
        self.assertEqual(self.meeting.participants, "A, B")
