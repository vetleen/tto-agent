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

    def test_list_view_is_get_only(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse("meeting_list"))
        # POST is no longer allowed on the list view; creation moved to meeting_create.
        self.assertEqual(response.status_code, 405)

    def test_create_endpoint_creates_meeting_with_default_name(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse("meeting_create"))
        self.assertEqual(response.status_code, 302)
        meeting = Meeting.objects.filter(created_by=self.user).first()
        self.assertIsNotNone(meeting)
        # Default name format: "YYMMDD - New meeting"
        self.assertRegex(meeting.name, r"^\d{6} - New meeting$")
        self.assertEqual(response["Location"], reverse("meeting_detail", args=[meeting.uuid]))

    def test_create_endpoint_with_transcribe_flag_appends_query(self):
        self.client.force_login(self.user)
        response = self.client.post(reverse("meeting_create"), {"transcribe": "1"})
        self.assertEqual(response.status_code, 302)
        meeting = Meeting.objects.filter(created_by=self.user).first()
        self.assertIsNotNone(meeting)
        self.assertEqual(
            response["Location"],
            f"{reverse('meeting_detail', args=[meeting.uuid])}?transcribe=1",
        )

    def test_create_endpoint_handles_slug_collision(self):
        self.client.force_login(self.user)
        self.client.post(reverse("meeting_create"))
        self.client.post(reverse("meeting_create"))
        self.assertEqual(Meeting.objects.filter(created_by=self.user).count(), 2)


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

    def test_update_metadata_can_rename(self):
        response = self.client.post(
            reverse("meeting_update_metadata", args=[self.meeting.uuid]),
            {"name": "Renamed via metadata"},
        )
        self.assertEqual(response.status_code, 200)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.name, "Renamed via metadata")

    def test_update_metadata_rejects_blank_name(self):
        response = self.client.post(
            reverse("meeting_update_metadata", args=[self.meeting.uuid]),
            {"name": "   "},
        )
        self.assertEqual(response.status_code, 400)


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingUnifiedUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="unified@example.com", password="pw")
        self.meeting = Meeting.objects.create(name="M", slug="m-unified", created_by=self.user)
        self.client.force_login(self.user)

    def test_upload_routes_text_to_transcript(self):
        f = SimpleUploadedFile("notes.txt", b"hello world", content_type="text/plain")
        response = self.client.post(
            reverse("meeting_upload", args=[self.meeting.uuid]),
            {"file": f},
        )
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "hello world")
        self.assertEqual(self.meeting.transcript_source, Meeting.TranscriptSource.TEXT_UPLOAD)

    @patch("meetings.tasks.transcribe_uploaded_audio_task.delay")
    def test_upload_routes_audio_to_audio_handler(self, mock_delay):
        f = SimpleUploadedFile("clip.mp3", b"\x00" * 32, content_type="audio/mpeg")
        response = self.client.post(
            reverse("meeting_upload", args=[self.meeting.uuid]),
            {"file": f},
        )
        self.assertEqual(response.status_code, 302)
        mock_delay.assert_called_once()
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript_source, Meeting.TranscriptSource.AUDIO_UPLOAD)

    def test_upload_rejects_unsupported_extension(self):
        f = SimpleUploadedFile("data.docx", b"binary", content_type="application/octet-stream")
        response = self.client.post(
            reverse("meeting_upload", args=[self.meeting.uuid]),
            {"file": f},
        )
        self.assertEqual(response.status_code, 302)
        self.meeting.refresh_from_db()
        self.assertEqual(self.meeting.transcript, "")


@override_settings(ALLOWED_HOSTS=["testserver"])
class MeetingSaveToDataRoomTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="save@example.com", password="pw")
        self.other = User.objects.create_user(email="save-other@example.com", password="pw")
        self.meeting = Meeting.objects.create(
            name="M", slug="m-save", created_by=self.user, transcript="full transcript text"
        )
        self.my_room = DataRoom.objects.create(name="Mine", slug="mine-save", created_by=self.user)
        self.foreign_room = DataRoom.objects.create(name="Theirs", slug="theirs-save", created_by=self.other)
        self.client.force_login(self.user)

    @patch("documents.tasks.process_document_task.delay")
    def test_save_transcript_to_owned_data_room(self, mock_delay):
        from documents.models import DataRoomDocument
        response = self.client.post(
            reverse("meeting_save_to_data_room", args=[self.meeting.uuid]),
            {"data_room_id": str(self.my_room.uuid)},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.my_room).count(), 1)
        mock_delay.assert_called_once()

    @patch("documents.tasks.process_document_task.delay")
    def test_save_uses_latest_artifact_when_present(self, mock_delay):
        from documents.models import DataRoomDocument
        MeetingArtifact.objects.create(
            meeting=self.meeting,
            kind=MeetingArtifact.Kind.MINUTES,
            content_md="# Minutes\nWilfred wrote these.",
            created_by=self.user,
        )
        self.client.post(
            reverse("meeting_save_to_data_room", args=[self.meeting.uuid]),
            {"data_room_id": str(self.my_room.uuid)},
        )
        doc = DataRoomDocument.objects.get(data_room=self.my_room)
        self.assertIn("minutes", doc.original_filename)

    def test_cannot_save_to_foreign_data_room(self):
        from documents.models import DataRoomDocument
        self.client.post(
            reverse("meeting_save_to_data_room", args=[self.meeting.uuid]),
            {"data_room_id": str(self.foreign_room.uuid)},
        )
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.foreign_room).count(), 0)

    def test_cannot_save_when_no_transcript_or_artifact(self):
        empty = Meeting.objects.create(name="Empty", slug="m-empty-save", created_by=self.user)
        from documents.models import DataRoomDocument
        self.client.post(
            reverse("meeting_save_to_data_room", args=[empty.uuid]),
            {"data_room_id": str(self.my_room.uuid)},
        )
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.my_room).count(), 0)
