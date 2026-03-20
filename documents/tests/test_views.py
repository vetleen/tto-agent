import json
import sys
import types
from datetime import timedelta
from io import BytesIO
from unittest.mock import Mock, call, patch, MagicMock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentTag
from documents.views import _relative_upload_date

User = get_user_model()


class RelativeUploadDateTests(SimpleTestCase):
    """Unit tests for _relative_upload_date() covering every branch."""

    def _make_dt(self, days_ago):
        return timezone.now() - timedelta(days=days_ago)

    def test_none_returns_empty_string(self):
        self.assertEqual(_relative_upload_date(None), "")

    def test_today(self):
        result = _relative_upload_date(timezone.now())
        self.assertTrue(result.startswith("Today at "), result)

    def test_yesterday(self):
        result = _relative_upload_date(self._make_dt(1))
        self.assertEqual(result, "Yesterday")

    def test_days_ago(self):
        result = _relative_upload_date(self._make_dt(15))
        self.assertEqual(result, "15 days ago")

    def test_one_month_ago(self):
        result = _relative_upload_date(self._make_dt(35))
        self.assertEqual(result, "1 month ago")

    def test_one_year_ago(self):
        result = _relative_upload_date(self._make_dt(370))
        self.assertEqual(result, "1 year ago")


@override_settings(ALLOWED_HOSTS=["testserver"])
class DocumentViewsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.data_room = DataRoom.objects.create(name="Test", slug="test", created_by=self.user)
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _make_doc(self):
        return DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=DataRoomDocument.Status.READY,
        )

    def test_data_room_list_requires_login(self):
        response = self.client.get(reverse("data_room_list"))
        self.assertEqual(response.status_code, 302)

    def test_data_room_create_retries_after_integrity_error(self):
        self.client.force_login(self.user)

        created_data_room = DataRoom(name="Test Retry", slug="test-retry-1", created_by=self.user)
        with patch("documents.views.DataRoom.objects.create", side_effect=[IntegrityError(), created_data_room]) as mock_create:
            response = self.client.post(reverse("data_room_list"), {"name": "Test Retry"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_create.call_count, 2)
        self.assertEqual(mock_create.call_args_list[0], call(name="Test Retry", slug="test-retry", created_by=self.user, description=""))
        self.assertEqual(mock_create.call_args_list[1], call(name="Test Retry", slug="test-retry-1", created_by=self.user, description=""))

    def test_data_room_list_renders_for_owner(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("data_room_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test")

    def test_data_room_documents_owner_sees_upload_form(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload documents")
        self.assertContains(response, "name=\"file\"")

    def test_document_upload_creates_document_and_redirects(self):
        self.client.force_login(self.user)
        content = b"Hello world"
        f = BytesIO(content)
        f.name = "test.txt"
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room).count(), 1)
        doc = DataRoomDocument.objects.get(data_room=self.data_room)
        self.assertEqual(doc.original_filename, "test.txt")
        # Status may be UPLOADED (task queued), READY (sync processed), or FAILED (sync ran but deps missing)
        self.assertIn(
            doc.status,
            (DataRoomDocument.Status.UPLOADED, DataRoomDocument.Status.READY, DataRoomDocument.Status.FAILED),
        )

    def test_document_upload_tags_source_user_uploaded(self):
        self.client.force_login(self.user)
        f = BytesIO(b"Hello world")
        f.name = "tagged.txt"
        self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )
        doc = DataRoomDocument.objects.get(data_room=self.data_room)
        tag = DataRoomDocumentTag.objects.get(document=doc, key="source")
        self.assertEqual(tag.value, "user_uploaded")

    def test_document_upload_marks_failed_when_task_enqueue_fails(self):
        self.client.force_login(self.user)
        fake_task = Mock()
        fake_task.delay.side_effect = RuntimeError("broker unavailable")
        fake_module = types.SimpleNamespace(process_document_task=fake_task)
        f = BytesIO(b"hello")
        f.name = "enqueue-fail.txt"

        with patch.dict(sys.modules, {"documents.tasks": fake_module}):
            with self.assertLogs("documents.views", level="ERROR"):
                response = self.client.post(
                    reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
                    {"file": f},
                    follow=True,
                )

        self.assertEqual(response.status_code, 200)
        doc = DataRoomDocument.objects.get(data_room=self.data_room, original_filename="enqueue-fail.txt")
        self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
        self.assertIn("broker unavailable", doc.processing_error)
        self.assertContains(response, "processing could not be started")

    @override_settings(DOCUMENT_ALLOWED_MIME_TYPES=set())
    def test_document_upload_allows_mime_when_allowlist_not_configured(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("mime-ok.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="mime-ok.txt").count(), 1)

    @override_settings(DOCUMENT_ALLOWED_MIME_TYPES={"application/pdf"})
    def test_document_upload_rejects_mime_not_in_allowlist(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("mime-blocked.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="mime-blocked.txt").count(), 0)
        self.assertContains(response, "unsupported file type")

    def test_document_upload_sanitizes_path_from_original_filename(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile(r"..\..\secret\report.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        doc = DataRoomDocument.objects.get(data_room=self.data_room)
        self.assertEqual(doc.original_filename, "report.txt")

    def test_document_upload_truncates_original_filename_to_model_limit(self):
        self.client.force_login(self.user)
        long_name = f"{'a' * 300}.txt"
        f = SimpleUploadedFile(long_name, b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        doc = DataRoomDocument.objects.get(data_room=self.data_room)
        self.assertLessEqual(len(doc.original_filename), 255)
        self.assertTrue(doc.original_filename.endswith('.txt'))

    def test_document_upload_rejects_empty_file(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("empty.txt", b"", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="empty.txt").count(), 0)
        self.assertContains(response, "file is empty")

    def test_document_upload_rejects_unsupported_extension(self):
        self.client.force_login(self.user)
        f = BytesIO(b"x")
        f.name = "file.xyz"
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room).count(), 0)
        self.assertContains(response, "unsupported file type")

    def test_document_chunks_api_requires_auth(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=DataRoomDocument.Status.READY,
        )
        response = self.client.get(
            reverse("document_chunks", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 302)

    def test_document_chunks_api_returns_ordered_chunks(self):
        self.client.force_login(self.user)
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=DataRoomDocument.Status.READY,
        )
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=0, text="A", token_count=1)
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=1, text="B", token_count=1)
        response = self.client.get(
            reverse("document_chunks", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["chunks"]), 2)
        self.assertEqual(data["chunks"][0]["text"], "A")
        self.assertEqual(data["chunks"][1]["text"], "B")

    # ------------------------------------------------------------------ #
    # data_room_delete                                                     #
    # ------------------------------------------------------------------ #

    def test_data_room_delete_removes_data_room(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_delete", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertRedirects(response, reverse("data_room_list"))
        self.assertFalse(DataRoom.objects.filter(pk=self.data_room.pk).exists())

    def test_data_room_delete_other_user_blocked(self):
        self.client.force_login(self.other)
        self.client.post(
            reverse("data_room_delete", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertTrue(DataRoom.objects.filter(pk=self.data_room.pk).exists())

    def test_data_room_delete_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("data_room_delete", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    # ------------------------------------------------------------------ #
    # data_room_rename                                                     #
    # ------------------------------------------------------------------ #

    def test_data_room_rename_updates_name(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_rename", kwargs={"data_room_id": self.data_room.uuid}),
            {"name": "Renamed Project"},
        )
        self.assertRedirects(response, reverse("data_room_list"))
        self.data_room.refresh_from_db()
        self.assertEqual(self.data_room.name, "Renamed Project")

    def test_data_room_rename_rejects_empty_name(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_rename", kwargs={"data_room_id": self.data_room.uuid}),
            {"name": ""},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.data_room.refresh_from_db()
        self.assertEqual(self.data_room.name, "Test")

    def test_data_room_rename_truncates_at_255(self):
        self.client.force_login(self.user)
        long_name = "x" * 300
        self.client.post(
            reverse("data_room_rename", kwargs={"data_room_id": self.data_room.uuid}),
            {"name": long_name},
        )
        self.data_room.refresh_from_db()
        self.assertLessEqual(len(self.data_room.name), 255)

    def test_data_room_rename_other_user_blocked(self):
        self.client.force_login(self.other)
        self.client.post(
            reverse("data_room_rename", kwargs={"data_room_id": self.data_room.uuid}),
            {"name": "Hacked"},
        )
        self.data_room.refresh_from_db()
        self.assertEqual(self.data_room.name, "Test")

    # ------------------------------------------------------------------ #
    # document_delete                                                      #
    # ------------------------------------------------------------------ #

    def test_document_delete_removes_document(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_delete", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        self.assertRedirects(
            response,
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertFalse(DataRoomDocument.objects.filter(pk=doc.pk).exists())

    def test_document_delete_other_user_blocked(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        self.client.post(
            reverse("document_delete", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )
        self.assertTrue(DataRoomDocument.objects.filter(pk=doc.pk).exists())

    # ------------------------------------------------------------------ #
    # document_rename                                                      #
    # ------------------------------------------------------------------ #

    def test_document_rename_updates_filename(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        self.client.post(
            reverse("document_rename", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
            {"name": "renamed.txt"},
        )
        doc.refresh_from_db()
        self.assertEqual(doc.original_filename, "renamed.txt")

    def test_document_rename_rejects_empty_name(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_rename", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
            {"name": ""},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        doc.refresh_from_db()
        self.assertEqual(doc.original_filename, "doc.txt")

    def test_document_rename_other_user_blocked(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        self.client.post(
            reverse("document_rename", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
            {"name": "hacked.txt"},
        )
        doc.refresh_from_db()
        self.assertEqual(doc.original_filename, "doc.txt")

    # ------------------------------------------------------------------ #
    # document_upload — oversized file                                    #
    # ------------------------------------------------------------------ #

    @override_settings(DOCUMENT_UPLOAD_MAX_SIZE_BYTES=5)
    def test_document_upload_rejects_oversized_file(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("big.txt", b"0123456789", content_type="text/plain")
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "too large")
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="big.txt").count(), 0)

    # ------------------------------------------------------------------ #
    # document_chunks API — additional authorization cases                #
    # ------------------------------------------------------------------ #

    def test_document_chunks_api_forbidden_for_other_user(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        response = self.client.get(
            reverse("document_chunks", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "Forbidden"})

    def test_document_chunks_api_returns_404_for_missing_doc(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("document_chunks", kwargs={"data_room_id": self.data_room.uuid, "document_id": 99999})
        )
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------ #
    # document_upload — multi-file                                         #
    # ------------------------------------------------------------------ #

    def test_multi_file_upload_creates_multiple_documents(self):
        self.client.force_login(self.user)
        f1 = SimpleUploadedFile("a.txt", b"hello", content_type="text/plain")
        f2 = SimpleUploadedFile("b.txt", b"world", content_type="text/plain")
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": [f1, f2]},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room).count(), 2)
        filenames = set(
            DataRoomDocument.objects.filter(data_room=self.data_room).values_list("original_filename", flat=True)
        )
        self.assertEqual(filenames, {"a.txt", "b.txt"})
        self.assertContains(response, "2 files uploaded successfully")

    def test_multi_file_upload_partial_rejection(self):
        self.client.force_login(self.user)
        good = SimpleUploadedFile("good.txt", b"content", content_type="text/plain")
        bad = BytesIO(b"content")
        bad.name = "bad.xyz"
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": [good, bad]},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="good.txt").count(), 1
        )
        self.assertEqual(
            DataRoomDocument.objects.filter(data_room=self.data_room, original_filename="bad.xyz").count(), 0
        )
        self.assertContains(response, "bad.xyz")
        self.assertContains(response, "unsupported file type")
        self.assertContains(response, "1 file uploaded successfully")

    def test_multi_file_upload_all_rejected(self):
        self.client.force_login(self.user)
        f1 = BytesIO(b"x")
        f1.name = "a.xyz"
        f2 = BytesIO(b"x")
        f2.name = "b.xyz"
        response = self.client.post(
            reverse("document_upload", kwargs={"data_room_id": self.data_room.uuid}),
            {"file": [f1, f2]},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(DataRoomDocument.objects.filter(data_room=self.data_room).count(), 0)

    # ------------------------------------------------------------------ #
    # document_bulk_delete                                                 #
    # ------------------------------------------------------------------ #

    def test_bulk_delete_removes_selected_docs(self):
        self.client.force_login(self.user)
        d1 = self._make_doc()
        d2 = self._make_doc()
        d3 = self._make_doc()
        response = self.client.post(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [d1.id, d2.id]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["deleted"], 2)
        self.assertFalse(DataRoomDocument.objects.filter(pk=d1.pk).exists())
        self.assertFalse(DataRoomDocument.objects.filter(pk=d2.pk).exists())
        self.assertTrue(DataRoomDocument.objects.filter(pk=d3.pk).exists())

    def test_bulk_delete_ignores_other_data_room_ids(self):
        self.client.force_login(self.user)
        other_data_room = DataRoom.objects.create(name="Other", slug="other", created_by=self.user)
        other_doc = DataRoomDocument.objects.create(
            data_room=other_data_room, uploaded_by=self.user,
            original_filename="other.txt", status=DataRoomDocument.Status.READY,
        )
        response = self.client.post(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [other_doc.id]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted"], 0)
        self.assertTrue(DataRoomDocument.objects.filter(pk=other_doc.pk).exists())

    def test_bulk_delete_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    def test_bulk_delete_blocks_other_user(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [doc.id]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(DataRoomDocument.objects.filter(pk=doc.pk).exists())

    def test_bulk_delete_rejects_empty_ids(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_delete_rejects_bad_json(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("document_bulk_delete", kwargs={"data_room_id": self.data_room.uuid}),
            data="not json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------ #
    # document_bulk_archive                                                #
    # ------------------------------------------------------------------ #

    def test_bulk_archive_archives_selected(self):
        self.client.force_login(self.user)
        d1 = self._make_doc()
        d2 = self._make_doc()
        response = self.client.post(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [d1.id, d2.id], "action": "archive"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated"], 2)
        d1.refresh_from_db()
        d2.refresh_from_db()
        self.assertTrue(d1.is_archived)
        self.assertTrue(d2.is_archived)

    def test_bulk_archive_restores_selected(self):
        self.client.force_login(self.user)
        d1 = self._make_doc()
        d1.is_archived = True
        d1.save(update_fields=["is_archived"])
        response = self.client.post(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [d1.id], "action": "restore"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated"], 1)
        d1.refresh_from_db()
        self.assertFalse(d1.is_archived)

    def test_bulk_archive_ignores_other_data_room_ids(self):
        self.client.force_login(self.user)
        other_data_room = DataRoom.objects.create(name="Other2", slug="other2", created_by=self.user)
        other_doc = DataRoomDocument.objects.create(
            data_room=other_data_room, uploaded_by=self.user,
            original_filename="other.txt", status=DataRoomDocument.Status.READY,
        )
        response = self.client.post(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [other_doc.id], "action": "archive"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated"], 0)
        other_doc.refresh_from_db()
        self.assertFalse(other_doc.is_archived)

    def test_bulk_archive_rejects_invalid_action(self):
        self.client.force_login(self.user)
        d1 = self._make_doc()
        response = self.client.post(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [d1.id], "action": "invalid"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_bulk_archive_blocks_other_user(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"document_ids": [doc.id], "action": "archive"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_bulk_archive_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("document_bulk_archive", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    # ------------------------------------------------------------------ #
    # document_status                                                      #
    # ------------------------------------------------------------------ #

    def test_document_status_returns_all_statuses(self):
        self.client.force_login(self.user)
        d1 = self._make_doc()  # READY
        d2 = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="fail.txt", status=DataRoomDocument.Status.FAILED,
        )
        response = self.client.get(
            reverse("document_status", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["statuses"][str(d1.id)], "ready")
        self.assertEqual(data["statuses"][str(d2.id)], "failed")

    def test_document_status_blocks_other_user(self):
        self.client.force_login(self.other)
        response = self.client.get(
            reverse("document_status", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 403)

    def test_document_status_requires_get(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("document_status", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    # ------------------------------------------------------------------ #
    # FAILED doc visibility                                                #
    # ------------------------------------------------------------------ #

    def test_failed_doc_visible_in_data_room_documents(self):
        self.client.force_login(self.user)
        failed_doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="broken.txt", status=DataRoomDocument.Status.FAILED,
        )
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 200)
        doc_ids = [d.id for d in response.context["documents"]]
        self.assertIn(failed_doc.id, doc_ids)

    # ------------------------------------------------------------------ #
    # data_room_list POST — description                                    #
    # ------------------------------------------------------------------ #

    def test_create_data_room_with_description(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_list"),
            {"name": "Described Room", "description": "Contains patents"},
        )
        self.assertEqual(response.status_code, 302)
        dr = DataRoom.objects.get(name="Described Room")
        self.assertEqual(dr.description, "Contains patents")

    def test_create_data_room_without_description(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_list"),
            {"name": "No Desc Room"},
        )
        self.assertEqual(response.status_code, 302)
        dr = DataRoom.objects.get(name="No Desc Room")
        self.assertEqual(dr.description, "")

    # ------------------------------------------------------------------ #
    # data_room_update_description                                         #
    # ------------------------------------------------------------------ #

    def test_update_description_saves(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_update_description", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"description": "Updated desc"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.data_room.refresh_from_db()
        self.assertEqual(self.data_room.description, "Updated desc")

    def test_update_description_truncates_at_1000(self):
        self.client.force_login(self.user)
        long_desc = "x" * 2000
        self.client.post(
            reverse("data_room_update_description", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"description": long_desc}),
            content_type="application/json",
        )
        self.data_room.refresh_from_db()
        self.assertLessEqual(len(self.data_room.description), 1000)

    def test_update_description_blocks_other_user(self):
        self.client.force_login(self.other)
        response = self.client.post(
            reverse("data_room_update_description", kwargs={"data_room_id": self.data_room.uuid}),
            data=json.dumps({"description": "hacked"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_update_description_rejects_bad_json(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_update_description", kwargs={"data_room_id": self.data_room.uuid}),
            data="not json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------ #
    # data_room_generate_description                                       #
    # ------------------------------------------------------------------ #

    @patch("documents.services.data_room_description.generate_data_room_description")
    def test_generate_description_returns_result(self, mock_gen):
        mock_gen.return_value = "AI-generated description"
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("data_room_generate_description", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["description"], "AI-generated description")

    # ------------------------------------------------------------------ #
    # View document button visibility                                      #
    # ------------------------------------------------------------------ #

    def test_view_document_button_shown_for_ready_doc(self):
        self.client.force_login(self.user)
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="ready.txt", status=DataRoomDocument.Status.READY,
        )
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertContains(response, "View document")
        self.assertContains(response, "data-chunks-url")

    def test_view_document_button_hidden_for_processing_doc(self):
        self.client.force_login(self.user)
        DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="pending.txt", status=DataRoomDocument.Status.PROCESSING,
        )
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertNotContains(response, "view-document-btn")

    def test_view_document_button_in_archived_section(self):
        self.client.force_login(self.user)
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="archived-ready.txt",
            status=DataRoomDocument.Status.READY,
            is_archived=True,
        )
        response = self.client.get(
            reverse("data_room_documents", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertContains(response, "View document")
        self.assertContains(
            response,
            reverse("document_chunks", kwargs={"data_room_id": self.data_room.uuid, "document_id": doc.id}),
        )

    def test_generate_description_blocks_other_user(self):
        self.client.force_login(self.other)
        response = self.client.post(
            reverse("data_room_generate_description", kwargs={"data_room_id": self.data_room.uuid}),
        )
        self.assertEqual(response.status_code, 403)
