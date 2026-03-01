import sys
import types
from datetime import timedelta
from io import BytesIO
from unittest.mock import Mock, call, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from documents.models import Project, ProjectDocument
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
        self.project = Project.objects.create(name="Test", slug="test", created_by=self.user)
        self.other = User.objects.create_user(email="other@example.com", password="testpass")

    def _make_doc(self):
        return ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=ProjectDocument.Status.READY,
        )

    def test_project_list_requires_login(self):
        response = self.client.get(reverse("project_list"))
        self.assertEqual(response.status_code, 302)

    def test_project_create_retries_after_integrity_error(self):
        self.client.force_login(self.user)

        created_project = Project(name="Test Retry", slug="test-retry-1", created_by=self.user)
        with patch("documents.views.Project.objects.create", side_effect=[IntegrityError(), created_project]) as mock_create:
            response = self.client.post(reverse("project_list"), {"name": "Test Retry"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(mock_create.call_count, 2)
        self.assertEqual(mock_create.call_args_list[0], call(name="Test Retry", slug="test-retry", created_by=self.user))
        self.assertEqual(mock_create.call_args_list[1], call(name="Test Retry", slug="test-retry-1", created_by=self.user))

    def test_project_list_renders_for_owner(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("project_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test")

    def test_project_detail_requires_login(self):
        response = self.client.get(reverse("project_detail", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 302)

    def test_project_detail_redirects_to_chat(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("project_detail", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project_chat", kwargs={"project_id": self.project.uuid}))

    def test_project_documents_owner_sees_upload_form(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("project_documents", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload document")
        self.assertContains(response, "name=\"file\"")

    def test_project_detail_other_user_redirected(self):
        self.client.force_login(self.other)
        response = self.client.get(
            reverse("project_detail", kwargs={"project_id": self.project.uuid}),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.resolver_match.url_name, "project_list")

    def test_document_upload_creates_document_and_redirects(self):
        self.client.force_login(self.user)
        content = b"Hello world"
        f = BytesIO(content)
        f.name = "test.txt"
        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ProjectDocument.objects.filter(project=self.project).count(), 1)
        doc = ProjectDocument.objects.get(project=self.project)
        self.assertEqual(doc.original_filename, "test.txt")
        # Status may be UPLOADED (task queued), READY (sync processed), or FAILED (sync ran but deps missing)
        self.assertIn(
            doc.status,
            (ProjectDocument.Status.UPLOADED, ProjectDocument.Status.READY, ProjectDocument.Status.FAILED),
        )

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
                    reverse("document_upload", kwargs={"project_id": self.project.uuid}),
                    {"file": f},
                    follow=True,
                )

        self.assertEqual(response.status_code, 200)
        doc = ProjectDocument.objects.get(project=self.project, original_filename="enqueue-fail.txt")
        self.assertEqual(doc.status, ProjectDocument.Status.FAILED)
        self.assertIn("broker unavailable", doc.processing_error)
        self.assertContains(response, "processing could not be started")

    @override_settings(DOCUMENT_ALLOWED_MIME_TYPES=set())
    def test_document_upload_allows_mime_when_allowlist_not_configured(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("mime-ok.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(ProjectDocument.objects.filter(project=self.project, original_filename="mime-ok.txt").count(), 1)

    @override_settings(DOCUMENT_ALLOWED_MIME_TYPES={"application/pdf"})
    def test_document_upload_rejects_mime_not_in_allowlist(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("mime-blocked.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(ProjectDocument.objects.filter(project=self.project, original_filename="mime-blocked.txt").count(), 0)
        self.assertContains(response, "Unsupported file type")

    def test_document_upload_sanitizes_path_from_original_filename(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile(r"..\..\secret\report.txt", b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        doc = ProjectDocument.objects.get(project=self.project)
        self.assertEqual(doc.original_filename, "report.txt")

    def test_document_upload_truncates_original_filename_to_model_limit(self):
        self.client.force_login(self.user)
        long_name = f"{'a' * 300}.txt"
        f = SimpleUploadedFile(long_name, b"hello", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        doc = ProjectDocument.objects.get(project=self.project)
        self.assertLessEqual(len(doc.original_filename), 255)
        self.assertTrue(doc.original_filename.endswith('.txt'))

    def test_document_upload_rejects_empty_file(self):
        self.client.force_login(self.user)
        f = SimpleUploadedFile("empty.txt", b"", content_type="text/plain")

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(ProjectDocument.objects.filter(project=self.project, original_filename="empty.txt").count(), 0)
        self.assertContains(response, "File is empty")

    def test_document_upload_rejects_unsupported_extension(self):
        self.client.force_login(self.user)
        f = BytesIO(b"x")
        f.name = "file.xyz"
        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ProjectDocument.objects.filter(project=self.project).count(), 0)
        self.assertContains(response, "Unsupported file type")

    def test_document_chunks_api_requires_auth(self):
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=ProjectDocument.Status.READY,
        )
        response = self.client.get(
            reverse("document_chunks", kwargs={"project_id": self.project.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 302)

    def test_document_chunks_api_returns_ordered_chunks(self):
        self.client.force_login(self.user)
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=ProjectDocument.Status.READY,
        )
        from documents.models import ProjectDocumentChunk
        ProjectDocumentChunk.objects.create(document=doc, chunk_index=0, text="A", token_count=1)
        ProjectDocumentChunk.objects.create(document=doc, chunk_index=1, text="B", token_count=1)
        response = self.client.get(
            reverse("document_chunks", kwargs={"project_id": self.project.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["chunks"]), 2)
        self.assertEqual(data["chunks"][0]["text"], "A")
        self.assertEqual(data["chunks"][1]["text"], "B")

    # ------------------------------------------------------------------ #
    # project_delete                                                       #
    # ------------------------------------------------------------------ #

    def test_project_delete_removes_project(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("project_delete", kwargs={"project_id": self.project.uuid}),
        )
        self.assertRedirects(response, reverse("project_list"))
        self.assertFalse(Project.objects.filter(pk=self.project.pk).exists())

    def test_project_delete_other_user_blocked(self):
        self.client.force_login(self.other)
        self.client.post(
            reverse("project_delete", kwargs={"project_id": self.project.uuid}),
        )
        self.assertTrue(Project.objects.filter(pk=self.project.pk).exists())

    def test_project_delete_requires_post(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("project_delete", kwargs={"project_id": self.project.uuid}),
        )
        self.assertEqual(response.status_code, 405)

    # ------------------------------------------------------------------ #
    # project_rename                                                       #
    # ------------------------------------------------------------------ #

    def test_project_rename_updates_name(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("project_rename", kwargs={"project_id": self.project.uuid}),
            {"name": "Renamed Project"},
        )
        self.assertRedirects(response, reverse("project_list"))
        self.project.refresh_from_db()
        self.assertEqual(self.project.name, "Renamed Project")

    def test_project_rename_rejects_empty_name(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("project_rename", kwargs={"project_id": self.project.uuid}),
            {"name": ""},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.project.refresh_from_db()
        self.assertEqual(self.project.name, "Test")

    def test_project_rename_truncates_at_255(self):
        self.client.force_login(self.user)
        long_name = "x" * 300
        self.client.post(
            reverse("project_rename", kwargs={"project_id": self.project.uuid}),
            {"name": long_name},
        )
        self.project.refresh_from_db()
        self.assertLessEqual(len(self.project.name), 255)

    def test_project_rename_other_user_blocked(self):
        self.client.force_login(self.other)
        self.client.post(
            reverse("project_rename", kwargs={"project_id": self.project.uuid}),
            {"name": "Hacked"},
        )
        self.project.refresh_from_db()
        self.assertEqual(self.project.name, "Test")

    # ------------------------------------------------------------------ #
    # document_delete                                                      #
    # ------------------------------------------------------------------ #

    def test_document_delete_removes_document(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_delete", kwargs={"project_id": self.project.uuid, "document_id": doc.id}),
        )
        self.assertRedirects(
            response,
            reverse("project_documents", kwargs={"project_id": self.project.uuid}),
        )
        self.assertFalse(ProjectDocument.objects.filter(pk=doc.pk).exists())

    def test_document_delete_other_user_blocked(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        self.client.post(
            reverse("document_delete", kwargs={"project_id": self.project.uuid, "document_id": doc.id}),
        )
        self.assertTrue(ProjectDocument.objects.filter(pk=doc.pk).exists())

    # ------------------------------------------------------------------ #
    # document_rename                                                      #
    # ------------------------------------------------------------------ #

    def test_document_rename_updates_filename(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        self.client.post(
            reverse("document_rename", kwargs={"project_id": self.project.uuid, "document_id": doc.id}),
            {"name": "renamed.txt"},
        )
        doc.refresh_from_db()
        self.assertEqual(doc.original_filename, "renamed.txt")

    def test_document_rename_rejects_empty_name(self):
        self.client.force_login(self.user)
        doc = self._make_doc()
        response = self.client.post(
            reverse("document_rename", kwargs={"project_id": self.project.uuid, "document_id": doc.id}),
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
            reverse("document_rename", kwargs={"project_id": self.project.uuid, "document_id": doc.id}),
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
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": f},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "too large")
        self.assertEqual(ProjectDocument.objects.filter(project=self.project, original_filename="big.txt").count(), 0)

    # ------------------------------------------------------------------ #
    # document_chunks API — additional authorization cases                #
    # ------------------------------------------------------------------ #

    def test_document_chunks_api_forbidden_for_other_user(self):
        self.client.force_login(self.other)
        doc = self._make_doc()
        response = self.client.get(
            reverse("document_chunks", kwargs={"project_id": self.project.uuid, "document_id": doc.id})
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"error": "Forbidden"})

    def test_document_chunks_api_returns_404_for_missing_doc(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("document_chunks", kwargs={"project_id": self.project.uuid, "document_id": 99999})
        )
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------ #
    # project_chat                                                         #
    # ------------------------------------------------------------------ #

    def test_project_chat_renders_for_owner(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("project_chat", kwargs={"project_id": self.project.uuid}),
        )
        self.assertEqual(response.status_code, 200)

    def test_project_chat_other_user_redirected(self):
        self.client.force_login(self.other)
        response = self.client.get(
            reverse("project_chat", kwargs={"project_id": self.project.uuid}),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.resolver_match.url_name, "project_list")
