from io import BytesIO

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Project, ProjectDocument

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class DocumentViewsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.project = Project.objects.create(name="Test", slug="test", created_by=self.user)

    def test_project_list_requires_login(self):
        response = self.client.get(reverse("project_list"))
        self.assertEqual(response.status_code, 302)

    def test_project_list_renders_for_owner(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("project_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test")

    def test_project_detail_requires_login(self):
        response = self.client.get(reverse("project_detail", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 302)

    def test_project_detail_owner_sees_upload_form(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("project_detail", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Upload document")
        self.assertContains(response, "name=\"file\"")

    def test_project_detail_other_user_redirected(self):
        other = User.objects.create_user(email="other@example.com", password="testpass")
        self.client.force_login(other)
        response = self.client.get(reverse("project_detail", kwargs={"project_id": self.project.uuid}))
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("project_list"))

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
