"""
Integration test: upload → process → retrieve flow.
Uses Celery eager mode and disables vector store (no OpenAI/pgvector) so the test runs without external services.
"""
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from documents.models import Project, ProjectDocument

User = get_user_model()


@override_settings(
    CELERY_TASK_ALWAYS_EAGER=True,
    PGVECTOR_CONNECTION="",  # Disable vector store so no OpenAI/pgvector calls
)
class UploadProcessRetrieveIntegrationTest(TestCase):
    """Test full flow: upload file → task processes → document READY → chunks API returns data."""

    def setUp(self):
        self.user = User.objects.create_user(email="integration@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.project = Project.objects.create(name="Integration Project", slug="integration-project", created_by=self.user)

    def test_upload_process_then_retrieve_chunks(self):
        content = b"First paragraph.\n\nSecond paragraph with more text for chunking."
        uploaded = SimpleUploadedFile("integration.txt", content, content_type="text/plain")
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("document_upload", kwargs={"project_id": self.project.uuid}),
            {"file": uploaded},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)

        doc = ProjectDocument.objects.get(project=self.project, original_filename="integration.txt")
        self.assertEqual(doc.status, ProjectDocument.Status.READY, doc.processing_error or "expected READY")
        self.assertGreater(doc.chunks.count(), 0, "chunks should be created")

        chunks_response = self.client.get(
            reverse("document_chunks", kwargs={"project_id": self.project.uuid, "document_id": doc.id})
        )
        self.assertEqual(chunks_response.status_code, 200)
        data = chunks_response.json()
        self.assertIn("chunks", data)
        self.assertEqual(len(data["chunks"]), doc.chunks.count())
        self.assertTrue(data["chunks"][0]["text"].startswith("First paragraph."))
