from django.contrib.auth import get_user_model
from django.test import TestCase

from documents.models import Project, ProjectDocument, ProjectDocumentChunk

User = get_user_model()


class ProjectModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")

    def test_create_project(self):
        project = Project.objects.create(name="Test Project", slug="test-project", created_by=self.user)
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.slug, "test-project")
        self.assertEqual(project.created_by, self.user)

    def test_project_str(self):
        project = Project.objects.create(name="My Project", slug="my-project", created_by=self.user)
        self.assertIn("My Project", str(project))


class ProjectDocumentModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.project = Project.objects.create(name="P", slug="p", created_by=self.user)

    def test_create_document(self):
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=ProjectDocument.Status.UPLOADED,
        )
        self.assertEqual(doc.status, ProjectDocument.Status.UPLOADED)
        self.assertEqual(doc.original_filename, "doc.txt")

    def test_document_str(self):
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="file.pdf",
            status=ProjectDocument.Status.READY,
        )
        self.assertIn("file.pdf", str(doc))

    def test_document_default_status_is_uploaded(self):
        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="default.txt",
        )
        self.assertEqual(doc.status, ProjectDocument.Status.UPLOADED)


class ProjectDocumentChunkModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.project = Project.objects.create(name="P", slug="p", created_by=self.user)
        self.doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=ProjectDocument.Status.READY,
        )

    def test_create_chunk(self):
        chunk = ProjectDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="Hello world",
            token_count=2,
        )
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.text, "Hello world")

    def test_unique_chunk_index_per_document(self):
        ProjectDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="First",
            token_count=1,
        )
        with self.assertRaises(Exception):
            ProjectDocumentChunk.objects.create(
                document=self.doc,
                chunk_index=0,
                text="Second",
                token_count=1,
            )

    def test_chunk_str(self):
        chunk = ProjectDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="Some text",
            token_count=2,
        )
        # Should not raise; result should be a non-empty string
        self.assertIsInstance(str(chunk), str)
        self.assertTrue(len(str(chunk)) > 0)


class ProjectDocumentStatusTests(TestCase):
    def test_all_four_statuses_exist(self):
        self.assertEqual(ProjectDocument.Status.UPLOADED, "uploaded")
        self.assertEqual(ProjectDocument.Status.PROCESSING, "processing")
        self.assertEqual(ProjectDocument.Status.READY, "ready")
        self.assertEqual(ProjectDocument.Status.FAILED, "failed")
