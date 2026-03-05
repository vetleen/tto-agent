from django.contrib.auth import get_user_model
from django.test import TestCase

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk

User = get_user_model()


class DataRoomModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")

    def test_create_data_room(self):
        data_room = DataRoom.objects.create(name="Test Project", slug="test-project", created_by=self.user)
        self.assertEqual(data_room.name, "Test Project")
        self.assertEqual(data_room.slug, "test-project")
        self.assertEqual(data_room.created_by, self.user)

    def test_data_room_str(self):
        data_room = DataRoom.objects.create(name="My Project", slug="my-project", created_by=self.user)
        self.assertIn("My Project", str(data_room))


class DataRoomDocumentModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="P", slug="p", created_by=self.user)

    def test_create_document(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="doc.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )
        self.assertEqual(doc.status, DataRoomDocument.Status.UPLOADED)
        self.assertEqual(doc.original_filename, "doc.txt")

    def test_document_str(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="file.pdf",
            status=DataRoomDocument.Status.READY,
        )
        self.assertIn("file.pdf", str(doc))

    def test_document_default_status_is_uploaded(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="default.txt",
        )
        self.assertEqual(doc.status, DataRoomDocument.Status.UPLOADED)


class DataRoomDocumentChunkModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="P", slug="p", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=DataRoomDocument.Status.READY,
        )

    def test_create_chunk(self):
        chunk = DataRoomDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="Hello world",
            token_count=2,
        )
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.text, "Hello world")

    def test_unique_chunk_index_per_document(self):
        DataRoomDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="First",
            token_count=1,
        )
        with self.assertRaises(Exception):
            DataRoomDocumentChunk.objects.create(
                document=self.doc,
                chunk_index=0,
                text="Second",
                token_count=1,
            )

    def test_chunk_str(self):
        chunk = DataRoomDocumentChunk.objects.create(
            document=self.doc,
            chunk_index=0,
            text="Some text",
            token_count=2,
        )
        # Should not raise; result should be a non-empty string
        self.assertIsInstance(str(chunk), str)
        self.assertTrue(len(str(chunk)) > 0)


class DataRoomDocumentStatusTests(TestCase):
    def test_all_four_statuses_exist(self):
        self.assertEqual(DataRoomDocument.Status.UPLOADED, "uploaded")
        self.assertEqual(DataRoomDocument.Status.PROCESSING, "processing")
        self.assertEqual(DataRoomDocument.Status.READY, "ready")
        self.assertEqual(DataRoomDocument.Status.FAILED, "failed")
