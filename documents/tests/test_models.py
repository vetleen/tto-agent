from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
    DataRoomDocumentVersion,
)

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

    def test_same_slug_allowed_for_different_users(self):
        other = User.objects.create_user(email="other@example.com", password="testpass")
        DataRoom.objects.create(name="Finance", slug="finance", created_by=self.user)
        room = DataRoom.objects.create(name="Finance", slug="finance", created_by=other)
        self.assertEqual(room.slug, "finance")

    def test_duplicate_slug_for_same_user_rejected(self):
        DataRoom.objects.create(name="Finance", slug="finance", created_by=self.user)
        with self.assertRaises(IntegrityError):
            DataRoom.objects.create(name="Finance Again", slug="finance", created_by=self.user)


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
        self.version = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=0)

    def test_create_chunk(self):
        chunk = DataRoomDocumentChunk.objects.create(
            version=self.version,
            chunk_index=0,
            text="Hello world",
            token_count=2,
        )
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.text, "Hello world")

    def test_unique_chunk_index_per_version(self):
        DataRoomDocumentChunk.objects.create(
            version=self.version,
            chunk_index=0,
            text="First",
            token_count=1,
        )
        with self.assertRaises(Exception):
            DataRoomDocumentChunk.objects.create(
                version=self.version,
                chunk_index=0,
                text="Second",
                token_count=1,
            )

    def test_chunk_str(self):
        chunk = DataRoomDocumentChunk.objects.create(
            version=self.version,
            chunk_index=0,
            text="Some text",
            token_count=2,
        )
        # Should not raise; result should be a non-empty string
        self.assertIsInstance(str(chunk), str)
        self.assertTrue(len(str(chunk)) > 0)


class DataRoomDescriptionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")

    def test_data_room_description_default_empty(self):
        data_room = DataRoom.objects.create(name="Test", slug="test-desc", created_by=self.user)
        self.assertEqual(data_room.description, "")

    def test_data_room_description_saved(self):
        data_room = DataRoom.objects.create(
            name="Test", slug="test-desc2", created_by=self.user,
            description="Contains patent files",
        )
        data_room.refresh_from_db()
        self.assertEqual(data_room.description, "Contains patent files")


class DataRoomDocumentTagTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="taguser@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="P", slug="p-tag", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="x.txt",
            status=DataRoomDocument.Status.READY,
        )
        self.version = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=0)

    def test_create_tag(self):
        tag = DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Agreement")
        self.assertEqual(tag.key, "document_type")
        self.assertEqual(tag.value, "Agreement")
        self.assertEqual(tag.version_id, self.version.pk)

    def test_tag_str(self):
        tag = DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Patent")
        self.assertIn("document_type=Patent", str(tag))

    def test_unique_constraint_per_version_key(self):
        DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Agreement")
        with self.assertRaises(Exception):
            DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Report")

    def test_same_key_different_versions(self):
        version2 = DataRoomDocumentVersion.objects.create(document=self.doc, version_index=1)
        DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Agreement")
        DataRoomDocumentTag.objects.create(version=version2, key="document_type", value="Report")
        self.assertEqual(DataRoomDocumentTag.objects.count(), 2)

    def test_cascade_delete_with_document(self):
        DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Agreement")
        self.assertEqual(DataRoomDocumentTag.objects.count(), 1)
        self.doc.delete()  # cascades document -> version -> tag
        self.assertEqual(DataRoomDocumentTag.objects.count(), 0)

    def test_related_name_tags(self):
        DataRoomDocumentTag.objects.create(version=self.version, key="document_type", value="Agreement")
        DataRoomDocumentTag.objects.create(version=self.version, key="category", value="Legal")
        self.assertEqual(self.version.tags.count(), 2)


class DataRoomDocumentDocIndexTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="idx@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="Idx", slug="idx", created_by=self.user)

    def test_auto_assigns_sequential_doc_index(self):
        doc1 = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user, original_filename="a.txt",
        )
        doc2 = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user, original_filename="b.txt",
        )
        doc3 = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user, original_filename="c.txt",
        )
        self.assertEqual(doc1.doc_index, 1)
        self.assertEqual(doc2.doc_index, 2)
        self.assertEqual(doc3.doc_index, 3)

    def test_doc_index_scoped_to_data_room(self):
        other_room = DataRoom.objects.create(name="Other", slug="other", created_by=self.user)
        doc1 = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user, original_filename="a.txt",
        )
        doc2 = DataRoomDocument.objects.create(
            data_room=other_room, uploaded_by=self.user, original_filename="b.txt",
        )
        self.assertEqual(doc1.doc_index, 1)
        self.assertEqual(doc2.doc_index, 1)

    def test_preserves_explicit_doc_index(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="a.txt", doc_index=42,
        )
        self.assertEqual(doc.doc_index, 42)

    def test_continues_from_highest_existing_index(self):
        DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="a.txt", doc_index=10,
        )
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user, original_filename="b.txt",
        )
        self.assertEqual(doc.doc_index, 11)


class DataRoomDocumentStatusTests(TestCase):
    def test_all_statuses_exist(self):
        self.assertEqual(DataRoomDocument.Status.UPLOADED, "uploaded")
        self.assertEqual(DataRoomDocument.Status.PROCESSING, "processing")
        self.assertEqual(DataRoomDocument.Status.SCANNING, "scanning")
        self.assertEqual(DataRoomDocument.Status.SCAN_FAILED, "scan_failed")
        self.assertEqual(DataRoomDocument.Status.READY, "ready")
        self.assertEqual(DataRoomDocument.Status.FAILED, "failed")
