from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument

User = get_user_model()


class DocumentStorageDeletionTests(TestCase):
    """The original binary in storage must be removed when a document row is deleted."""

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
        self.user = User.objects.create_user(email="sig@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="Sig", slug="sig", created_by=self.user)

    def _make_doc(self, filename: str = "hello.txt", content: bytes = b"hello") -> DataRoomDocument:
        upload = SimpleUploadedFile(filename, content, content_type="text/plain")
        return DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_file=upload,
            original_filename=filename,
            status=DataRoomDocument.Status.UPLOADED,
        )

    def test_single_document_delete_removes_file(self):
        doc = self._make_doc()
        stored_path = Path(doc.original_file.path)
        self.assertTrue(stored_path.exists(), "setup precondition: uploaded file exists")

        doc.delete()
        self.assertFalse(stored_path.exists(), "original binary must be removed on delete")

    def test_data_room_delete_cascades_to_file_removal(self):
        doc1 = self._make_doc("a.txt", b"aaa")
        doc2 = self._make_doc("b.txt", b"bbb")
        paths = [Path(doc1.original_file.path), Path(doc2.original_file.path)]
        for p in paths:
            self.assertTrue(p.exists())

        self.data_room.delete()

        for p in paths:
            self.assertFalse(p.exists(), f"{p} must be removed when the parent data room is deleted")

    def test_bulk_queryset_delete_removes_files(self):
        doc1 = self._make_doc("a.txt", b"aaa")
        doc2 = self._make_doc("b.txt", b"bbb")
        paths = [Path(doc1.original_file.path), Path(doc2.original_file.path)]
        for p in paths:
            self.assertTrue(p.exists())

        DataRoomDocument.objects.filter(pk__in=[doc1.pk, doc2.pk]).delete()

        for p in paths:
            self.assertFalse(p.exists(), f"{p} must be removed on bulk queryset delete")

    def test_delete_without_file_does_not_raise(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="no-file.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )
        doc.delete()
        self.assertFalse(DataRoomDocument.objects.filter(pk=doc.pk).exists())

    def test_storage_delete_failure_does_not_block_db_delete(self):
        doc = self._make_doc()
        doc_pk = doc.pk

        with patch.object(
            doc.original_file.storage,
            "delete",
            side_effect=OSError("simulated S3 outage"),
        ):
            doc.delete()

        self.assertFalse(
            DataRoomDocument.objects.filter(pk=doc_pk).exists(),
            "DB row should be deleted even when storage delete fails",
        )


@patch("documents.services.vector_store.delete_vectors_for_document")
class DocumentVectorDeletionTests(TestCase):
    """Embedding rows (which hold the full chunk text) must be removed from the
    vector store when a document row is deleted — GDPR erasure."""

    def setUp(self):
        self.user = User.objects.create_user(email="vec@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="Vec", slug="vec", created_by=self.user)

    def _make_doc(self, filename: str = "a.txt") -> DataRoomDocument:
        return DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename=filename,
            status=DataRoomDocument.Status.READY,
        )

    def test_single_document_delete_removes_vectors(self, mock_delete):
        doc = self._make_doc()
        doc_pk = doc.pk

        doc.delete()

        mock_delete.assert_called_once_with(doc_pk)

    def test_data_room_delete_cascades_to_vector_removal(self, mock_delete):
        doc1 = self._make_doc("a.txt")
        doc2 = self._make_doc("b.txt")
        pks = {doc1.pk, doc2.pk}

        self.data_room.delete()

        self.assertEqual({c.args[0] for c in mock_delete.call_args_list}, pks)

    def test_bulk_queryset_delete_removes_vectors(self, mock_delete):
        doc1 = self._make_doc("a.txt")
        doc2 = self._make_doc("b.txt")
        pks = {doc1.pk, doc2.pk}

        DataRoomDocument.objects.filter(pk__in=pks).delete()

        self.assertEqual({c.args[0] for c in mock_delete.call_args_list}, pks)

    def test_vector_delete_failure_does_not_block_db_delete(self, mock_delete):
        mock_delete.side_effect = RuntimeError("simulated pgvector outage")
        doc = self._make_doc()
        doc_pk = doc.pk

        doc.delete()

        self.assertFalse(
            DataRoomDocument.objects.filter(pk=doc_pk).exists(),
            "DB row should be deleted even when vector delete fails",
        )
