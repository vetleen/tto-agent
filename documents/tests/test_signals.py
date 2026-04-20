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
