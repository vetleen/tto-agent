"""Smoke tests for document versioning: retrieval gating, sensitivity union,
status, and rollback — at the model/service level, no LLM/embedding/Celery."""
from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase

from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
    DataRoomDocumentVersion,
)
from documents.services import retrieval
from documents.services.versioning import (
    document_status,
    recompute_document_sensitivity,
    restore_version,
)

User = get_user_model()
READY = DataRoomDocument.Status.READY


class VersioningSmokeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="u@x.io", password="p")
        self.room = DataRoom.objects.create(name="R", slug="r", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="d.md", status=READY,
        )

    def _version(self, idx, *, searchable, status=READY, quarantined=False, chunk_texts=()):
        v = DataRoomDocumentVersion.objects.create(
            document=self.doc, version_index=idx, status=status,
            is_searchable=searchable, is_quarantined=quarantined,
            quarantine_reason="GDPR Article 9" if quarantined else "",
        )
        for i, t in enumerate(chunk_texts):
            DataRoomDocumentChunk.objects.create(
                version=v, chunk_index=i, text=t, token_count=len(t.split()),
            )
        return v

    def test_retrieval_returns_only_active_version_chunks(self):
        v0 = self._version(0, searchable=True, chunk_texts=["alpha beta", "gamma delta"])
        v1 = self._version(1, searchable=False, chunk_texts=["v1 stale text"])
        self.doc.current_version = v1
        self.doc.active_searchable_version = v0
        self.doc.save(update_fields=["current_version", "active_searchable_version"])

        chunks = retrieval.get_chunks_by_document(self.doc.id)
        texts = {c["text"] for c in chunks}
        self.assertEqual(texts, {"alpha beta", "gamma delta"})
        self.assertNotIn("v1 stale text", texts)

    def test_sensitivity_union_keeps_doc_flagged_while_v0_quarantined(self):
        # v0 (original) contained Article-9 data; v1 (clean edit) does not.
        self._version(0, searchable=False, quarantined=True, chunk_texts=["sensitive"])
        v1 = self._version(1, searchable=True, quarantined=False, chunk_texts=["clean"])
        self.doc.current_version = v1
        self.doc.active_searchable_version = v1
        self.doc.save(update_fields=["current_version", "active_searchable_version"])

        recompute_document_sensitivity(self.doc.id)
        self.doc.refresh_from_db()
        # The clean v1 is live and retrievable...
        chunks = retrieval.get_chunks_by_document(self.doc.id)
        self.assertEqual({c["text"] for c in chunks}, {"clean"})
        # ...but the document stays flagged because the original v0 still has it.
        self.assertTrue(self.doc.is_quarantined)

    def test_document_status_reports_processing_when_current_ahead_of_active(self):
        v0 = self._version(0, searchable=True)
        v1 = self._version(1, searchable=False, status=DataRoomDocument.Status.SCANNING)
        self.doc.current_version = v1
        self.doc.active_searchable_version = v0
        self.doc.save(update_fields=["current_version", "active_searchable_version"])

        status = document_status(self.doc)
        self.assertEqual(status["state"], "processing")
        self.assertEqual(status["active_version"], 0)
        self.assertEqual(status["current_version"], 1)

    def test_restore_flips_pointers(self):
        v0 = self._version(0, searchable=False, chunk_texts=["original"])
        v1 = self._version(1, searchable=True, chunk_texts=["edited"])
        self.doc.current_version = v1
        self.doc.active_searchable_version = v1
        self.doc.save(update_fields=["current_version", "active_searchable_version"])

        restore_version(self.doc, v0)
        self.doc.refresh_from_db()
        v0.refresh_from_db()
        v1.refresh_from_db()
        self.assertEqual(self.doc.active_searchable_version_id, v0.id)
        self.assertEqual(self.doc.current_version_id, v0.id)
        self.assertTrue(v0.is_searchable)
        self.assertFalse(v1.is_searchable)
        # Retrieval now serves the restored version.
        self.assertEqual(
            {c["text"] for c in retrieval.get_chunks_by_document(self.doc.id)}, {"original"}
        )

    def test_restore_rejects_quarantined_version(self):
        v0 = self._version(0, searchable=False, quarantined=True, chunk_texts=["bad"])
        v1 = self._version(1, searchable=True, chunk_texts=["good"])
        self.doc.current_version = v1
        self.doc.active_searchable_version = v1
        self.doc.save(update_fields=["current_version", "active_searchable_version"])

        with self.assertRaises(ValueError):
            restore_version(self.doc, v0)

    def test_tag_unique_per_version(self):
        v0 = self._version(0, searchable=True)
        v1 = self._version(1, searchable=False)
        # Same key allowed on different versions...
        DataRoomDocumentTag.objects.create(version=v0, key="document_type", value="patent")
        DataRoomDocumentTag.objects.create(version=v1, key="document_type", value="memo")
        self.assertEqual(DataRoomDocumentTag.objects.filter(key="document_type").count(), 2)
