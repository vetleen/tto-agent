"""Tests for synchronous scan-at-save (documents.services.sync_scan) and the shared
``finalize_version`` core extracted from the ``finalize_document_metadata`` task.

The scan LLM calls (guardrail chunk scan + PII category scan) are mocked so the tests
are deterministic; the point is the orchestration and verdict mapping, not the models.
"""
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentVersion,
)

User = get_user_model()

# Resolves the document_description / pii_scan feature models to a real id.
_MODELS = dict(LLM_DEFAULT_MID_MODEL="openai/gpt-4o-mini", LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
_GATE_ON = ("openai/gpt-4o-mini", True, True)  # (model, pii_enabled, pii_quarantine_enabled)
Origin = DataRoomDocumentVersion.Origin
Status = DataRoomDocument.Status


def _clean_scan(version):
    return None


def _partial_scan(version):
    """Mimic a chunk-level quarantine (e.g. prompt injection): flag the first chunk and
    set the version's partial flag, WITHOUT version-level is_quarantined."""
    chunk = version.chunks.first()
    if chunk:
        DataRoomDocumentChunk.objects.filter(pk=chunk.pk).update(
            is_quarantined=True,
            quarantine_reason="Reviewer: prompt_injection (confidence: 0.95)",
        )
    DataRoomDocumentVersion.objects.filter(pk=version.id).update(is_partially_quarantined=True)


def _no_pii(*args, **kwargs):
    return {}


def _article9(*args, **kwargs):
    return {"pii_special_category": True}


class SyncScanOrchestratorTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sync@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="Sync", slug="sync", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.md", status=Status.UPLOADED,
        )

    def _agent_version(self, content="Some clean content about widgets.\n\nMore detail follows here."):
        from documents.services.versioning import create_version
        return create_version(
            self.doc, content=content, origin=Origin.CANVAS_EXPORT,
            created_by=self.user, enqueue=False,
        )

    def _run(self, chunk_scan, pii):
        from documents.services.sync_scan import scan_version_synchronously
        version = self._agent_version()
        with override_settings(**_MODELS), \
             patch("guardrails.tasks._scan_chunks_for_version", side_effect=chunk_scan), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", side_effect=pii), \
             patch("documents.services.pii_scan.resolve_pii_gate", return_value=_GATE_ON), \
             patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "d", "tags": {}, "document_date": None}):
            return version, scan_version_synchronously(version.id)

    def test_clean_save_becomes_active(self):
        version, verdict = self._run(_clean_scan, _no_pii)
        self.assertEqual(verdict.status, "clean")
        self.assertTrue(verdict.became_active)
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_searchable_version_id, version.id)

    def test_partial_quarantine_is_warn_and_still_active(self):
        version, verdict = self._run(_partial_scan, _no_pii)
        self.assertEqual(verdict.status, "warn")
        self.assertTrue(verdict.became_active)  # a flagged chunk does NOT block the version
        self.assertIn("prompt_injection", (verdict.reviewer_reasoning or ""))
        self.doc.refresh_from_db()
        self.assertEqual(self.doc.active_searchable_version_id, version.id)

    def test_article_9_10_is_blocked_and_not_active(self):
        version, verdict = self._run(_clean_scan, _article9)
        self.assertEqual(verdict.status, "blocked")
        self.assertTrue(verdict.is_quarantined)
        self.assertFalse(verdict.became_active)
        self.assertIn("Article 9", verdict.reasons[0])
        self.doc.refresh_from_db()
        self.assertIsNone(self.doc.active_searchable_version_id)

    def test_scan_error_is_scan_failed(self):
        def _boom(version):
            raise RuntimeError("classifier down")

        version, verdict = self._run(_boom, _no_pii)
        self.assertEqual(verdict.status, "scan_failed")
        version.refresh_from_db()
        self.assertEqual(version.status, Status.SCAN_FAILED)


class DispatchScanFlagTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="ds@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="DS", slug="ds", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="d.md", status=Status.UPLOADED,
        )

    def _version(self):
        from documents.services.versioning import create_version
        return create_version(self.doc, content="Body text about things.", origin=Origin.AGENT_CREATED,
                              created_by=self.user, enqueue=False)

    def test_dispatch_scan_false_suppresses_async_handoff(self):
        from documents.services.process_document import process_document_version
        version = self._version()
        with patch("guardrails.tasks.scan_document_version.delay") as mock_delay:
            process_document_version(version.id, dispatch_scan=False)
        mock_delay.assert_not_called()
        version.refresh_from_db()
        self.assertEqual(version.status, Status.SCANNING)

    def test_dispatch_scan_true_hands_off(self):
        from documents.services.process_document import process_document_version
        version = self._version()
        with patch("guardrails.tasks.scan_document_version.delay") as mock_delay:
            process_document_version(version.id)
        mock_delay.assert_called_once()


class FinalizeVersionEagerTests(TestCase):
    """The eager core handles a gated PII failure terminally — no Celery retry."""

    def setUp(self):
        self.user = User.objects.create_user(email="fv@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="FV", slug="fv", created_by=self.user)

    def _held_version(self):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="h.md", status=Status.SCANNING,
        )
        make_version(doc, status=Status.SCANNING, searchable=False,
                     origin=Origin.CANVAS_EXPORT, chunks=[{"text": "content"}],
                     make_active=False)
        return doc

    @override_settings(**_MODELS)
    def test_eager_pii_failure_marks_scan_failed_without_retry(self):
        from documents.tasks import finalize_version
        doc = self._held_version()
        with patch("documents.services.pii_scan.resolve_pii_gate", return_value=_GATE_ON), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   side_effect=RuntimeError("LLM down")), \
             patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "d", "tags": {}, "document_date": None}):
            finalize_version(doc.current_version_id, eager=True)  # must not raise / retry
        version = DataRoomDocumentVersion.objects.get(pk=doc.current_version_id)
        self.assertEqual(version.status, Status.SCAN_FAILED)

    @override_settings(**_MODELS)
    def test_eager_clean_releases_and_advances(self):
        from documents.tasks import finalize_version
        doc = self._held_version()
        with patch("documents.services.pii_scan.resolve_pii_gate", return_value=_GATE_ON), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}), \
             patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "d", "tags": {}, "document_date": None}):
            finalize_version(doc.current_version_id, eager=True)
        doc.refresh_from_db()
        self.assertEqual(doc.active_searchable_version_id, doc.current_version_id)
