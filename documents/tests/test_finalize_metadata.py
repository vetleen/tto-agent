"""Tests for the deferred metadata pipeline introduced with the worker-memory refactor:

- ``finalize_document_metadata`` task (description + windowed PII, dispatched after READY)
- ``vector_store.add_chunk_vectors`` batching
- ``chunk_access`` keyset streaming + head/tail reconstruction
- ``pii_scan.scan_pii_categories_for_version`` windowed full-document scan
- ``process_document`` dispatching the finalize task
"""
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from documents.models import (
    DataRoom,
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentTag,
)

User = get_user_model()

try:
    import langchain_core  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Resolves both the document_description and pii_scan feature models to a real id.
_MODELS = dict(LLM_DEFAULT_MID_MODEL="openai/gpt-4o-mini", LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")


class FinalizeDocumentMetadataTests(TestCase):
    """Description + PII now run in finalize_document_metadata, not process_document."""

    def setUp(self):
        self.user = User.objects.create_user(email="fin@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="FinProject", slug="fin-project", created_by=self.user)

    def _ready_doc(self, texts=("Some content about Acme Corp.",), token_each=10):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="fin.txt",
            status=DataRoomDocument.Status.READY,
            token_count=token_each * len(texts),
        )
        # These tests exercise description/PII/quarantine on a READY (non-held)
        # version; the held-release path is covered by ScanGateTransitionTests.
        make_version(
            doc, chunks=[{"text": t, "token_count": token_each} for t in texts],
        )
        return doc

    @override_settings(**_MODELS)
    def test_description_and_tags_written(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "A description", "tags": {"document_type": "Report"}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertEqual(doc.description, "A description")
        self.assertEqual(DataRoomDocumentTag.objects.get(version__document=doc, key="document_type").value, "Report")

    @override_settings(**_MODELS)
    def test_description_failure_doesnt_raise(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   side_effect=RuntimeError("LLM down")), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}):
            finalize_document_metadata(doc.current_version_id)  # must not raise

        doc.refresh_from_db()
        self.assertFalse(doc.description)

    @override_settings(**_MODELS)
    def test_description_delete_race_logged_as_info(self):
        """A doc deleted during description generation logs at INFO from documents.tasks."""
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        original_save = DataRoomDocument.save

        def save_raising_notupdated(self, *args, **kwargs):
            if "description" in (kwargs.get("update_fields") or []):
                raise DataRoomDocument.NotUpdated("Save with update_fields did not affect any rows.")
            return original_save(self, *args, **kwargs)

        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "A description", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}), \
             patch.object(DataRoomDocument, "save", save_raising_notupdated):
            with self.assertLogs("documents.tasks", level="INFO") as cm:
                finalize_document_metadata(doc.current_version_id)

        log_output = "\n".join(cm.output)
        self.assertIn("deleted during description generation", log_output)
        self.assertNotIn("description generation failed", log_output)

    @override_settings(**_MODELS)
    def test_pii_tags_written(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_ordinary_identity": True, "pii_special_category": True}):
            finalize_document_metadata(doc.current_version_id)

        tags = dict(
            DataRoomDocumentTag.objects.filter(version__document=doc, key__startswith="pii_").values_list("key", "value")
        )
        self.assertEqual(tags.get("pii_ordinary_identity"), "true")
        self.assertEqual(tags.get("pii_special_category"), "true")

    @override_settings(**_MODELS)
    def test_pii_failure_doesnt_raise(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", side_effect=RuntimeError("boom")):
            finalize_document_metadata(doc.current_version_id)  # must not raise

        self.assertFalse(DataRoomDocumentTag.objects.filter(version__document=doc, key__startswith="pii_").exists())

    @override_settings(**_MODELS)
    def test_pii_respects_org_toggle(self):
        from accounts.models import Membership, Organization
        from documents.tasks import finalize_document_metadata

        org = Organization.objects.create(name="ToggleOrg", slug="toggle-org", preferences={"pii_scan_enabled": False})
        Membership.objects.create(user=self.user, org=org, role="admin")
        doc = self._ready_doc()

        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version") as mock_scan:
            finalize_document_metadata(doc.current_version_id)
            mock_scan.assert_not_called()

        self.assertFalse(DataRoomDocumentTag.objects.filter(version__document=doc, key__startswith="pii_").exists())

    @override_settings(**_MODELS)
    def test_quarantine_on_special_category(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_ordinary_identity": True, "pii_special_category": True}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertTrue(doc.is_quarantined)
        self.assertIn("Article 9", doc.quarantine_reason)

    @override_settings(**_MODELS)
    def test_quarantine_on_criminal_offence(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_criminal_offence": True}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertTrue(doc.is_quarantined)
        self.assertIn("Article 10", doc.quarantine_reason)

    @override_settings(**_MODELS)
    def test_no_quarantine_for_ordinary_only(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_ordinary_identity": True}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertFalse(doc.is_quarantined)
        self.assertEqual(doc.quarantine_reason, "")

    @override_settings(**_MODELS)
    def test_quarantine_respects_org_toggle(self):
        """pii_quarantine_enabled=False leaves Art. 9 docs un-quarantined (tags still written)."""
        from accounts.models import Membership, Organization
        from documents.tasks import finalize_document_metadata

        org = Organization.objects.create(
            name="QuarOffOrg", slug="quar-off-org", preferences={"pii_quarantine_enabled": False},
        )
        Membership.objects.create(user=self.user, org=org, role="admin")
        doc = self._ready_doc()

        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value={"description": "", "tags": {}, "document_date": None}), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_special_category": True}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertFalse(doc.is_quarantined)
        self.assertTrue(
            DataRoomDocumentTag.objects.filter(version__document=doc, key="pii_special_category").exists()
        )

    def test_skips_when_no_models_configured(self):
        from documents.tasks import finalize_document_metadata

        doc = self._ready_doc()
        # No feature models resolved -> finalize returns before touching chunks/LLMs.
        with patch("core.preferences.resolve_org_feature_model", return_value=""), \
             patch("documents.services.description.generate_description_and_tags_from_text") as mock_desc, \
             patch("documents.services.pii_scan.scan_pii_categories_for_version") as mock_pii:
            finalize_document_metadata(doc.current_version_id)
            mock_desc.assert_not_called()
            mock_pii.assert_not_called()

    def test_missing_document_is_noop(self):
        from documents.tasks import finalize_document_metadata

        finalize_document_metadata(999999)  # nonexistent — must not raise


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain_core not installed")
class AddChunkVectorsBatchingTests(TestCase):
    """add_chunk_vectors flushes store.add_documents in batches over any iterable."""

    def _chunks(self, n):
        return [{"id": i, "text": f"chunk {i}", "chunk_index": i} for i in range(n)]

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_batches_by_size(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        add_chunk_vectors(self._chunks(5), document_id=1, data_room_id=2, version_id=3, batch_size=4)

        self.assertEqual(store.add_documents.call_count, 2)
        sizes = [len(call.args[0]) for call in store.add_documents.call_args_list]
        self.assertEqual(sizes, [4, 1])

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_exact_batch_size_single_call(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        add_chunk_vectors(self._chunks(4), document_id=1, data_room_id=2, version_id=3, batch_size=4)
        self.assertEqual(store.add_documents.call_count, 1)

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_empty_iterable_no_calls(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        add_chunk_vectors([], document_id=1, data_room_id=2, version_id=3, batch_size=4)
        store.add_documents.assert_not_called()

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_total_documents_and_metadata(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        add_chunk_vectors(self._chunks(7), document_id=11, data_room_id=22, version_id=33, batch_size=3)

        all_docs = [d for call in store.add_documents.call_args_list for d in call.args[0]]
        self.assertEqual(len(all_docs), 7)
        first = all_docs[0]
        self.assertEqual(first.page_content, "chunk 0")
        self.assertEqual(first.metadata["chunk_id"], 0)
        self.assertEqual(first.metadata["document_id"], 11)
        self.assertEqual(first.metadata["data_room_id"], 22)
        self.assertEqual(first.metadata["chunk_index"], 0)

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_consumes_generator_single_pass(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        gen = (c for c in self._chunks(5))
        add_chunk_vectors(gen, document_id=1, data_room_id=2, version_id=3, batch_size=2)
        self.assertEqual(store.add_documents.call_count, 3)  # 2 + 2 + 1

    @override_settings(EMBEDDING_BATCH_SIZE=2)
    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_uses_embedding_batch_size_setting(self, _conn, mock_get_store):
        from documents.services.vector_store import add_chunk_vectors

        store = MagicMock()
        mock_get_store.return_value = store
        add_chunk_vectors(self._chunks(5), document_id=1, data_room_id=2, version_id=3)  # no batch_size -> setting (2)
        self.assertEqual(store.add_documents.call_count, 3)


class ChunkAccessTests(TestCase):
    """Keyset streaming and head/tail reconstruction."""

    def setUp(self):
        self.user = User.objects.create_user(email="ca@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="CAProject", slug="ca-project", created_by=self.user)

    def _doc(self, token_count=0):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="ca.txt",
            status=DataRoomDocument.Status.READY,
            token_count=token_count,
        )
        make_version(doc)
        return doc

    def test_iter_version_chunks_pages_in_order(self):
        from documents.services.chunk_access import iter_version_chunks

        doc = self._doc()
        for i in range(7):
            DataRoomDocumentChunk.objects.create(version=doc.current_version, chunk_index=i, text=f"t{i}", token_count=5)

        rows = list(iter_version_chunks(doc.current_version_id, fields=("id", "text", "chunk_index"), page_size=3))
        self.assertEqual([r["chunk_index"] for r in rows], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(len({r["id"] for r in rows}), 7)  # no duplicates / skips

    def test_iter_version_chunks_includes_index_zero_and_forces_key(self):
        from documents.services.chunk_access import iter_version_chunks

        doc = self._doc()
        DataRoomDocumentChunk.objects.create(version=doc.current_version, chunk_index=0, text="zero", token_count=5)
        rows = list(iter_version_chunks(doc.current_version_id, fields=("text",), page_size=10))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chunk_index"], 0)  # chunk_index always selected even if omitted

    def test_build_head_tail_small_doc_returns_full_text(self):
        from documents.services.chunk_access import build_head_tail_text

        doc = self._doc(token_count=30)  # <= _MAX_INPUT_TOKENS
        for i in range(3):
            DataRoomDocumentChunk.objects.create(version=doc.current_version, chunk_index=i, text=f"AlphaContent{i}", token_count=10)

        text = build_head_tail_text(doc.current_version_id)
        self.assertIn("AlphaContent0", text)
        self.assertIn("AlphaContent2", text)
        self.assertNotIn("omitted", text)

    def test_build_head_tail_large_doc_omits_middle(self):
        from documents.services.chunk_access import build_head_tail_text

        doc = self._doc(token_count=30000)  # > _MAX_INPUT_TOKENS (10000)
        for i in range(30):
            DataRoomDocumentChunk.objects.create(version=doc.current_version, chunk_index=i, text=f"Para{i}Body", token_count=1000)

        text = build_head_tail_text(doc.current_version_id)
        self.assertIn("Para0Body", text)        # head (first ~5k tokens)
        self.assertIn("Para29Body", text)       # tail (last ~2k tokens)
        self.assertNotIn("Para15Body", text)    # middle omitted
        self.assertIn("omitted", text)


class ScanPIICategoriesForDocumentTests(TestCase):
    """Windowed full-document PII scan."""

    def setUp(self):
        self.user = User.objects.create_user(email="piidoc@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="PIIDocProject", slug="piidoc-project", created_by=self.user)

    def _doc_with_chunks(self, n, token_each=10):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="piidoc.txt",
            status=DataRoomDocument.Status.READY,
            token_count=token_each * n,
        )
        make_version(
            doc,
            chunks=[{"text": f"window text {i}", "token_count": token_each} for i in range(n)],
        )
        return doc

    @override_settings(PII_SCAN_WINDOW_TOKENS=10)
    def test_multiple_windows_union_categories(self):
        from documents.services.pii_scan import scan_pii_categories_for_version

        doc = self._doc_with_chunks(2, token_each=10)  # each chunk fills its own window
        with patch("documents.services.pii_scan.scan_pii_categories",
                   side_effect=[{"pii_ordinary_identity": True}, {"pii_special_category": True}]) as mock_scan:
            result = scan_pii_categories_for_version(doc.current_version_id)

        self.assertEqual(mock_scan.call_count, 2)
        self.assertEqual(result, {"pii_ordinary_identity": True, "pii_special_category": True})

    def test_single_window_one_call(self):
        from documents.services.pii_scan import scan_pii_categories_for_version

        doc = self._doc_with_chunks(2, token_each=10)  # default budget (6000) -> one window
        with patch("documents.services.pii_scan.scan_pii_categories",
                   return_value={"pii_ordinary_identity": True}) as mock_scan:
            result = scan_pii_categories_for_version(doc.current_version_id)

        self.assertEqual(mock_scan.call_count, 1)
        self.assertEqual(result, {"pii_ordinary_identity": True})

    @override_settings(PII_SCAN_WINDOW_TOKENS=10)
    def test_early_exit_when_all_categories_found(self):
        from documents.services.pii_scan import PII_CATEGORIES, scan_pii_categories_for_version

        doc = self._doc_with_chunks(5, token_each=10)  # 5 windows available
        all_true = {cat: True for cat in PII_CATEGORIES}
        with patch("documents.services.pii_scan.scan_pii_categories", return_value=all_true) as mock_scan:
            result = scan_pii_categories_for_version(doc.current_version_id)

        self.assertEqual(mock_scan.call_count, 1)  # first window found everything -> stop
        self.assertEqual(len(result), len(PII_CATEGORIES))

    @override_settings(PII_SCAN_WINDOW_TOKENS=10)
    def test_window_failure_propagates(self):
        """A failed window must raise — a silently skipped window would let a
        document go READY without ever being fully scanned (the caller retries
        and marks the document SCAN_FAILED when retries are exhausted)."""
        from documents.services.pii_scan import scan_pii_categories_for_version

        doc = self._doc_with_chunks(2, token_each=10)
        with patch("documents.services.pii_scan.scan_pii_categories",
                   side_effect=[RuntimeError("boom"), {"pii_ordinary_identity": True}]):
            with self.assertRaises(RuntimeError):
                scan_pii_categories_for_version(doc.current_version_id)


class PIIGateTests(TestCase):
    """pii_gate_applies: documents are held from retrieval (SCANNING) only when a
    scan model is resolved and the org has both scan and quarantine enabled."""

    def setUp(self):
        self.user = User.objects.create_user(email="gate@example.com", password="testpass")

    def _org(self, **prefs):
        from accounts.models import Membership, Organization

        org = Organization.objects.create(name="GateOrg", slug="gate-org", preferences=prefs)
        Membership.objects.create(user=self.user, org=org, role="admin")
        return org

    @override_settings(**_MODELS)
    def test_gate_applies_with_defaults(self):
        from documents.services.pii_scan import pii_gate_applies

        org = self._org()
        self.assertTrue(pii_gate_applies(org.id))

    @override_settings(**_MODELS)
    def test_gate_off_when_scan_disabled(self):
        from documents.services.pii_scan import pii_gate_applies

        org = self._org(pii_scan_enabled=False)
        self.assertFalse(pii_gate_applies(org.id))

    @override_settings(**_MODELS)
    def test_gate_off_when_quarantine_disabled(self):
        """Quarantine off means the scan is informational — nothing to gate on."""
        from documents.services.pii_scan import pii_gate_applies

        org = self._org(pii_quarantine_enabled=False)
        self.assertFalse(pii_gate_applies(org.id))

    def test_gate_off_when_no_model_resolved(self):
        from documents.services.pii_scan import pii_gate_applies

        org = self._org()
        with patch("core.preferences.resolve_org_feature_model", return_value=""):
            self.assertFalse(pii_gate_applies(org.id))

    @override_settings(**_MODELS)
    def test_org_id_for_document(self):
        from documents.services.pii_scan import org_id_for_document

        org = self._org()
        room = DataRoom.objects.create(name="GateRoom", slug="gate-room", created_by=self.user)
        doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="g.txt",
        )
        self.assertEqual(org_id_for_document(doc), org.id)

    @override_settings(**_MODELS)
    def test_org_id_none_without_membership(self):
        from documents.services.pii_scan import org_id_for_document

        room = DataRoom.objects.create(name="SoloRoom", slug="solo-room", created_by=self.user)
        doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="s.txt",
        )
        self.assertIsNone(org_id_for_document(doc))


class ProcessDocumentDispatchTests(TestCase):
    """process_document holds the document in SCANNING and dispatches the guardrail scan."""

    def setUp(self):
        self.user = User.objects.create_user(email="disp@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="DispProject", slug="disp-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_process_document_holds_scanning_and_dispatches_scan(self):
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [{"text": "chunk", "token_count": 5, "chunk_index": 0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="disp.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("disp.txt", ContentFile(b"hi"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="hi")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("documents.tasks.finalize_document_metadata.delay") as mock_finalize, \
                     patch("guardrails.tasks.scan_document_version.delay") as mock_scan:
                    process_document(doc.id)

                # The guardrail scan is dispatched (it hands off to finalize); the doc
                # is held in SCANNING, not released to READY here.
                doc.refresh_from_db()
                mock_scan.assert_called_once_with(doc.current_version_id)
                mock_finalize.assert_not_called()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)


class ProcessDocumentGateTests(TestCase):
    """process_document holds every document in SCANNING until the guardrail scan releases it."""

    def setUp(self):
        self.user = User.objects.create_user(email="procgate@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="ProcGate", slug="proc-gate", created_by=self.user)

    def _run_process(self, mock_scan_delay):
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [{"text": "chunk", "token_count": 5, "chunk_index": 0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="gate.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("gate.txt", ContentFile(b"hi"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="hi")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay", mock_scan_delay):
                    process_document(doc.id)
                doc.refresh_from_db()
                return doc

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_doc_held_in_scanning(self):
        doc = self._run_process(MagicMock())
        self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
        self.assertIsNone(doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_scan_dispatch_failure_marks_scan_failed(self):
        from documents.services.pii_scan import SCAN_FAILED_MESSAGE

        doc = self._run_process(MagicMock(side_effect=RuntimeError("broker down")))
        self.assertEqual(doc.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(doc.processing_error, SCAN_FAILED_MESSAGE)


class ScanGateTransitionTests(TestCase):
    """finalize_document_metadata releases SCANNING documents to READY or SCAN_FAILED."""

    _DESC = {"description": "A description", "tags": {}, "document_date": None}

    def setUp(self):
        self.user = User.objects.create_user(email="scangate@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="ScanGate", slug="scan-gate", created_by=self.user)

    def _scanning_doc(self, description=""):
        from documents.tests._helpers import make_version
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="scan.txt",
            status=DataRoomDocument.Status.SCANNING,
            description=description,
            token_count=10,
        )
        make_version(
            doc, status=DataRoomDocument.Status.SCANNING, make_active=False,
            chunks=[{"text": "text", "token_count": 10}],
        )
        return doc

    @override_settings(**_MODELS)
    def test_clean_scan_releases_to_ready(self):
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)
        self.assertFalse(doc.is_quarantined)

    @override_settings(**_MODELS)
    def test_quarantined_doc_still_released_to_ready(self):
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   return_value={"pii_special_category": True}):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)
        self.assertTrue(doc.is_quarantined)
        self.assertIn("Article 9", doc.quarantine_reason)

    @override_settings(**_MODELS)
    def test_config_error_marks_scan_failed_without_retry(self):
        from documents.services.pii_scan import SCAN_FAILED_MESSAGE
        from documents.tasks import finalize_document_metadata
        from llm.service.errors import LLMConfigurationError

        doc = self._scanning_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   side_effect=LLMConfigurationError("no api key")), \
             patch.object(finalize_document_metadata, "retry") as mock_retry:
            finalize_document_metadata(doc.current_version_id)

        mock_retry.assert_not_called()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(doc.processing_error, SCAN_FAILED_MESSAGE)

    @override_settings(**_MODELS)
    def test_transient_failure_retries_and_stays_scanning(self):
        from celery.exceptions import Retry
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   side_effect=RuntimeError("LLM hiccup")), \
             patch.object(finalize_document_metadata, "retry", side_effect=Retry("will retry")) as mock_retry:
            with self.assertRaises(Retry):
                finalize_document_metadata(doc.current_version_id)

        mock_retry.assert_called_once()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)

    @override_settings(**_MODELS)
    def test_retries_exhausted_marks_scan_failed(self):
        from celery.exceptions import MaxRetriesExceededError
        from documents.services.pii_scan import SCAN_FAILED_MESSAGE
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   side_effect=RuntimeError("LLM down")), \
             patch.object(finalize_document_metadata, "retry", side_effect=MaxRetriesExceededError()):
            finalize_document_metadata(doc.current_version_id)  # must not raise

        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.SCAN_FAILED)
        self.assertEqual(doc.processing_error, SCAN_FAILED_MESSAGE)

    @override_settings(**_MODELS)
    def test_ungated_failure_leaves_ready_doc_alone(self):
        """A READY (ungated) doc with a failing scan keeps today's best-effort behavior."""
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        # Ungated/already-released: the document AND its version are READY (not held).
        doc.status = DataRoomDocument.Status.READY
        doc.save(update_fields=["status"])
        v = doc.current_version
        v.status = DataRoomDocument.Status.READY
        v.save(update_fields=["status"])
        with patch("documents.services.description.generate_description_and_tags_from_text",
                   return_value=self._DESC), \
             patch("documents.services.pii_scan.scan_pii_categories_for_version",
                   side_effect=RuntimeError("boom")), \
             patch.object(finalize_document_metadata, "retry") as mock_retry:
            finalize_document_metadata(doc.current_version_id)  # must not raise

        mock_retry.assert_not_called()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)

    def test_gate_disabled_after_processing_still_releases(self):
        """If the org disabled the scan between processing and finalize, release the doc."""
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc()
        with patch("core.preferences.resolve_org_feature_model", return_value=""):
            finalize_document_metadata(doc.current_version_id)

        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)

    @override_settings(**_MODELS)
    def test_description_skipped_when_already_set(self):
        """Scan retries must not regenerate an existing description (burns LLM calls)."""
        from documents.tasks import finalize_document_metadata

        doc = self._scanning_doc(description="Already described")
        with patch("documents.services.description.generate_description_and_tags_from_text") as mock_desc, \
             patch("documents.services.pii_scan.scan_pii_categories_for_version", return_value={}):
            finalize_document_metadata(doc.current_version_id)

        mock_desc.assert_not_called()
        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)
        self.assertEqual(doc.description, "Already described")
