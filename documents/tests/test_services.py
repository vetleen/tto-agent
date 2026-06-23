import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk
from documents.services.chunking import (
    EmailAttachment,
    _count_tokens,
    _extract_attachment_content,
    _format_email_as_markdown,
    _format_size,
    _load_eml_as_markdown,
    _load_msg_as_markdown,
    _strip_nul_bytes,
    clean_extracted_text,
    load_documents,
    structure_aware_chunk,
)
from documents.services.retrieval import (
    _rrf_score,
    get_chunks_by_document,
    get_chunks_by_data_room,
    get_merged_context_windows,
    hybrid_search_chunks,
    rerank_chunks,
    similarity_search_chunks,
)

User = get_user_model()


def _pdf_with_image(width=120, height=80):
    """Return PDF bytes containing a single embedded raster image (Pillow page)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, format="PDF")
    return buf.getvalue()


def _doc_chunk(doc, **kw):
    """Create a chunk on doc's working version (lazily making one searchable per status)."""
    from documents.models import DataRoomDocumentChunk
    from documents.tests._helpers import make_version
    v = (
        doc.current_version if doc.current_version_id
        else make_version(doc, status=doc.status, is_quarantined=doc.is_quarantined)
    )
    return DataRoomDocumentChunk.objects.create(version=v, **kw)

try:
    import langchain_core  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import langchain_openai  # noqa: F401
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False


class ChunkingTests(TestCase):
    def test_count_tokens(self):
        n = _count_tokens("Hello world")
        self.assertGreaterEqual(n, 1)
        self.assertLessEqual(n, 10)

    def test_count_tokens_falls_back_when_encoding_is_unavailable(self):
        n = _count_tokens("hello world from fallback", encoding_name="definitely_missing_encoding")
        self.assertEqual(n, 7)

    def test_count_tokens_fallback_uses_char_heuristic_for_unspaced_text(self):
        text = "x" * 401
        n = _count_tokens(text, encoding_name="definitely_missing_encoding")
        self.assertEqual(n, 101)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_load_documents_txt(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello from file")
            f.flush()
            path = Path(f.name)
        try:
            docs = load_documents(path, "txt")
            self.assertEqual(len(docs), 1)
            self.assertIn("Hello from file", docs[0].page_content)
        finally:
            path.unlink(missing_ok=True)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_strip_nul_bytes_removes_null_characters(self):
        from langchain_core.documents import Document
        docs = [
            Document(page_content="hello\x00world\x00", metadata={"page": 1}),
            Document(page_content="clean text", metadata={"page": 2}),
        ]
        result = _strip_nul_bytes(docs)
        self.assertEqual(result[0].page_content, "helloworld")
        self.assertEqual(result[1].page_content, "clean text")

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_load_documents_unsupported_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"x")
            path = Path(f.name)
        try:
            with self.assertRaises(ValueError):
                load_documents(path, "xyz")
        finally:
            path.unlink(missing_ok=True)


class SemanticChunkTests(TestCase):
    """Tests for semantic_chunk() function."""

    @unittest.skipIf(not LANGCHAIN_OPENAI_AVAILABLE, "langchain-openai not installed")
    @patch("langchain_experimental.text_splitter.SemanticChunker")
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_semantic_chunk_output_format(self, mock_embeddings_cls, mock_chunker_cls):
        from langchain_core.documents import Document as LCDoc
        from documents.services.chunking import semantic_chunk

        mock_chunker = MagicMock()
        mock_chunker.create_documents.return_value = [
            LCDoc(page_content="First semantic chunk."),
            LCDoc(page_content="Second semantic chunk."),
        ]
        mock_chunker_cls.return_value = mock_chunker

        result = semantic_chunk("First semantic chunk. Second semantic chunk.")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "First semantic chunk.")
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertIn("token_count", result[0])
        self.assertGreater(result[0]["token_count"], 0)
        self.assertEqual(result[1]["chunk_index"], 1)

    def test_semantic_chunk_empty_text(self):
        from documents.services.chunking import semantic_chunk
        result = semantic_chunk("")
        self.assertEqual(result, [])

    def test_semantic_chunk_whitespace_only(self):
        from documents.services.chunking import semantic_chunk
        result = semantic_chunk("   \n\n  ")
        self.assertEqual(result, [])

    @unittest.skipIf(not LANGCHAIN_OPENAI_AVAILABLE, "langchain-openai not installed")
    @patch("langchain_experimental.text_splitter.SemanticChunker")
    @patch("langchain_openai.OpenAIEmbeddings")
    def test_semantic_chunk_skips_empty_chunks(self, mock_embeddings_cls, mock_chunker_cls):
        from langchain_core.documents import Document as LCDoc
        from documents.services.chunking import semantic_chunk

        mock_chunker = MagicMock()
        mock_chunker.create_documents.return_value = [
            LCDoc(page_content="Real content."),
            LCDoc(page_content="   "),  # whitespace-only
            LCDoc(page_content="More content."),
        ]
        mock_chunker_cls.return_value = mock_chunker

        result = semantic_chunk("Real content. More content.")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Real content.")
        self.assertEqual(result[1]["text"], "More content.")


class VectorStoreTests(TestCase):
    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_delete_vectors_for_document_scopes_to_collection(self, _mock_conn, mock_get_store):
        """The delete runs on the store's own SQLAlchemy session (PGVECTOR_CONNECTION),
        scoped to this app's collection and the given document_id."""
        from documents.services.vector_store import COLLECTION_NAME, delete_vectors_for_document

        mock_session = MagicMock()
        store = Mock()
        store.session_maker.return_value.__enter__ = Mock(return_value=mock_session)
        store.session_maker.return_value.__exit__ = Mock(return_value=False)
        mock_get_store.return_value = store

        delete_vectors_for_document(document_id=42)

        stmt, params = mock_session.execute.call_args.args
        self.assertIn("col.name = :collection", str(stmt))
        self.assertIn("emb.cmetadata->>'document_id' = :document_id", str(stmt))
        self.assertEqual(params, {"collection": COLLECTION_NAME, "document_id": "42"})
        mock_session.commit.assert_called_once()

    @patch("documents.services.vector_store._get_vector_store")
    @patch("documents.services.vector_store._get_connection_string", return_value=None)
    def test_delete_vectors_noop_without_connection(self, _mock_conn, mock_get_store):
        """No PGVECTOR_CONNECTION (e.g. SQLite dev/tests) -> silent no-op."""
        from documents.services.vector_store import delete_vectors_for_document

        delete_vectors_for_document(document_id=42)

        mock_get_store.assert_not_called()

    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    @patch("documents.services.vector_store._get_vector_store")
    def test_similarity_search_bounds_k(self, mock_get_store, _mock_conn):
        from documents.services.vector_store import similarity_search

        store = Mock()
        store.similarity_search.return_value = ["result"]
        mock_get_store.return_value = store

        similarity_search(data_room_ids=[1], query="hello", k=500, document_id=9)

        self.assertEqual(store.similarity_search.call_args.kwargs["k"], 50)
        self.assertEqual(store.similarity_search.call_args.kwargs["filter"], {"data_room_id": {"$in": [1]}, "is_searchable": True, "document_id": 9})


class RRFScoreTests(TestCase):
    """Unit tests for the Reciprocal Rank Fusion scoring function."""

    def test_rrf_score_rank_zero(self):
        # rank 0 with default rrf_k=60: 1.0 / (60 + 0 + 1) = 1/61
        score = _rrf_score(0)
        self.assertAlmostEqual(score, 1.0 / 61)

    def test_rrf_score_with_weight(self):
        score = _rrf_score(0, weight=2.0)
        self.assertAlmostEqual(score, 2.0 / 61)

    def test_rrf_score_decreases_with_rank(self):
        s0 = _rrf_score(0)
        s1 = _rrf_score(1)
        s5 = _rrf_score(5)
        self.assertGreater(s0, s1)
        self.assertGreater(s1, s5)


class HybridSearchTests(TestCase):
    """Tests for hybrid_search_chunks RRF fusion logic."""

    def setUp(self):
        # Semantic hits are now validated against the DB (quarantine/status
        # filters), so every semantic chunk_id used in a test must exist as a
        # real READY-document chunk.
        self.user = User.objects.create_user(email="hybrid@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="HybridProject", slug="hybrid-project", created_by=self.user)
        self.document = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="hybrid.txt", status=DataRoomDocument.Status.READY,
        )
        from documents.tests._helpers import make_version
        self.version = make_version(self.document)  # READY + searchable + active
        self._next_chunk_index = 0

    def _make_chunk(self, chunk_id, text="chunk text"):
        self._next_chunk_index += 1
        return DataRoomDocumentChunk.objects.create(
            id=chunk_id, version=self.version,
            chunk_index=self._next_chunk_index, text=text, token_count=5,
        )

    def _make_semantic_doc(self, chunk_id, text, document_id=1, chunk_index=0):
        """Create a mock LangChain Document as returned by pgvector."""
        doc = MagicMock()
        doc.page_content = text
        doc.metadata = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "data_room_id": 1,
            "chunk_index": chunk_index,
        }
        return doc

    def _make_fts_hit(self, chunk_id, text, document_id=1, chunk_index=0, rank=0.5):
        return {
            "id": chunk_id,
            "chunk_index": chunk_index,
            "text": text,
            "heading": None,
            "document_id": document_id,
            "rank": rank,
        }

    @patch("documents.services.retrieval.fulltext_search_chunks", return_value=[])
    @patch("documents.services.retrieval.vs.similarity_search", return_value=[])
    def test_hybrid_returns_empty_when_no_results(self, _mock_sem, _mock_fts):
        results = hybrid_search_chunks(data_room_ids=[1], query="nothing", k=5)
        self.assertEqual(results, [])

    @patch("documents.services.retrieval.fulltext_search_chunks", return_value=[])
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_semantic_only_when_fts_empty(self, mock_sem, _mock_fts):
        self._make_chunk(10)
        mock_sem.return_value = [
            self._make_semantic_doc(10, "semantic result"),
        ]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 10)
        self.assertEqual(results[0]["text"], "semantic result")
        self.assertGreater(results[0]["rrf_score"], 0)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search", return_value=[])
    def test_hybrid_fts_only_when_semantic_empty(self, _mock_sem, mock_fts):
        mock_fts.return_value = [
            self._make_fts_hit(20, "fulltext result"),
        ]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 20)
        self.assertEqual(results[0]["text"], "fulltext result")
        self.assertGreater(results[0]["rrf_score"], 0)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_fuses_and_boosts_overlapping_results(self, mock_sem, mock_fts):
        """A chunk appearing in both result sets gets a higher RRF score."""
        shared_id = 100
        only_semantic_id = 200
        only_fts_id = 300
        self._make_chunk(shared_id)
        self._make_chunk(only_semantic_id)

        mock_sem.return_value = [
            self._make_semantic_doc(shared_id, "shared chunk"),
            self._make_semantic_doc(only_semantic_id, "semantic only"),
        ]
        mock_fts.return_value = [
            self._make_fts_hit(shared_id, "shared chunk"),
            self._make_fts_hit(only_fts_id, "fts only"),
        ]

        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=10)

        ids = [r["id"] for r in results]
        self.assertIn(shared_id, ids)
        self.assertIn(only_semantic_id, ids)
        self.assertIn(only_fts_id, ids)

        # Shared chunk should be ranked first (highest RRF from both lists)
        self.assertEqual(results[0]["id"], shared_id)

        # Its score should be higher than either single-source result
        shared_score = results[0]["rrf_score"]
        other_scores = [r["rrf_score"] for r in results[1:]]
        for s in other_scores:
            self.assertGreater(shared_score, s)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_respects_k_limit(self, mock_sem, mock_fts):
        for i in range(1, 11):
            self._make_chunk(i)
        mock_sem.return_value = [
            self._make_semantic_doc(i, f"sem {i}") for i in range(1, 11)
        ]
        mock_fts.return_value = [
            self._make_fts_hit(i + 100, f"fts {i}") for i in range(10)
        ]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=3)
        self.assertEqual(len(results), 3)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search", side_effect=Exception("pgvector down"))
    def test_hybrid_degrades_gracefully_when_semantic_fails(self, _mock_sem, mock_fts):
        mock_fts.return_value = [
            self._make_fts_hit(50, "fallback result"),
        ]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 50)

    @patch("documents.services.retrieval.fulltext_search_chunks", side_effect=Exception("FTS down"))
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_degrades_gracefully_when_fts_fails(self, mock_sem, _mock_fts):
        self._make_chunk(60)
        mock_sem.return_value = [
            self._make_semantic_doc(60, "fallback semantic"),
        ]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 60)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_prefers_fts_heading_over_semantic(self, mock_sem, mock_fts):
        """When same chunk appears in both, heading from FTS result should be kept."""
        self._make_chunk(1)
        mock_sem.return_value = [
            self._make_semantic_doc(1, "some text"),
        ]
        mock_fts.return_value = [{
            "id": 1,
            "chunk_index": 0,
            "text": "some text",
            "heading": "Important Section",
            "document_id": 1,
            "rank": 0.8,
        }]
        results = hybrid_search_chunks(data_room_ids=[1], query="test", k=5)
        self.assertEqual(results[0]["heading"], "Important Section")


class ProcessDocumentServiceTests(TestCase):
    """Unit tests for documents.services.process_document.process_document()."""

    def setUp(self):
        self.user = User.objects.create_user(email="svc@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="SvcProject", slug="svc-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_process_document_happy_path(self):
        """UPLOADED doc with a real file is held in SCANNING (guardrail scan) and gets chunks."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {"text": "First chunk", "heading": None, "token_count": 10,
             "chunk_index": 0,
             "source_page_start": None, "source_page_end": None,
             "source_offset_start": 0, "source_offset_end": 11},
            {"text": "Second chunk", "heading": "Section", "token_count": 12,
             "chunk_index": 1,
             "source_page_start": None, "source_page_end": None,
             "source_offset_start": 12, "source_offset_end": 24},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="test.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("test.txt", ContentFile(b"hello world"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="hello world")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                self.assertEqual(doc.current_version.chunks.count(), 2)
                self.assertEqual(doc.token_count, 22)
                self.assertEqual(doc.chunking_strategy, "semantic")
                self.assertIsNotNone(doc.processed_at)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_sets_failed_when_file_missing(self):
        """Doc with no attached file transitions to FAILED with a processing_error."""
        from documents.services.process_document import process_document

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="missing.txt",
            status=DataRoomDocument.Status.UPLOADED,
        )
        process_document(doc.id)

        doc.refresh_from_db()
        self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
        self.assertIsNotNone(doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_sets_failed_on_chunking_error(self):
        """Chunking exception causes status=FAILED and error stored in processing_error."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="bad.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("bad.txt", ContentFile(b"content"), save=True)

                with patch(
                    "documents.services.process_document.load_documents",
                    side_effect=RuntimeError("parse error"),
                ):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("parse error", doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_fails_when_no_text_extracted(self):
        """A PDF that yields no extractable text (e.g. scanned) should be marked FAILED."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="scanned.pdf",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("scanned.pdf", ContentFile(b"fake"), save=True)

                with patch(
                    "documents.services.process_document.load_documents",
                    return_value=[Mock(page_content="")],
                ):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("No text could be extracted", doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_fails_when_zero_chunks(self):
        """Document with extractable text but 0 chunks should be marked FAILED."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="weird.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("weird.txt", ContentFile(b"has text"), save=True)

                with patch(
                    "documents.services.process_document.load_documents",
                    return_value=[Mock(page_content="some real content")],
                ), patch(
                    "documents.services.process_document.structure_aware_chunk",
                    return_value=[],
                ):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("0 chunks", doc.processing_error)

    def test_process_document_skips_nonexistent_document(self):
        """Calling with a non-existent ID logs a warning and returns without raising."""
        from documents.services.process_document import process_document

        with self.assertLogs("documents.services.process_document", level="WARNING") as cm:
            process_document(99999)

        self.assertTrue(any("not found" in line for line in cm.output))

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_image_uses_description_as_content(self):
        """An uploaded image is described by a vision model; the description
        becomes the document's searchable text and parser_type='image'."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {"text": "A bar chart of quarterly revenue.", "heading": None,
             "token_count": 8, "chunk_index": 0,
             "source_page_start": None, "source_page_end": None,
             "source_offset_start": 0, "source_offset_end": 33},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="chart.png",
                    mime_type="image/png",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("chart.png", ContentFile(b"\x89PNG\r\n\x1a\nfakebytes"), save=True)

                with patch("chat.services.describe_image", return_value="A bar chart of quarterly revenue.") as mock_describe, \
                     patch("core.preferences.resolve_org_feature_model", return_value="anthropic/claude-opus-4-8"), \
                     patch("documents.services.process_document.structure_aware_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                self.assertEqual(doc.current_version.parser_type, "image")
                self.assertEqual(doc.current_version.chunks.count(), 1)
                self.assertTrue(mock_describe.called)
                # The org-resolved describer model is threaded through.
                _, kwargs = mock_describe.call_args
                self.assertEqual(kwargs.get("model"), "anthropic/claude-opus-4-8")

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_image_fails_without_vision_model(self):
        """With no vision-capable describer resolved, the image doc fails with a
        clear error instead of producing empty content."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="chart.png",
                    mime_type="image/png",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("chart.png", ContentFile(b"\x89PNGfake"), save=True)

                with patch("core.preferences.resolve_org_feature_model", return_value=""):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("Image description is not enabled", doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_docx_embedded_images_become_assets(self):
        """Embedded docx images are stored as version-scoped ImageAssets with an
        inline [[image:uuid|...]] token left in the searchable content."""
        from django.core.files.base import ContentFile

        from chat.models import ImageAsset
        from chat.tests.test_attachments import _docx_with_image
        from documents.services.process_document import process_document

        docx_bytes = _docx_with_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="deck.docx",
                    mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("deck.docx", ContentFile(docx_bytes), save=True)

                with patch("chat.services.describe_image", return_value="A revenue chart"), \
                     patch("core.preferences.resolve_org_feature_model", return_value="anthropic/claude-opus-4-8"), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                version = doc.current_version
                assets = list(ImageAsset.objects.filter(version=version))
                self.assertEqual(len(assets), 1)
                self.assertEqual(assets[0].description, "A revenue chart")
                self.assertTrue(assets[0].sha256)
                token = f"[[image:{assets[0].id}|"
                self.assertTrue(
                    any(token in c.text for c in version.chunks.all()),
                    "asset token should appear in the extracted/chunked content",
                )

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_pdf_embedded_images_become_assets(self):
        """Embedded PDF images are stored as version-scoped ImageAssets with an
        inline [[image:uuid|...]] token left in the searchable content — parity
        with the docx path."""
        from django.core.files.base import ContentFile

        from chat.models import ImageAsset
        from documents.services.process_document import process_document

        pdf_bytes = _pdf_with_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="report.pdf",
                    mime_type="application/pdf",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("report.pdf", ContentFile(pdf_bytes), save=True)

                with patch("chat.services.describe_image", return_value="A revenue chart"), \
                     patch("core.preferences.resolve_org_feature_model", return_value="anthropic/claude-opus-4-8"), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                version = doc.current_version
                assets = list(ImageAsset.objects.filter(version=version))
                self.assertEqual(len(assets), 1)
                self.assertEqual(assets[0].description, "A revenue chart")
                self.assertTrue(assets[0].sha256)
                token = f"[[image:{assets[0].id}|"
                self.assertTrue(
                    any(token in c.text for c in version.chunks.all()),
                    "asset token should appear in the extracted/chunked content",
                )

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_image_only_pdf_no_longer_fails(self):
        """A scanned/image-only PDF (no extractable text) used to hard-fail with
        the empty-text error; now its embedded page image is described, so the
        description becomes the searchable content and the doc succeeds."""
        from django.core.files.base import ContentFile

        from documents.services.process_document import process_document

        pdf_bytes = _pdf_with_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="scan.pdf",
                    mime_type="application/pdf",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("scan.pdf", ContentFile(pdf_bytes), save=True)

                with patch("chat.services.describe_image", return_value="A scanned page of text"), \
                     patch("core.preferences.resolve_org_feature_model", return_value="anthropic/claude-opus-4-8"), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                self.assertNotEqual(doc.status, DataRoomDocument.Status.FAILED)


class ProcessDocumentSemanticTests(TestCase):
    """Tests specific to the semantic chunking pipeline."""

    def setUp(self):
        self.user = User.objects.create_user(email="sem@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="SemProject", slug="sem-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_flat_chunks_created(self):
        """Verify flat chunks are created with sequential indexes."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {"text": f"Chunk {i}", "token_count": 10, "chunk_index": i}
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="flat.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("flat.txt", ContentFile(b"test content"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="test content")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay"):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                self.assertEqual(doc.current_version.chunks.count(), 5)

                # All chunks should have sequential indexes
                indexes = list(doc.current_version.chunks.order_by("chunk_index").values_list("chunk_index", flat=True))
                self.assertEqual(indexes, [0, 1, 2, 3, 4])


class RetrievalServiceTests(TestCase):
    """Unit tests for documents.services.retrieval non-search functions."""

    def setUp(self):
        self.user = User.objects.create_user(email="ret@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="RetProject", slug="ret-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="ret.txt",
            status=DataRoomDocument.Status.READY,
        )

    def test_get_chunks_by_document_returns_ordered(self):
        """Chunks are returned in chunk_index order regardless of insertion order."""
        _doc_chunk(self.doc, chunk_index=1, text="Second", token_count=2)
        _doc_chunk(self.doc, chunk_index=0, text="First", token_count=1)

        result = get_chunks_by_document(self.doc.id)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertEqual(result[0]["text"], "First")
        self.assertEqual(result[1]["chunk_index"], 1)
        # Required fields are present
        for key in ("id", "text", "heading", "token_count", "source_page_start", "source_page_end"):
            self.assertIn(key, result[0])

    def test_get_chunks_by_data_room_excludes_failed_documents(self):
        """Chunks from FAILED documents are not returned."""
        _doc_chunk(self.doc, chunk_index=0, text="Ready chunk", token_count=5)

        failed_doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="fail.txt",
            status=DataRoomDocument.Status.FAILED,
        )
        _doc_chunk(failed_doc, chunk_index=0, text="Failed chunk", token_count=5)

        result = get_chunks_by_data_room(self.data_room.id)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Ready chunk")
        self.assertEqual(result[0]["document_id"], self.doc.id)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_similarity_search_chunks_returns_langchain_documents(self):
        """similarity_search_chunks wraps hybrid_search output as LangChain Documents with doc_index."""
        from langchain_core.documents import Document

        fake_results = [
            {"id": 42, "text": "hello", "document_id": self.doc.pk, "chunk_index": 0, "rrf_score": 0.5, "heading": None},
        ]
        with patch("documents.services.retrieval.hybrid_search_chunks", return_value=fake_results):
            results = similarity_search_chunks(data_room_ids=[self.data_room.pk], query="test", k=5)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertEqual(results[0].page_content, "hello")
        self.assertEqual(results[0].metadata["chunk_id"], 42)
        self.assertEqual(results[0].metadata["data_room_id"], self.data_room.pk)
        self.assertEqual(results[0].metadata["doc_index"], self.doc.doc_index)
        self.assertNotIn("document_id", results[0].metadata)


class QuarantineRetrievalTests(TestCase):
    """A document-level quarantine excludes all of a doc's content from retrieval."""

    def setUp(self):
        self.user = User.objects.create_user(email="quar@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="QuarProject", slug="quar-project", created_by=self.user)
        self.clean = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="clean.txt", status=DataRoomDocument.Status.READY,
        )
        self.clean_chunk = _doc_chunk(self.clean, chunk_index=0, text="Clean content", token_count=5,
        )
        self.quarantined = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="quar.txt", status=DataRoomDocument.Status.READY,
            is_quarantined=True, quarantine_reason="Contains GDPR Article 9 (special category) personal data.",
        )
        self.quar_chunk = _doc_chunk(self.quarantined, chunk_index=0, text="Sensitive content", token_count=5,
        )

    def test_get_chunks_by_document_excludes_quarantined_doc(self):
        self.assertEqual(get_chunks_by_document(self.quarantined.id), [])
        self.assertEqual(len(get_chunks_by_document(self.clean.id)), 1)

    def test_get_chunks_by_data_room_excludes_quarantined_doc(self):
        result = get_chunks_by_data_room(self.data_room.id)
        doc_ids = {r["document_id"] for r in result}
        self.assertEqual(doc_ids, {self.clean.id})

    def test_get_merged_context_windows_drops_quarantined_doc(self):
        windows = get_merged_context_windows([self.quar_chunk.id, self.clean_chunk.id])
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["document_id"], self.clean.id)

    @unittest.skipUnless(connection.vendor == "postgresql", "fulltext search requires PostgreSQL")
    def test_fulltext_search_excludes_quarantined_doc(self):
        from documents.services.retrieval import fulltext_search_chunks

        results = fulltext_search_chunks([self.data_room.id], "content", k=10)
        doc_ids = {r["document_id"] for r in results}
        self.assertNotIn(self.quarantined.id, doc_ids)

    @patch("documents.services.retrieval.fulltext_search_chunks", return_value=[])
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_semantic_path_drops_quarantined_chunk(self, mock_sem, _mock_fts):
        """The semantic side must enforce chunk-level quarantine like the FTS side does in SQL."""
        bad_chunk = _doc_chunk(self.clean, chunk_index=1, text="Injected content", token_count=5,
            is_quarantined=True, quarantine_reason="Adversarial content",
        )

        def _sem_doc(chunk):
            doc = MagicMock()
            doc.page_content = chunk.text
            doc.metadata = {"chunk_id": chunk.id, "document_id": chunk.version.document_id,
                            "data_room_id": self.data_room.id, "chunk_index": chunk.chunk_index}
            return doc

        mock_sem.return_value = [_sem_doc(bad_chunk), _sem_doc(self.clean_chunk)]
        results = hybrid_search_chunks(data_room_ids=[self.data_room.id], query="content", k=5)
        ids = {r["id"] for r in results}
        self.assertNotIn(bad_chunk.id, ids)
        self.assertIn(self.clean_chunk.id, ids)


class ScanningStatusRetrievalTests(TestCase):
    """Documents not yet READY (scanning / scan_failed / failed) never surface in retrieval."""

    def setUp(self):
        self.user = User.objects.create_user(email="scanret@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="ScanRet", slug="scan-ret", created_by=self.user)
        self.ready = self._doc("ready.txt", DataRoomDocument.Status.READY)
        self.ready_chunk = self._chunk(self.ready, "Ready content")
        self.scanning = self._doc("scanning.txt", DataRoomDocument.Status.SCANNING)
        self.scanning_chunk = self._chunk(self.scanning, "Unscanned content")
        self.scan_failed = self._doc("scanfailed.txt", DataRoomDocument.Status.SCAN_FAILED)
        self.scan_failed_chunk = self._chunk(self.scan_failed, "Unscannable content")

    def _doc(self, name, status):
        return DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename=name, status=status,
        )

    def _chunk(self, doc, text):
        return _doc_chunk(doc, chunk_index=0, text=text, token_count=5,
        )

    def test_get_chunks_by_document_requires_ready(self):
        self.assertEqual(get_chunks_by_document(self.scanning.id), [])
        self.assertEqual(get_chunks_by_document(self.scan_failed.id), [])
        self.assertEqual(len(get_chunks_by_document(self.ready.id)), 1)

    def test_get_chunks_by_data_room_requires_ready(self):
        result = get_chunks_by_data_room(self.data_room.id)
        doc_ids = {r["document_id"] for r in result}
        self.assertEqual(doc_ids, {self.ready.id})

    def test_get_merged_context_windows_requires_ready(self):
        windows = get_merged_context_windows(
            [self.scanning_chunk.id, self.scan_failed_chunk.id, self.ready_chunk.id]
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["document_id"], self.ready.id)

    @patch("documents.services.retrieval.fulltext_search_chunks", return_value=[])
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_semantic_path_requires_ready(self, mock_sem, _mock_fts):
        def _sem_doc(chunk):
            doc = MagicMock()
            doc.page_content = chunk.text
            doc.metadata = {"chunk_id": chunk.id, "document_id": chunk.version.document_id,
                            "data_room_id": self.data_room.id, "chunk_index": 0}
            return doc

        mock_sem.return_value = [
            _sem_doc(self.scanning_chunk), _sem_doc(self.scan_failed_chunk), _sem_doc(self.ready_chunk),
        ]
        results = hybrid_search_chunks(data_room_ids=[self.data_room.id], query="content", k=5)
        ids = {r["id"] for r in results}
        self.assertEqual(ids, {self.ready_chunk.id})


class CleanExtractedTextTests(TestCase):
    """Tests for clean_extracted_text() PDF artifact cleaning."""

    def test_hyphenated_line_breaks_rejoined(self):
        self.assertEqual(clean_extracted_text("work-\nforce"), "workforce")
        self.assertEqual(clean_extracted_text("tech-\n  nology"), "technology")

    def test_doi_lines_removed(self):
        text = "Some content.\nhttps://doi.org/10.1080/12345\nMore content."
        result = clean_extracted_text(text)
        self.assertNotIn("doi.org", result)
        self.assertIn("Some content.", result)
        self.assertIn("More content.", result)

    def test_journal_header_lines_removed(self):
        text = "JOURNAL OF CHANGE MANAGEMENT\nActual body text here."
        result = clean_extracted_text(text)
        self.assertNotIn("JOURNAL OF CHANGE MANAGEMENT", result)
        self.assertIn("Actual body text here.", result)

    def test_journal_header_too_short_preserved(self):
        """Lines under 10 chars of all-caps are not treated as headers."""
        text = "OK SURE\nBody text."
        result = clean_extracted_text(text)
        self.assertIn("OK SURE", result)

    def test_standalone_page_numbers_removed(self):
        text = "First paragraph.\n42\nSecond paragraph."
        result = clean_extracted_text(text)
        self.assertNotIn("\n42\n", result)
        self.assertIn("First paragraph.", result)
        self.assertIn("Second paragraph.", result)

    def test_page_n_format_removed(self):
        text = "Content.\nPage 3\nMore content."
        result = clean_extracted_text(text)
        self.assertNotIn("Page 3", result)

    def test_n_of_m_format_removed(self):
        text = "Content.\n5 of 20\nMore content."
        result = clean_extracted_text(text)
        self.assertNotIn("5 of 20", result)

    def test_excessive_blank_lines_collapsed(self):
        text = "First.\n\n\n\n\nSecond."
        result = clean_extracted_text(text)
        self.assertNotIn("\n\n\n", result)
        self.assertIn("First.\n\nSecond.", result)

    def test_excess_inline_whitespace_collapsed(self):
        text = "word     word"
        result = clean_extracted_text(text)
        self.assertEqual(result, "word word")

    def test_inline_urls_preserved(self):
        text = "Visit https://example.com/path?q=1 for more info."
        result = clean_extracted_text(text)
        self.assertIn("https://example.com/path?q=1", result)

    def test_inline_emails_preserved(self):
        text = "Contact user@example.com for details."
        result = clean_extracted_text(text)
        self.assertIn("user@example.com", result)

    def test_body_content_preserved(self):
        body = (
            "This study examines technology transfer from universities "
            "to industry. Results show a 42% increase in patent licensing."
        )
        result = clean_extracted_text(body)
        self.assertEqual(result, body)

    def test_combined_artifacts(self):
        """Multiple artifact types cleaned in one pass."""
        text = (
            "JOURNAL OF TECHNOLOGY TRANSFER\n"
            "https://doi.org/10.1007/s10961\n\n"
            "The work-\nforce adapted quickly.\n\n\n\n"
            "Page 1\n"
            "Some  extra   spaces here."
        )
        result = clean_extracted_text(text)
        self.assertNotIn("JOURNAL OF TECHNOLOGY TRANSFER", result)
        self.assertNotIn("doi.org", result)
        self.assertIn("workforce", result)
        self.assertNotIn("\n\n\n", result)
        self.assertNotIn("Page 1", result)
        self.assertNotIn("  extra", result)
        self.assertIn("Some extra spaces here.", result)

    def test_numbers_in_body_text_preserved(self):
        """Numbers that are part of sentences should not be removed."""
        text = "The sample included 42 participants across 3 sites."
        result = clean_extracted_text(text)
        self.assertEqual(result, text)

    def test_literal_newline_escapes_normalised(self):
        """Literal \\n sequences (from JSON exports / LLM copy-paste) are converted to real newlines."""
        text = "# Heading\\n\\nParagraph one.\\n\\n## Sub\\n\\nParagraph two."
        result = clean_extracted_text(text)
        self.assertIn("\n", result)
        self.assertNotIn("\\n", result)
        self.assertIn("# Heading", result)
        self.assertIn("Paragraph one.", result)

    def test_literal_newline_escapes_skipped_when_real_newlines_dominate(self):
        """When the text already has more real newlines than literal \\n, leave them alone."""
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nMentions \\n once"
        result = clean_extracted_text(text)
        self.assertIn("\\n", result)

    def test_empty_input(self):
        self.assertEqual(clean_extracted_text(""), "")
        self.assertEqual(clean_extracted_text("   "), "")


class StructureAwareChunkTests(TestCase):
    """Tests for structure_aware_chunk() function."""

    def test_splits_on_headings(self):
        """Headings create section boundaries and are extracted."""
        text = "Intro paragraph.\n\n# Section One\n\nContent of section one.\n\n# Section Two\n\nContent of section two."
        chunks = structure_aware_chunk(text)
        self.assertGreaterEqual(len(chunks), 2)
        # First chunk should be the intro (no heading)
        self.assertIn("Intro paragraph.", chunks[0]["text"])
        # Heading-based chunks should have heading set
        headed_chunks = [c for c in chunks if c.get("heading")]
        self.assertTrue(any(c["heading"] == "Section One" for c in headed_chunks))
        self.assertTrue(any(c["heading"] == "Section Two" for c in headed_chunks))

    @override_settings(TARGET_CHUNK_TOKENS=5000)
    def test_preserves_tables_as_atomic_units(self):
        """A Markdown table should not be split mid-row."""
        table = "| Col A | Col B |\n|-------|-------|\n| val 1 | val 2 |\n| val 3 | val 4 |"
        text = f"Intro text.\n\n{table}\n\nAfter table."
        chunks = structure_aware_chunk(text)
        # The table should appear intact in one of the chunks
        table_found = any(
            "| val 1 | val 2 |" in c["text"] and "| val 3 | val 4 |" in c["text"]
            for c in chunks
        )
        self.assertTrue(table_found, "Table was split across chunks")

    @override_settings(TARGET_CHUNK_TOKENS=5000)
    def test_preserves_lists_as_atomic_units(self):
        """A Markdown list should not be split mid-item."""
        list_text = "- Item one\n- Item two\n- Item three"
        text = f"Intro.\n\n{list_text}\n\nAfter list."
        chunks = structure_aware_chunk(text)
        list_found = any(
            "- Item one" in c["text"] and "- Item three" in c["text"]
            for c in chunks
        )
        self.assertTrue(list_found, "List was split across chunks")

    @unittest.skipIf(not LANGCHAIN_OPENAI_AVAILABLE, "langchain-openai not installed")
    @patch("documents.services.chunking.semantic_chunk")
    def test_large_section_delegates_to_semantic_chunk(self, mock_semantic):
        """Sections larger than TARGET_CHUNK_TOKENS delegate to semantic_chunk."""
        mock_semantic.return_value = [
            {"text": "sub chunk 1", "token_count": 50, "chunk_index": 0},
            {"text": "sub chunk 2", "token_count": 50, "chunk_index": 1},
        ]
        # Create text with a single very large block (no structural breaks within it)
        large_block = "word " * 2000  # ~2000 tokens, well above default 768
        text = f"# Big Section\n\n{large_block}"
        with self.settings(TARGET_CHUNK_TOKENS=100):
            chunks = structure_aware_chunk(text)
        mock_semantic.assert_called()
        self.assertTrue(len(chunks) >= 2)

    @override_settings(TARGET_CHUNK_TOKENS=5000)
    def test_small_sections_not_split_further(self):
        """Sections under the token threshold become single chunks."""
        text = "# Small Section\n\nJust a few words here."
        chunks = structure_aware_chunk(text)
        headed = [c for c in chunks if c.get("heading") == "Small Section"]
        self.assertEqual(len(headed), 1)
        self.assertIn("Just a few words here.", headed[0]["text"])

    def test_heading_propagated_to_chunks(self):
        """Heading field is set on all chunks within a section."""
        text = "# My Heading\n\nParagraph one.\n\nParagraph two."
        chunks = structure_aware_chunk(text)
        for c in chunks:
            if c.get("heading"):
                self.assertEqual(c["heading"], "My Heading")

    def test_empty_text_returns_empty(self):
        result = structure_aware_chunk("")
        self.assertEqual(result, [])
        result = structure_aware_chunk("   \n\n  ")
        self.assertEqual(result, [])

    def test_sequential_chunk_indices(self):
        """All output chunks have sequential 0-based chunk_index."""
        text = "# A\n\nContent A.\n\n# B\n\nContent B.\n\n# C\n\nContent C."
        chunks = structure_aware_chunk(text)
        indices = [c["chunk_index"] for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="structure_aware")
    def test_chunking_strategy_setting(self):
        """process_document respects CHUNKING_STRATEGY setting."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        user = User.objects.create_user(email="struct@example.com", password="testpass")
        data_room = DataRoom.objects.create(name="StructProject", slug="struct-project", created_by=user)

        sample_chunks = [
            {"text": "Chunk 0", "token_count": 10, "chunk_index": 0, "heading": "Intro"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=data_room,
                    uploaded_by=user,
                    original_filename="struct.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("struct.txt", ContentFile(b"hello world"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="hello world")]), \
                     patch("documents.services.process_document.structure_aware_chunk", return_value=sample_chunks) as mock_sa:
                    process_document(doc.id)

                mock_sa.assert_called_once()
                doc.refresh_from_db()
                self.assertEqual(doc.chunking_strategy, "structure_aware")

    def test_underline_style_headings(self):
        """Underline-style headings (=== and ---) are recognized."""
        text = "My Title\n=========\n\nContent under title.\n\nSubsection\n-----------\n\nSub content."
        chunks = structure_aware_chunk(text)
        headings = [c.get("heading") for c in chunks if c.get("heading")]
        self.assertIn("My Title", headings)
        self.assertIn("Subsection", headings)


class RerankTests(TestCase):
    """Tests for rerank_chunks() function."""

    def _make_results(self, n):
        return [
            {"id": i, "text": f"chunk text {i}", "chunk_index": i, "document_id": 1, "rrf_score": 0.5 - i * 0.01}
            for i in range(n)
        ]

    @patch("documents.services.retrieval._get_ranker")
    def test_rerank_reorders_results(self, mock_get_ranker):
        """Reranker reorders results according to relevance scores."""
        mock_ranker = MagicMock()
        # Return in reversed order
        mock_ranker.rerank.return_value = [
            {"id": 2, "score": 0.9},
            {"id": 0, "score": 0.7},
            {"id": 1, "score": 0.5},
        ]
        mock_get_ranker.return_value = mock_ranker

        results = self._make_results(3)
        reranked = rerank_chunks(results, query="test query", top_n=3)
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0]["id"], 2)
        self.assertEqual(reranked[1]["id"], 0)
        self.assertEqual(reranked[2]["id"], 1)

    @patch("documents.services.retrieval._get_ranker")
    def test_rerank_returns_top_n(self, mock_get_ranker):
        """Reranker returns only top_n results."""
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": i, "score": 1.0 - i * 0.1} for i in range(10)
        ]
        mock_get_ranker.return_value = mock_ranker

        results = self._make_results(10)
        reranked = rerank_chunks(results, query="test", top_n=3)
        self.assertEqual(len(reranked), 3)

    @override_settings(RERANK_ENABLED=False)
    def test_rerank_disabled(self):
        """When RERANK_ENABLED=False, results are returned as-is (truncated)."""
        results = self._make_results(5)
        reranked = rerank_chunks(results, query="test", top_n=3)
        self.assertEqual(len(reranked), 3)
        # Order preserved (no reranking)
        self.assertEqual(reranked[0]["id"], 0)
        self.assertEqual(reranked[1]["id"], 1)
        self.assertEqual(reranked[2]["id"], 2)

    @patch("documents.services.retrieval.flashrank", create=True)
    def test_rerank_graceful_when_flashrank_missing(self, mock_flashrank):
        """When flashrank is not importable, falls back gracefully."""
        results = self._make_results(5)
        with patch.dict("sys.modules", {"flashrank": None}):
            # Force re-import failure
            with patch("documents.services.retrieval._get_ranker", side_effect=ImportError("no flashrank")):
                # The function catches the exception internally
                reranked = rerank_chunks(results, query="test", top_n=3)
        self.assertEqual(len(reranked), 3)

    @patch("documents.services.retrieval._get_ranker")
    def test_rerank_graceful_on_exception(self, mock_get_ranker):
        """Reranker exception returns unranked results."""
        mock_ranker = MagicMock()
        mock_ranker.rerank.side_effect = RuntimeError("model error")
        mock_get_ranker.return_value = mock_ranker

        results = self._make_results(5)
        reranked = rerank_chunks(results, query="test", top_n=3)
        self.assertEqual(len(reranked), 3)
        # Falls back to first 3
        self.assertEqual(reranked[0]["id"], 0)

    @patch("documents.services.retrieval.rerank_chunks")
    @patch("documents.services.retrieval.hybrid_search_chunks")
    def test_similarity_search_calls_rerank(self, mock_hybrid, mock_rerank):
        """similarity_search_chunks integrates with rerank_chunks."""
        user = User.objects.create_user(email="rerank@example.com", password="testpass")
        data_room = DataRoom.objects.create(name="RerankProject", slug="rerank-project", created_by=user)
        doc = DataRoomDocument.objects.create(
            data_room=data_room, uploaded_by=user, original_filename="r.txt",
            status=DataRoomDocument.Status.READY,
        )

        mock_hybrid.return_value = [
            {"id": 42, "text": "hello", "document_id": doc.pk, "chunk_index": 0, "rrf_score": 0.5, "heading": None},
        ]
        mock_rerank.return_value = mock_hybrid.return_value

        results = similarity_search_chunks(data_room_ids=[data_room.pk], query="test", k=5)
        mock_rerank.assert_called_once()
        self.assertEqual(len(results), 1)

    def test_rerank_empty_results(self):
        """Empty input returns empty output."""
        self.assertEqual(rerank_chunks([], query="test", top_n=5), [])


class MergedContextWindowTests(TestCase):
    """Tests for get_merged_context_windows()."""

    def setUp(self):
        self.user = User.objects.create_user(email="merge@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="MergeProject", slug="merge-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="merge.txt",
            status=DataRoomDocument.Status.READY,
        )

    def _create_chunks(self, count, tokens_each=100, doc=None):
        """Create sequential chunks with given token counts."""
        doc = doc or self.doc
        chunks = []
        for i in range(count):
            text = f"Chunk {i} content. " * max(1, tokens_each // 5)
            c = _doc_chunk(doc,
                chunk_index=i,
                text=text,
                token_count=tokens_each,
            )
            chunks.append(c)
        return chunks

    def test_adjacent_chunks_merged(self):
        """Hits on adjacent chunks (2 and 3) produce a single merged window."""
        chunks = self._create_chunks(6, tokens_each=100)
        # Target 300 tokens: each hit expands to ~3 chunks, so 2 and 3 overlap
        windows = get_merged_context_windows(
            [chunks[2].id, chunks[3].id], target_tokens_per_window=300,
        )
        self.assertEqual(len(windows), 1)
        # Both hit chunk IDs should be in the window
        self.assertIn(chunks[2].id, windows[0]["chunk_ids"])
        self.assertIn(chunks[3].id, windows[0]["chunk_ids"])

    def test_non_adjacent_chunks_separate(self):
        """Hits on distant chunks (0 and 9) produce two separate windows."""
        chunks = self._create_chunks(10, tokens_each=100)
        windows = get_merged_context_windows(
            [chunks[0].id, chunks[9].id], target_tokens_per_window=200,
        )
        self.assertEqual(len(windows), 2)

    def test_cross_document_not_merged(self):
        """Chunks from different documents are never merged."""
        doc2 = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="merge2.txt",
            status=DataRoomDocument.Status.READY,
        )
        chunks1 = self._create_chunks(3, tokens_each=100, doc=self.doc)
        chunks2 = self._create_chunks(3, tokens_each=100, doc=doc2)
        windows = get_merged_context_windows(
            [chunks1[1].id, chunks2[1].id], target_tokens_per_window=500,
        )
        self.assertEqual(len(windows), 2)
        doc_ids = {w["document_id"] for w in windows}
        self.assertEqual(len(doc_ids), 2)

    def test_single_hit_unchanged(self):
        """Single hit returns one symmetrically expanded window."""
        chunks = self._create_chunks(5, tokens_each=100)
        windows = get_merged_context_windows(
            [chunks[2].id], target_tokens_per_window=300,
        )
        self.assertEqual(len(windows), 1)
        self.assertIn(chunks[2].id, windows[0]["chunk_ids"])
        self.assertIn(2, windows[0]["chunks_included"])

    def test_three_overlapping_hits_merged(self):
        """Three adjacent hits (3, 4, 5) merge into one window."""
        chunks = self._create_chunks(8, tokens_each=100)
        windows = get_merged_context_windows(
            [chunks[3].id, chunks[4].id, chunks[5].id], target_tokens_per_window=300,
        )
        self.assertEqual(len(windows), 1)
        for cid in [chunks[3].id, chunks[4].id, chunks[5].id]:
            self.assertIn(cid, windows[0]["chunk_ids"])

    def test_token_budget_respected(self):
        """Single-hit window token count respects the target budget."""
        chunks = self._create_chunks(10, tokens_each=100)
        # With target 200 and 100-token chunks, should get center + 1 neighbor = 200
        windows = get_merged_context_windows(
            [chunks[5].id], target_tokens_per_window=200,
        )
        self.assertEqual(len(windows), 1)
        # Should include at most 2 chunks (200 tokens budget, 100 each)
        self.assertLessEqual(windows[0]["context_token_count"], 200)

    def test_empty_input(self):
        """Empty chunk_ids returns empty list."""
        self.assertEqual(get_merged_context_windows([]), [])


class GenerateDescriptionAndTagsTests(TestCase):
    """Tests for generate_description_and_tags_from_text."""

    def test_empty_text_returns_empty(self):
        from documents.services.description import generate_description_and_tags_from_text
        result = generate_description_and_tags_from_text("   ")
        self.assertEqual(result["description"], "")
        self.assertEqual(result["tags"], {})

    @patch("llm.get_llm_service")
    def test_valid_structured_response(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_parsed = DocumentDescriptionOutput(
            description="A patent document.", document_type="Patent"
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some patent text", user_id=1)
        self.assertEqual(result["description"], "A patent document.")
        self.assertEqual(result["tags"], {"document_type": "Patent"})

    @patch("llm.get_llm_service")
    def test_is_image_adds_image_note_to_prompt(self, mock_get_service):
        """For image docs the cheap model is told its input is a vision
        description, so it summarises the image (not the description)."""
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_service = Mock()
        mock_service.run_structured.return_value = (
            DocumentDescriptionOutput(description="An aerial photo of a cove.", document_type="Image"),
            None,
        )
        mock_get_service.return_value = mock_service

        generate_description_and_tags_from_text("A detailed vision description.", user_id=1, is_image=True)
        request = mock_service.run_structured.call_args.args[0]
        self.assertIn("This document is an image", request.messages[0].content)

    @patch("llm.get_llm_service")
    def test_non_image_omits_image_note(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_service = Mock()
        mock_service.run_structured.return_value = (
            DocumentDescriptionOutput(description="A report.", document_type="Report"),
            None,
        )
        mock_get_service.return_value = mock_service

        generate_description_and_tags_from_text("Some text", user_id=1)
        request = mock_service.run_structured.call_args.args[0]
        self.assertNotIn("This document is an image", request.messages[0].content)

    @patch("llm.get_llm_service")
    def test_empty_document_type(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_parsed = DocumentDescriptionOutput(
            description="A document about something.", document_type=""
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some text", user_id=1)
        self.assertEqual(result["description"], "A document about something.")
        self.assertEqual(result["tags"], {})

    @patch("llm.get_llm_service")
    def test_generate_description_from_text_backward_compat(self, mock_get_service):
        from documents.services.description import generate_description_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_parsed = DocumentDescriptionOutput(
            description="A license agreement.", document_type="Agreement"
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_from_text("Some text", user_id=1)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "A license agreement.")


class DocumentDateInDescriptionTests(TestCase):
    """Tests for document_date extraction in generate_description_and_tags_from_text."""

    def test_empty_text_returns_no_date(self):
        from documents.services.description import generate_description_and_tags_from_text
        result = generate_description_and_tags_from_text("   ")
        self.assertIsNone(result.get("document_date"))

    @patch("llm.get_llm_service")
    def test_valid_date_parsed(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput
        import datetime

        mock_parsed = DocumentDescriptionOutput(
            description="A contract.", document_type="Agreement",
            document_date="2024-06-15",
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some contract text", user_id=1)
        self.assertEqual(result["document_date"], datetime.date(2024, 6, 15))

    @patch("llm.get_llm_service")
    def test_null_date_returns_none(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_parsed = DocumentDescriptionOutput(
            description="A doc.", document_type="Report",
            document_date=None,
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some text", user_id=1)
        self.assertIsNone(result["document_date"])

    @patch("llm.get_llm_service")
    def test_invalid_date_returns_none(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text
        from llm.types.structured import DocumentDescriptionOutput

        mock_parsed = DocumentDescriptionOutput(
            description="A doc.", document_type="Report",
            document_date="not-a-date",
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some text", user_id=1)
        self.assertIsNone(result["document_date"])


class PIIScanServiceTests(TestCase):
    """Tests for scan_pii_categories."""

    def test_empty_text_returns_empty(self):
        from documents.services.pii_scan import scan_pii_categories
        result = scan_pii_categories("   ")
        self.assertEqual(result, {})

    @patch("llm.get_llm_service")
    def test_returns_only_true_categories(self, mock_get_service):
        from documents.services.pii_scan import scan_pii_categories
        from llm.types.structured import PIICategoryOutput

        mock_parsed = PIICategoryOutput(
            pii_ordinary_identity=True,
            pii_ordinary_professional=True,
            pii_ordinary_communication=False,
            pii_ordinary_financial=False,
        )
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = scan_pii_categories("John Doe, Engineer at ACME Corp", user_id=1)
        self.assertEqual(result, {
            "pii_ordinary_identity": True,
            "pii_ordinary_professional": True,
        })

    @patch("llm.get_llm_service")
    def test_all_false_returns_empty(self, mock_get_service):
        from documents.services.pii_scan import scan_pii_categories
        from llm.types.structured import PIICategoryOutput

        mock_parsed = PIICategoryOutput()
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        result = scan_pii_categories("Generic technical content.", user_id=1)
        self.assertEqual(result, {})

    @patch("llm.get_llm_service")
    @patch("core.preferences.resolve_org_feature_model", return_value="openai/gpt-4o")
    def test_uses_pii_scan_feature_key(self, mock_resolve, mock_get_service):
        from documents.services.pii_scan import scan_pii_categories
        from llm.types.structured import PIICategoryOutput

        mock_parsed = PIICategoryOutput()
        mock_service = Mock()
        mock_service.run_structured.return_value = (mock_parsed, None)
        mock_get_service.return_value = mock_service

        scan_pii_categories("test text", org_id=42)
        mock_resolve.assert_called_with(42, "pii_scan")


class FileMetadataDateTests(TestCase):
    """Tests for extract_file_metadata_date."""

    def test_txt_returns_none(self):
        from documents.services.chunking import extract_file_metadata_date
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Hello")
            f.flush()
            path = Path(f.name)
        try:
            result = extract_file_metadata_date(path, "txt")
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)

    def test_unsupported_ext_returns_none(self):
        from documents.services.chunking import extract_file_metadata_date
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(b"a,b,c")
            f.flush()
            path = Path(f.name)
        try:
            result = extract_file_metadata_date(path, "csv")
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)

    def test_eml_extracts_date(self):
        import datetime
        from documents.services.chunking import extract_file_metadata_date

        eml_content = (
            b"From: sender@example.com\r\n"
            b"To: recipient@example.com\r\n"
            b"Date: Tue, 15 Oct 2024 14:30:00 +0000\r\n"
            b"Subject: Test\r\n"
            b"\r\n"
            b"Body text.\r\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as f:
            f.write(eml_content)
            f.flush()
            path = Path(f.name)
        try:
            result = extract_file_metadata_date(path, "eml")
            self.assertEqual(result, datetime.date(2024, 10, 15))
        finally:
            path.unlink(missing_ok=True)

    def test_corrupt_file_returns_none(self):
        from documents.services.chunking import extract_file_metadata_date
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"not a pdf")
            f.flush()
            path = Path(f.name)
        try:
            result = extract_file_metadata_date(path, "pdf")
            self.assertIsNone(result)
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _write_docx_with_core(core_xml):
        import zipfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = Path(f.name)
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("docProps/core.xml", core_xml)
        return path

    def test_docx_prefers_modified_over_created(self):
        import datetime
        from documents.services.chunking import extract_file_metadata_date
        # python-docx's default template hardcodes the 2013 created date; the
        # real save time lives in dcterms:modified.
        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties'
            ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
            ' xmlns:dcterms="http://purl.org/dc/terms/"'
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dcterms:created xsi:type="dcterms:W3CDTF">2013-12-23T23:15:00Z</dcterms:created>'
            '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-17T12:24:00Z</dcterms:modified>'
            '</cp:coreProperties>'
        )
        path = self._write_docx_with_core(core_xml)
        try:
            result = extract_file_metadata_date(path, "docx")
            self.assertEqual(result, datetime.date(2026, 6, 17))
        finally:
            path.unlink(missing_ok=True)

    def test_docx_falls_back_to_created_when_no_modified(self):
        import datetime
        from documents.services.chunking import extract_file_metadata_date
        core_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties'
            ' xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"'
            ' xmlns:dcterms="http://purl.org/dc/terms/"'
            ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dcterms:created xsi:type="dcterms:W3CDTF">2019-05-01T08:00:00Z</dcterms:created>'
            '</cp:coreProperties>'
        )
        path = self._write_docx_with_core(core_xml)
        try:
            result = extract_file_metadata_date(path, "docx")
            self.assertEqual(result, datetime.date(2019, 5, 1))
        finally:
            path.unlink(missing_ok=True)


class EmailLoaderTests(TestCase):
    """Tests for .msg and .eml email loading functions."""

    # ---- _format_email_as_markdown ----

    def test_format_email_as_markdown(self):
        """Shared helper produces structured Markdown with heading, table, body."""
        result = _format_email_as_markdown(
            subject="Test Subject",
            from_addr="sender@example.com",
            to_addr="recipient@example.com",
            date="2024-01-15 10:30:00",
            cc="cc@example.com",
            body_markdown="Hello, this is the body.",
            attachments=[EmailAttachment(filename="report.pdf", size_str="245 KB", content=None)],
        )
        self.assertIn("# Test Subject", result)
        self.assertIn("| **From** | sender@example.com |", result)
        self.assertIn("| **To** | recipient@example.com |", result)
        self.assertIn("| **Date** | 2024-01-15 10:30:00 |", result)
        self.assertIn("| **CC** | cc@example.com |", result)
        self.assertIn("Hello, this is the body.", result)
        self.assertIn("**Attachments:**", result)
        self.assertIn("- report.pdf (245 KB)", result)

    def test_format_email_no_subject(self):
        """None subject defaults to '(No Subject)'."""
        result = _format_email_as_markdown(
            subject=None, from_addr="a@b.com", to_addr="c@d.com",
            date=None, cc=None, body_markdown="body",
        )
        self.assertIn("# (No Subject)", result)

    def test_format_email_no_attachments(self):
        """No attachments section when list is empty/None."""
        result = _format_email_as_markdown(
            subject="Hi", from_addr="a@b.com", to_addr="c@d.com",
            date=None, cc=None, body_markdown="body",
        )
        self.assertNotIn("**Attachments:**", result)

    # ---- _load_msg_as_markdown ----

    @patch("extract_msg.Message")
    def test_load_msg_basic(self, mock_msg_cls):
        """Mock extract_msg.Message, verify structured Markdown output."""
        mock_msg = MagicMock()
        mock_msg.subject = "Meeting Notes"
        mock_msg.sender = "alice@example.com"
        mock_msg.to = "bob@example.com"
        mock_msg.date = "2024-06-01 09:00:00"
        mock_msg.cc = None
        mock_msg.htmlBody = None
        mock_msg.body = "Please review the attached."
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        self.assertEqual(len(docs), 1)
        content = docs[0].page_content
        self.assertIn("# Meeting Notes", content)
        self.assertIn("alice@example.com", content)
        self.assertIn("Please review the attached.", content)

    @patch("extract_msg.Message")
    def test_load_msg_prefers_html_body(self, mock_msg_cls):
        """When htmlBody present, markdownify is used instead of plain body."""
        mock_msg = MagicMock()
        mock_msg.subject = "HTML Email"
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = "<h1>Important</h1><p>Details here.</p>"
        mock_msg.body = "Fallback plain text"
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        content = docs[0].page_content
        # markdownify should have converted <p>Details here.</p>
        self.assertIn("Details here.", content)
        # Should NOT contain the raw fallback plain text as the body
        self.assertNotIn("Fallback plain text", content)

    @patch("extract_msg.Message")
    def test_load_msg_falls_back_to_plain(self, mock_msg_cls):
        """When htmlBody is None, plain body is used."""
        mock_msg = MagicMock()
        mock_msg.subject = "Plain Email"
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = None
        mock_msg.body = "This is plain text body."
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        self.assertIn("This is plain text body.", docs[0].page_content)

    @patch("extract_msg.Message")
    def test_load_msg_no_subject(self, mock_msg_cls):
        """Subject defaults to '(No Subject)' when None."""
        mock_msg = MagicMock()
        mock_msg.subject = None
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = None
        mock_msg.body = "body"
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        self.assertIn("# (No Subject)", docs[0].page_content)

    @patch("extract_msg.Message")
    def test_load_msg_no_body_raises(self, mock_msg_cls):
        """ValueError when both body and htmlBody are None."""
        mock_msg = MagicMock()
        mock_msg.subject = "Empty"
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = None
        mock_msg.body = None
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        with self.assertRaises(ValueError):
            _load_msg_as_markdown(Path("fake.msg"))

    @patch("extract_msg.Message")
    def test_load_msg_html_body_bytes(self, mock_msg_cls):
        """htmlBody as bytes is decoded to UTF-8."""
        mock_msg = MagicMock()
        mock_msg.subject = "Bytes HTML"
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = b"<p>Hello from bytes</p>"
        mock_msg.body = None
        mock_msg.attachments = []
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        self.assertIn("Hello from bytes", docs[0].page_content)

    # ---- _load_eml_as_markdown ----

    def test_load_eml_basic(self):
        """Construct real .eml via stdlib email.mime, verify output."""
        from email.mime.text import MIMEText

        msg = MIMEText("This is the email body.")
        msg["Subject"] = "Test EML"
        msg["From"] = "sender@example.com"
        msg["To"] = "recipient@example.com"
        msg["Date"] = "Mon, 15 Jan 2024 10:30:00 +0000"

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            self.assertEqual(len(docs), 1)
            content = docs[0].page_content
            self.assertIn("# Test EML", content)
            self.assertIn("sender@example.com", content)
            self.assertIn("This is the email body.", content)
        finally:
            eml_path.unlink()

    def test_load_eml_html_body(self):
        """Multipart .eml with HTML alternative uses markdownify."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["Subject"] = "HTML EML"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("Plain fallback", "plain"))
        msg.attach(MIMEText("<p>Rich <strong>HTML</strong> content</p>", "html"))

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            content = docs[0].page_content
            self.assertIn("Rich", content)
            self.assertIn("HTML", content)
        finally:
            eml_path.unlink()

    def test_load_eml_with_attachments(self):
        """Attachment filenames appear in output."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg["Subject"] = "With Attachment"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("See attached.", "plain"))

        att = MIMEBase("application", "octet-stream")
        att.set_payload(b"x" * 2048)
        encoders.encode_base64(att)
        att.add_header("Content-Disposition", "attachment", filename="data.bin")
        msg.attach(att)

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            content = docs[0].page_content
            self.assertIn("data.bin", content)
            self.assertIn("**Attachments:**", content)
        finally:
            eml_path.unlink()

    def test_load_eml_no_subject(self):
        """Defaults to '(No Subject)' when subject is missing."""
        from email.mime.text import MIMEText

        msg = MIMEText("body")
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        # No Subject header

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            self.assertIn("# (No Subject)", docs[0].page_content)
        finally:
            eml_path.unlink()

    # ---- load_documents routing ----

    @patch("documents.services.chunking._load_msg_as_markdown")
    def test_load_documents_routes_msg(self, mock_loader):
        """Router dispatches .msg to _load_msg_as_markdown."""
        mock_loader.return_value = [Mock(page_content="msg content")]

        with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as f:
            f.write(b"fake")
            path = Path(f.name)

        try:
            docs = load_documents(path, "msg")
            mock_loader.assert_called_once_with(path)
            self.assertEqual(len(docs), 1)
        finally:
            path.unlink()

    @patch("documents.services.chunking._load_eml_as_markdown")
    def test_load_documents_routes_eml(self, mock_loader):
        """Router dispatches .eml to _load_eml_as_markdown."""
        mock_loader.return_value = [Mock(page_content="eml content")]

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as f:
            f.write(b"fake")
            path = Path(f.name)

        try:
            docs = load_documents(path, "eml")
            mock_loader.assert_called_once_with(path)
            self.assertEqual(len(docs), 1)
        finally:
            path.unlink()

    # ---- _format_size ----

    def test_format_size(self):
        """_format_size returns correct B/KB/MB strings."""
        self.assertEqual(_format_size(500), "500 B")
        self.assertEqual(_format_size(0), "0 B")
        self.assertEqual(_format_size(1023), "1023 B")
        self.assertEqual(_format_size(1024), "1 KB")
        self.assertEqual(_format_size(2048), "2 KB")
        self.assertEqual(_format_size(1048576), "1.0 MB")
        self.assertEqual(_format_size(5242880), "5.0 MB")

    # ---- _format_email_as_markdown with EmailAttachment ----

    def test_format_email_with_extracted_attachment(self):
        """Extracted attachment renders as ## Attachment: heading with content."""
        result = _format_email_as_markdown(
            subject="Test", from_addr="a@b.com", to_addr="c@d.com",
            date=None, cc=None, body_markdown="body",
            attachments=[EmailAttachment(filename="notes.txt", size_str="1 KB", content="Hello from notes")],
        )
        self.assertIn("## Attachment: notes.txt (1 KB)", result)
        self.assertIn("Hello from notes", result)
        self.assertNotIn("**Attachments:**", result)

    def test_format_email_with_unsupported_attachment(self):
        """Unsupported attachment (content=None) renders as bullet list."""
        result = _format_email_as_markdown(
            subject="Test", from_addr="a@b.com", to_addr="c@d.com",
            date=None, cc=None, body_markdown="body",
            attachments=[EmailAttachment(filename="photo.png", size_str="2 MB", content=None)],
        )
        self.assertIn("**Attachments:**", result)
        self.assertIn("- photo.png (2 MB)", result)
        self.assertNotIn("## Attachment:", result)

    def test_format_email_mixed_attachments(self):
        """Both extracted and unsupported attachments coexist correctly."""
        result = _format_email_as_markdown(
            subject="Test", from_addr="a@b.com", to_addr="c@d.com",
            date=None, cc=None, body_markdown="body",
            attachments=[
                EmailAttachment(filename="doc.txt", size_str="1 KB", content="Text content"),
                EmailAttachment(filename="image.png", size_str="500 KB", content=None),
            ],
        )
        self.assertIn("## Attachment: doc.txt (1 KB)", result)
        self.assertIn("Text content", result)
        self.assertIn("**Attachments:**", result)
        self.assertIn("- image.png (500 KB)", result)
        # Extracted section should appear before the bullet list
        extracted_pos = result.index("## Attachment: doc.txt")
        bullet_pos = result.index("**Attachments:**")
        self.assertLess(extracted_pos, bullet_pos)

    # ---- _extract_attachment_content ----

    @override_settings(DOCUMENT_ALLOWED_EXTENSIONS={"txt", "pdf", "msg", "eml"})
    def test_extract_attachment_content_txt(self):
        """Real text extraction works end-to-end for .txt files."""
        result = _extract_attachment_content(b"Hello from attachment", "readme.txt")
        self.assertIsNotNone(result)
        self.assertIn("Hello from attachment", result)

    @override_settings(DOCUMENT_ALLOWED_EXTENSIONS={"txt", "pdf"})
    def test_extract_attachment_content_unsupported_ext(self):
        """Returns None for unsupported extensions like .png."""
        result = _extract_attachment_content(b"fake image data", "photo.png")
        self.assertIsNone(result)

    @override_settings(DOCUMENT_ALLOWED_EXTENSIONS={"txt", "pdf"})
    def test_extract_attachment_content_corrupt_file(self):
        """Returns None for corrupt file, no crash."""
        result = _extract_attachment_content(b"not a real pdf", "broken.pdf")
        self.assertIsNone(result)

    @override_settings(DOCUMENT_ALLOWED_EXTENSIONS={"txt", "msg", "eml"})
    def test_extract_attachment_content_depth_limit(self):
        """Returns None when _depth >= 1 for email extensions."""
        result = _extract_attachment_content(b"fake", "nested.eml", _depth=1)
        self.assertIsNone(result)
        result = _extract_attachment_content(b"fake", "nested.msg", _depth=1)
        self.assertIsNone(result)
        # Non-email types should still work at depth >= 1
        result = _extract_attachment_content(b"some text", "file.txt", _depth=1)
        self.assertIsNotNone(result)

    # ---- _load_msg_as_markdown with attachment extraction ----

    @patch("extract_msg.Message")
    def test_load_msg_extracts_supported_attachment(self, mock_msg_cls):
        """Mock-based: .txt attachment content appears under ## Attachment:."""
        mock_att = MagicMock()
        mock_att.longFilename = "notes.txt"
        mock_att.shortFilename = None
        mock_att.data = b"Extracted text content"
        mock_att.dataLength = len(mock_att.data)

        mock_msg = MagicMock()
        mock_msg.subject = "With TXT"
        mock_msg.sender = "a@b.com"
        mock_msg.to = "c@d.com"
        mock_msg.date = None
        mock_msg.cc = None
        mock_msg.htmlBody = None
        mock_msg.body = "See attached."
        mock_msg.attachments = [mock_att]
        mock_msg_cls.return_value = mock_msg

        docs = _load_msg_as_markdown(Path("fake.msg"))
        content = docs[0].page_content
        self.assertIn("## Attachment: notes.txt", content)
        self.assertIn("Extracted text content", content)

    # ---- _load_eml_as_markdown with attachment extraction ----

    def test_load_eml_extracts_supported_attachment(self):
        """Real .eml with .txt attachment — full integration."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg["Subject"] = "With TXT Attachment"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("See attached.", "plain"))

        att = MIMEBase("text", "plain")
        att.set_payload(b"Hello from the text file")
        encoders.encode_base64(att)
        att.add_header("Content-Disposition", "attachment", filename="readme.txt")
        msg.attach(att)

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            content = docs[0].page_content
            self.assertIn("## Attachment: readme.txt", content)
            self.assertIn("Hello from the text file", content)
        finally:
            eml_path.unlink()

    def test_load_eml_corrupt_attachment_falls_back(self):
        """Garbage .pdf attachment → listed by name only."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg["Subject"] = "Corrupt PDF"
        msg["From"] = "a@b.com"
        msg["To"] = "c@d.com"
        msg.attach(MIMEText("See attached.", "plain"))

        att = MIMEBase("application", "pdf")
        att.set_payload(b"this is not a real pdf")
        encoders.encode_base64(att)
        att.add_header("Content-Disposition", "attachment", filename="broken.pdf")
        msg.attach(att)

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(msg.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            content = docs[0].page_content
            self.assertIn("broken.pdf", content)
            self.assertIn("**Attachments:**", content)
            self.assertNotIn("## Attachment:", content)
        finally:
            eml_path.unlink()

    def test_load_eml_nested_eml_depth_limit(self):
        """Nested .eml body extracted at depth 0, but its own .eml attachment is not recursed."""
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        # Inner-inner email (should NOT be extracted — depth 2)
        inner_inner = MIMEText("You should not see this inner-inner body.")
        inner_inner["Subject"] = "Inner Inner"
        inner_inner["From"] = "z@z.com"
        inner_inner["To"] = "z@z.com"

        # Inner email with its own .eml attachment
        inner = MIMEMultipart()
        inner["Subject"] = "Inner Email"
        inner["From"] = "y@y.com"
        inner["To"] = "y@y.com"
        inner.attach(MIMEText("Inner email body content.", "plain"))

        # Use application/octet-stream (how most clients attach .eml files)
        inner_att = MIMEBase("application", "octet-stream")
        inner_att.set_payload(inner_inner.as_bytes())
        encoders.encode_base64(inner_att)
        inner_att.add_header("Content-Disposition", "attachment", filename="deep.eml")
        inner.attach(inner_att)

        # Outer email with inner .eml attachment
        outer = MIMEMultipart()
        outer["Subject"] = "Outer Email"
        outer["From"] = "a@b.com"
        outer["To"] = "c@d.com"
        outer.attach(MIMEText("Outer body.", "plain"))

        outer_att = MIMEBase("application", "octet-stream")
        outer_att.set_payload(inner.as_bytes())
        encoders.encode_base64(outer_att)
        outer_att.add_header("Content-Disposition", "attachment", filename="inner.eml")
        outer.attach(outer_att)

        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False, mode="wb") as f:
            f.write(outer.as_bytes())
            eml_path = Path(f.name)

        try:
            docs = _load_eml_as_markdown(eml_path)
            content = docs[0].page_content
            # Outer body
            self.assertIn("Outer body.", content)
            # Inner email extracted as attachment at depth 0→1
            self.assertIn("## Attachment: inner.eml", content)
            self.assertIn("Inner email body content.", content)
            # Inner-inner should NOT be extracted (depth 1→2 blocked)
            self.assertNotIn("You should not see this inner-inner body.", content)
            # deep.eml should appear as bullet-list attachment within the inner email content
            self.assertIn("deep.eml", content)
        finally:
            eml_path.unlink()

    # ---- process_document parser_type ----

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="structure_aware")
    def test_process_document_msg_parser_type(self):
        """process_document sets parser_type='msg' for .msg files."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        user = User.objects.create_user(email="msg@example.com", password="testpass")
        data_room = DataRoom.objects.create(name="MsgProject", slug="msg-project", created_by=user)

        sample_chunks = [
            {"text": "Chunk 0", "token_count": 10, "chunk_index": 0, "heading": "Subject"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=data_room,
                    uploaded_by=user,
                    original_filename="email.msg",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("email.msg", ContentFile(b"fake msg"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="email content")]), \
                     patch("documents.services.process_document.structure_aware_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay"), \
                     patch("documents.services.pii_scan.pii_gate_applies", return_value=False):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.parser_type, "msg")
                # process_document always holds documents in SCANNING; the
                # guardrail scan releases them to READY.
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="structure_aware")
    def test_process_document_eml_parser_type(self):
        """process_document sets parser_type='eml' for .eml files."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        user = User.objects.create_user(email="eml@example.com", password="testpass")
        data_room = DataRoom.objects.create(name="EmlProject", slug="eml-project", created_by=user)

        sample_chunks = [
            {"text": "Chunk 0", "token_count": 10, "chunk_index": 0, "heading": "Subject"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=data_room,
                    uploaded_by=user,
                    original_filename="email.eml",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("email.eml", ContentFile(b"fake eml"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="email content")]), \
                     patch("documents.services.process_document.structure_aware_chunk", return_value=sample_chunks), \
                     patch("guardrails.tasks.scan_document_version.delay"), \
                     patch("documents.services.pii_scan.pii_gate_applies", return_value=False):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.parser_type, "eml")
                # process_document always holds documents in SCANNING; the
                # guardrail scan releases them to READY.
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)


class ProcessDocumentAudioTests(TestCase):
    """Tests for audio file processing through the document pipeline."""

    def setUp(self):
        self.user = User.objects.create_user(email="audio@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="AudioProject", slug="audio-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="structure_aware")
    def test_process_document_audio_file(self):
        """Audio file goes through transcribe -> chunk -> SCANNING (guardrail gate)."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {"text": "Transcript chunk", "heading": None, "token_count": 15,
             "chunk_index": 0,
             "source_page_start": None, "source_page_end": None,
             "source_offset_start": 0, "source_offset_end": 16},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="meeting.mp3",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("meeting.mp3", ContentFile(b"\x00" * 100), save=True)

                mock_prefs = MagicMock()
                mock_prefs.allowed_transcription_models = ["openai/gpt-4o-mini-transcribe"]
                mock_prefs.transcription_model = "openai/gpt-4o-mini-transcribe"

                with patch("core.preferences.get_preferences", return_value=mock_prefs), \
                     patch("documents.services.transcription.transcribe_audio", return_value="This is a transcript.") as mock_transcribe, \
                     patch("documents.services.process_document.structure_aware_chunk", return_value=sample_chunks) as mock_chunk, \
                     patch("guardrails.tasks.scan_document_version.delay"), \
                     patch("documents.services.pii_scan.pii_gate_applies", return_value=False):
                    process_document(doc.id)

                mock_transcribe.assert_called_once()
                doc.refresh_from_db()
                # process_document always holds documents in SCANNING; the
                # guardrail scan releases them to READY.
                self.assertEqual(doc.status, DataRoomDocument.Status.SCANNING)
                self.assertEqual(doc.parser_type, "audio")
                self.assertEqual(doc.transcript, "This is a transcript.")
                self.assertEqual(doc.transcription_model, "openai/gpt-4o-mini-transcribe")
                self.assertEqual(doc.current_version.chunks.count(), 1)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_audio_transcription_disabled(self):
        """When org disallows transcription, audio doc should fail."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="disabled.wav",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("disabled.wav", ContentFile(b"\x00" * 100), save=True)

                mock_prefs = MagicMock()
                mock_prefs.allowed_transcription_models = []
                mock_prefs.transcription_model = ""

                with patch("core.preferences.get_preferences", return_value=mock_prefs):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("not enabled", doc.processing_error)


class ResourceGuardTests(TestCase):
    """Decompression-bomb and size guards in the processing pipeline."""

    @override_settings(DOCX_MAX_UNCOMPRESSED_BYTES=1000)
    def test_docx_uncompressed_size_guard(self):
        """A docx whose declared uncompressed size exceeds the cap is rejected
        before mammoth ever unzips it."""
        import io
        import zipfile

        from documents.services.chunking import _load_docx_as_markdown

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bomb.docx"
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("word/document.xml", b"\x00" * 10_000)  # 10 KB > 1 KB cap
            path.write_bytes(buf.getvalue())

            with self.assertRaises(ValueError) as ctx:
                _load_docx_as_markdown(path)
        self.assertIn("unusually large", str(ctx.exception))

    def test_docx_corrupt_zip_rejected(self):
        from documents.services.chunking import _load_docx_as_markdown

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.docx"
            path.write_bytes(b"this is not a zip file")

            with self.assertRaises(ValueError) as ctx:
                _load_docx_as_markdown(path)
        self.assertIn("corrupt", str(ctx.exception))

    @override_settings(DOCUMENT_ATTACHMENT_MAX_BYTES=10)
    def test_attachment_size_guard(self):
        """Oversized email attachments are listed but never extracted."""
        self.assertIsNone(_extract_attachment_content(b"x" * 100, "big.txt"))
        self.assertEqual(_extract_attachment_content(b"hello", "ok.txt"), "hello")

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic", DOCUMENT_MAX_EXTRACTED_CHARS=100)
    def test_extracted_text_cap_marks_failed(self):
        """Extraction output beyond the cap fails the document before chunking."""
        from django.core.files.base import ContentFile

        from documents.services.process_document import process_document

        user = User.objects.create_user(email="guard@example.com", password="testpass")
        data_room = DataRoom.objects.create(name="GuardProject", slug="guard-project", created_by=user)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=data_room,
                    uploaded_by=user,
                    original_filename="huge.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("huge.txt", ContentFile(b"x"), save=True)

                with patch("documents.services.process_document.load_documents",
                           return_value=[Mock(page_content="y" * 5000)]), \
                     patch("documents.services.process_document.semantic_chunk") as mock_chunk:
                    process_document(doc.id)

                mock_chunk.assert_not_called()
                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
                self.assertIn("too large to process", doc.processing_error)
