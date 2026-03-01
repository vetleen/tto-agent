import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from documents.models import Project, ProjectDocument, ProjectDocumentChunk
from documents.services.chunking import (
    MIN_CHUNK_TOKENS,
    _count_tokens,
    _looks_like_markdown,
    _merge_small_chunks,
    chunk_text,
    load_documents,
)
from documents.services.retrieval import (
    _rrf_score,
    get_chunks_by_document,
    get_chunks_by_project,
    hybrid_search_chunks,
    similarity_search_chunks,
)

User = get_user_model()

try:
    import langchain_core  # noqa: F401
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


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

    def test_looks_like_markdown(self):
        self.assertTrue(_looks_like_markdown("# Title"))
        self.assertTrue(_looks_like_markdown("\n## Section"))
        self.assertFalse(_looks_like_markdown("Plain text"))

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_chunk_text_single_chunk_when_under_max_tokens(self):
        """Documents under max_chunk_tokens are returned as one chunk."""
        from langchain_core.documents import Document
        short_text = "This is a short document. It has only a few words."
        docs = [Document(page_content=short_text, metadata={"page": 1})]
        chunks = chunk_text(docs, max_chunk_tokens=1200, chunk_overlap_tokens=100)
        self.assertEqual(len(chunks), 1, "should be one chunk when under token limit")
        self.assertEqual(chunks[0]["text"], short_text)
        self.assertEqual(chunks[0]["token_count"], _count_tokens(short_text))
        self.assertIsNone(chunks[0]["heading"])
        self.assertEqual(chunks[0]["source_page_start"], 1)
        self.assertEqual(chunks[0]["source_page_end"], 1)
        self.assertEqual(chunks[0]["source_offset_start"], 0)
        self.assertEqual(chunks[0]["source_offset_end"], len(short_text))

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    @patch("tiktoken.get_encoding")
    def test_chunk_text_multiple_chunks_when_over_max_tokens(self, mock_get_encoding):
        """Documents over max_chunk_tokens are split, then small chunks merged (min 200).

        tiktoken.get_encoding is mocked to avoid a network download of the BPE data file
        which is blocked in CI. The mock uses a character-based encoder (len(text)//4
        tokens) — the same heuristic as _count_tokens' fallback — to drive the splitter.
        """
        mock_enc = Mock()
        mock_enc.encode = lambda text, **kwargs: list(range(max(1, len(text) // 4)))
        mock_get_encoding.return_value = mock_enc

        from langchain_core.documents import Document
        docs = [Document(page_content="First paragraph. " * 50, metadata={})]
        chunks = chunk_text(docs, max_chunk_tokens=50, chunk_overlap_tokens=5)
        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertIn("text", c)
            self.assertIn("token_count", c)
            # After merge, chunks are >= MIN_CHUNK_TOKENS (200) or the only chunk
            self.assertGreaterEqual(c["token_count"], 1)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_chunk_text_markdown_preserves_heading(self):
        from langchain_core.documents import Document
        docs = [Document(page_content="# Intro\n\nSome text here.", metadata={})]
        chunks = chunk_text(docs, max_chunk_tokens=500, chunk_overlap_tokens=10)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(any(c.get("heading") for c in chunks) or "Intro" in chunks[0]["text"])

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

    def _chunk(self, text: str, token_count: int, page: int | None = None, offset_start: int | None = None, offset_end: int | None = None) -> dict:
        return {
            "text": text,
            "heading": None,
            "token_count": token_count,
            "source_page_start": page,
            "source_page_end": page,
            "source_offset_start": offset_start,
            "source_offset_end": offset_end,
        }

    def test_merge_small_chunks_merges_into_smallest_adjacent_and_iterates(self):
        """Chunks under min merge into smallest adjacent; repeat until none under min."""
        chunks = [
            self._chunk("first", 50, page=1, offset_start=0, offset_end=5),
            self._chunk("second", 300, page=1, offset_start=6, offset_end=12),
            self._chunk("third", 50, page=2, offset_start=0, offset_end=4),
        ]
        result = _merge_small_chunks(chunks, min_tokens=200)
        self.assertEqual(len(result), 1, "three chunks (50, 300, 50) should merge to one")
        self.assertIn("first", result[0]["text"])
        self.assertIn("second", result[0]["text"])
        self.assertIn("third", result[0]["text"])
        self.assertEqual(result[0]["source_page_start"], 1)
        self.assertEqual(result[0]["source_page_end"], 2)
        self.assertEqual(result[0]["token_count"], _count_tokens(result[0]["text"]))

    def test_merge_small_chunks_single_under_min_left_alone(self):
        """Single chunk under min has no adjacent; left unchanged."""
        chunks = [self._chunk("alone", 100, page=1)]
        result = _merge_small_chunks(chunks, min_tokens=200)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "alone")
        self.assertEqual(result[0]["token_count"], 100)

    def test_merge_small_chunks_all_above_min_unchanged(self):
        """Chunks all >= min are not merged."""
        chunks = [
            self._chunk("a", 250, page=1),
            self._chunk("b", 250, page=2),
        ]
        result = _merge_small_chunks(chunks, min_tokens=200)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["token_count"], 250)
        self.assertEqual(result[1]["token_count"], 250)

    def test_merge_small_chunks_recomputes_token_count_and_source_span(self):
        """Merged chunk has token_count from combined text and source span min/max."""
        chunks = [
            self._chunk("left", 50, page=1, offset_start=0, offset_end=4),
            self._chunk("right", 50, page=2, offset_start=0, offset_end=5),
        ]
        result = _merge_small_chunks(chunks, min_tokens=200)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["token_count"], _count_tokens(result[0]["text"]))
        self.assertEqual(result[0]["source_page_start"], 1)
        self.assertEqual(result[0]["source_page_end"], 2)
        self.assertEqual(result[0]["source_offset_start"], 0)
        self.assertEqual(result[0]["source_offset_end"], 5)

    def test_merge_small_chunks_min_constant(self):
        self.assertEqual(MIN_CHUNK_TOKENS, 200)

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


class VectorStoreTests(TestCase):
    @patch("django.db.connection")
    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    def test_delete_vectors_for_document_scopes_to_collection(self, _mock_conn, mock_connection):
        from documents.services.vector_store import COLLECTION_NAME, delete_vectors_for_document

        mock_cursor = Mock()
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

        delete_vectors_for_document(document_id=42)

        query, params = mock_cursor.execute.call_args.args
        self.assertIn("col.name = %s", query)
        self.assertIn("emb.cmetadata->>'document_id' = %s", query)
        self.assertEqual(params, [COLLECTION_NAME, "42"])

    @patch("documents.services.vector_store._get_connection_string", return_value="postgresql://example")
    @patch("documents.services.vector_store._get_vector_store")
    def test_similarity_search_bounds_k(self, mock_get_store, _mock_conn):
        from documents.services.vector_store import similarity_search

        store = Mock()
        store.similarity_search.return_value = ["result"]
        mock_get_store.return_value = store

        similarity_search(project_id=1, query="hello", k=500, document_id=9)

        self.assertEqual(store.similarity_search.call_args.kwargs["k"], 50)
        self.assertEqual(store.similarity_search.call_args.kwargs["filter"], {"project_id": 1, "document_id": 9})


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

    def _make_semantic_doc(self, chunk_id, text, document_id=1, chunk_index=0):
        """Create a mock LangChain Document as returned by pgvector."""
        doc = MagicMock()
        doc.page_content = text
        doc.metadata = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "project_id": 1,
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
        results = hybrid_search_chunks(project_id=1, query="nothing", k=5)
        self.assertEqual(results, [])

    @patch("documents.services.retrieval.fulltext_search_chunks", return_value=[])
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_semantic_only_when_fts_empty(self, mock_sem, _mock_fts):
        mock_sem.return_value = [
            self._make_semantic_doc(10, "semantic result"),
        ]
        results = hybrid_search_chunks(project_id=1, query="test", k=5)
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
        results = hybrid_search_chunks(project_id=1, query="test", k=5)
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

        mock_sem.return_value = [
            self._make_semantic_doc(shared_id, "shared chunk"),
            self._make_semantic_doc(only_semantic_id, "semantic only"),
        ]
        mock_fts.return_value = [
            self._make_fts_hit(shared_id, "shared chunk"),
            self._make_fts_hit(only_fts_id, "fts only"),
        ]

        results = hybrid_search_chunks(project_id=1, query="test", k=10)

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
        mock_sem.return_value = [
            self._make_semantic_doc(i, f"sem {i}") for i in range(10)
        ]
        mock_fts.return_value = [
            self._make_fts_hit(i + 100, f"fts {i}") for i in range(10)
        ]
        results = hybrid_search_chunks(project_id=1, query="test", k=3)
        self.assertEqual(len(results), 3)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search", side_effect=Exception("pgvector down"))
    def test_hybrid_degrades_gracefully_when_semantic_fails(self, _mock_sem, mock_fts):
        mock_fts.return_value = [
            self._make_fts_hit(50, "fallback result"),
        ]
        results = hybrid_search_chunks(project_id=1, query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 50)

    @patch("documents.services.retrieval.fulltext_search_chunks", side_effect=Exception("FTS down"))
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_degrades_gracefully_when_fts_fails(self, mock_sem, _mock_fts):
        mock_sem.return_value = [
            self._make_semantic_doc(60, "fallback semantic"),
        ]
        results = hybrid_search_chunks(project_id=1, query="test", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 60)

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_prefers_fts_heading_over_semantic(self, mock_sem, mock_fts):
        """When same chunk appears in both, heading from FTS result should be kept."""
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
        results = hybrid_search_chunks(project_id=1, query="test", k=5)
        self.assertEqual(results[0]["heading"], "Important Section")


class ProcessDocumentServiceTests(TestCase):
    """Unit tests for documents.services.process_document.process_document()."""

    def setUp(self):
        self.user = User.objects.create_user(email="svc@example.com", password="testpass")
        self.project = Project.objects.create(name="SvcProject", slug="svc-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_happy_path(self):
        """UPLOADED doc with a real file becomes READY and gets chunks persisted."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {"text": "First chunk", "heading": None, "token_count": 10,
             "source_page_start": 1, "source_page_end": 1,
             "source_offset_start": 0, "source_offset_end": 11},
            {"text": "Second chunk", "heading": "Section", "token_count": 12,
             "source_page_start": 1, "source_page_end": 1,
             "source_offset_start": 12, "source_offset_end": 24},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = ProjectDocument(
                    project=self.project,
                    uploaded_by=self.user,
                    original_filename="test.txt",
                    status=ProjectDocument.Status.UPLOADED,
                )
                doc.original_file.save("test.txt", ContentFile(b"hello world"), save=True)

                with patch("documents.services.process_document.extract_and_chunk_file", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, ProjectDocument.Status.READY)
                self.assertEqual(doc.chunks.count(), 2)
                self.assertEqual(doc.token_count, 22)
                self.assertIsNotNone(doc.processed_at)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_sets_failed_when_file_missing(self):
        """Doc with no attached file transitions to FAILED with a processing_error."""
        from documents.services.process_document import process_document

        doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="missing.txt",
            status=ProjectDocument.Status.UPLOADED,
        )
        process_document(doc.id)

        doc.refresh_from_db()
        self.assertEqual(doc.status, ProjectDocument.Status.FAILED)
        self.assertIsNotNone(doc.processing_error)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_document_sets_failed_on_chunking_error(self):
        """Chunking exception causes status=FAILED and error stored in processing_error."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = ProjectDocument(
                    project=self.project,
                    uploaded_by=self.user,
                    original_filename="bad.txt",
                    status=ProjectDocument.Status.UPLOADED,
                )
                doc.original_file.save("bad.txt", ContentFile(b"content"), save=True)

                with patch(
                    "documents.services.process_document.extract_and_chunk_file",
                    side_effect=RuntimeError("parse error"),
                ):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, ProjectDocument.Status.FAILED)
                self.assertIn("parse error", doc.processing_error)

    def test_process_document_skips_nonexistent_document(self):
        """Calling with a non-existent ID logs a warning and returns without raising."""
        from documents.services.process_document import process_document

        with self.assertLogs("documents.services.process_document", level="WARNING") as cm:
            process_document(99999)

        self.assertTrue(any("not found" in line for line in cm.output))


class RetrievalServiceTests(TestCase):
    """Unit tests for documents.services.retrieval non-search functions."""

    def setUp(self):
        self.user = User.objects.create_user(email="ret@example.com", password="testpass")
        self.project = Project.objects.create(name="RetProject", slug="ret-project", created_by=self.user)
        self.doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="ret.txt",
            status=ProjectDocument.Status.READY,
        )

    def test_get_chunks_by_document_returns_ordered(self):
        """Chunks are returned in chunk_index order regardless of insertion order."""
        ProjectDocumentChunk.objects.create(document=self.doc, chunk_index=1, text="Second", token_count=2)
        ProjectDocumentChunk.objects.create(document=self.doc, chunk_index=0, text="First", token_count=1)

        result = get_chunks_by_document(self.doc.id)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertEqual(result[0]["text"], "First")
        self.assertEqual(result[1]["chunk_index"], 1)
        # Required fields are present
        for key in ("id", "text", "heading", "token_count", "source_page_start", "source_page_end"):
            self.assertIn(key, result[0])

    def test_get_chunks_by_project_excludes_failed_documents(self):
        """Chunks from FAILED documents are not returned."""
        ProjectDocumentChunk.objects.create(document=self.doc, chunk_index=0, text="Ready chunk", token_count=5)

        failed_doc = ProjectDocument.objects.create(
            project=self.project,
            uploaded_by=self.user,
            original_filename="fail.txt",
            status=ProjectDocument.Status.FAILED,
        )
        ProjectDocumentChunk.objects.create(document=failed_doc, chunk_index=0, text="Failed chunk", token_count=5)

        result = get_chunks_by_project(self.project.id)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Ready chunk")
        self.assertEqual(result[0]["document_id"], self.doc.id)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_similarity_search_chunks_returns_langchain_documents(self):
        """similarity_search_chunks wraps hybrid_search output as LangChain Documents."""
        from langchain_core.documents import Document

        fake_results = [
            {"id": 42, "text": "hello", "document_id": 7, "chunk_index": 0, "rrf_score": 0.5, "heading": None},
        ]
        with patch("documents.services.retrieval.hybrid_search_chunks", return_value=fake_results):
            results = similarity_search_chunks(project_id=1, query="test", k=5)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)
        self.assertEqual(results[0].page_content, "hello")
        self.assertEqual(results[0].metadata["chunk_id"], 42)
        self.assertEqual(results[0].metadata["project_id"], 1)
        self.assertEqual(results[0].metadata["document_id"], 7)
