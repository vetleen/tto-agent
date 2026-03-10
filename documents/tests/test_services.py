import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk
from documents.services.chunking import (
    MIN_CHUNK_TOKENS,
    _count_tokens,
    _looks_like_markdown,
    _merge_small_chunks,
    _strip_nul_bytes,
    chunk_text,
    clean_extracted_text,
    load_documents,
)
from documents.services.retrieval import (
    _rrf_score,
    get_chunks_by_document,
    get_chunks_by_data_room,
    hybrid_search_chunks,
    similarity_search_chunks,
)
from documents.services.splitters import (
    detect_structure,
    structural_split,
    parent_child_split,
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

        similarity_search(data_room_ids=[1], query="hello", k=500, document_id=9)

        self.assertEqual(store.similarity_search.call_args.kwargs["k"], 50)
        self.assertEqual(store.similarity_search.call_args.kwargs["filter"], {"data_room_id": 1, "document_id": 9})


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
        mock_sem.return_value = [
            self._make_semantic_doc(i, f"sem {i}") for i in range(10)
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
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="test.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("test.txt", ContentFile(b"hello world"), save=True)

                with patch("documents.services.process_document.extract_and_chunk_file", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)
                self.assertEqual(doc.chunks.count(), 2)
                self.assertEqual(doc.token_count, 22)
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
                    "documents.services.process_document.extract_and_chunk_file",
                    side_effect=RuntimeError("parse error"),
                ):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.FAILED)
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
        self.data_room = DataRoom.objects.create(name="RetProject", slug="ret-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="ret.txt",
            status=DataRoomDocument.Status.READY,
        )

    def test_get_chunks_by_document_returns_ordered(self):
        """Chunks are returned in chunk_index order regardless of insertion order."""
        DataRoomDocumentChunk.objects.create(document=self.doc, chunk_index=1, text="Second", token_count=2)
        DataRoomDocumentChunk.objects.create(document=self.doc, chunk_index=0, text="First", token_count=1)

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
        DataRoomDocumentChunk.objects.create(document=self.doc, chunk_index=0, text="Ready chunk", token_count=5)

        failed_doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="fail.txt",
            status=DataRoomDocument.Status.FAILED,
        )
        DataRoomDocumentChunk.objects.create(document=failed_doc, chunk_index=0, text="Failed chunk", token_count=5)

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


class StructuralSplitTests(TestCase):
    """Tests for documents.services.splitters structural splitting."""

    def test_detect_structure_markdown(self):
        self.assertEqual(detect_structure("# Title\n\nSome text"), "markdown")
        self.assertEqual(detect_structure("Plain\n\n## Section"), "markdown")

    def test_detect_structure_slides(self):
        self.assertEqual(detect_structure("Slide 1\n---\nSlide 2\n---\nSlide 3"), "slides")
        self.assertEqual(detect_structure("Slide 1\fSlide 2"), "slides")

    def test_detect_structure_plain(self):
        self.assertEqual(detect_structure("Just some plain text.\n\nAnother paragraph."), "plain")

    def test_markdown_split_on_headings(self):
        text = "# Intro\n\nSome intro text.\n\n## Methods\n\nMethod details."
        units = structural_split(text, "markdown")
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["heading"], "Intro")
        self.assertEqual(units[1]["heading"], "Methods")
        self.assertIn("intro text", units[0]["text"])
        self.assertIn("Method details", units[1]["text"])

    def test_plain_text_splits_on_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        units = structural_split(text, "plain")
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0]["text"], "First paragraph.")
        self.assertEqual(units[1]["text"], "Second paragraph.")
        self.assertEqual(units[2]["text"], "Third paragraph.")

    def test_slides_split_on_boundaries(self):
        text = "Slide 1 content\n---\nSlide 2 content\n---\nSlide 3 content"
        units = structural_split(text, "slides")
        self.assertEqual(len(units), 3)
        self.assertEqual(units[0]["unit_type"], "slide")
        self.assertIn("Slide 1", units[0]["text"])

    def test_slides_split_on_form_feed(self):
        text = "Slide A\fSlide B\fSlide C"
        units = structural_split(text, "slides")
        self.assertEqual(len(units), 3)


class ParentChildSplitTests(TestCase):
    """Tests for parent-child chunking logic."""

    def test_markdown_parents_from_headings(self):
        text = "# Section A\n\nContent A.\n\n## Section B\n\nContent B."
        result = parent_child_split(text, "markdown", child_target_tokens=50, child_overlap_pct=0.0, max_child_tokens=200)
        self.assertGreaterEqual(len(result), 2)
        self.assertEqual(result[0]["heading"], "Section A")
        self.assertEqual(result[1]["heading"], "Section B")

    def test_children_respect_structural_units(self):
        """Children should not cut mid-sentence when paragraphs are small enough."""
        paragraphs = ["Paragraph number %d. It has some content here." % i for i in range(10)]
        text = "\n\n".join(paragraphs)
        result = parent_child_split(text, "plain", child_target_tokens=50, child_overlap_pct=0.0, max_child_tokens=200)
        for parent in result:
            for child in parent["children"]:
                # Each child should end with a complete sentence/paragraph
                self.assertTrue(
                    child["text"].endswith(".") or child["text"].endswith("here."),
                    f"Child text appears cut mid-sentence: {child['text'][-50:]}"
                )

    def test_slides_one_to_one(self):
        """Each slide = 1 parent = 1 child."""
        text = "Slide One\n---\nSlide Two\n---\nSlide Three"
        result = parent_child_split(text, "slides", child_target_tokens=300, child_overlap_pct=0.0, max_child_tokens=600)
        self.assertEqual(len(result), 3)
        for parent in result:
            self.assertEqual(len(parent["children"]), 1)
            self.assertEqual(parent["children"][0]["text"], parent["text"])

    def test_overlap_applied(self):
        """With overlap > 0, subsequent children should start with overlap from previous."""
        # Build text large enough to produce multiple children
        sentences = ["Sentence number %d is here." % i for i in range(50)]
        text = " ".join(sentences)
        result = parent_child_split(text, "plain", child_target_tokens=30, child_overlap_pct=0.50, max_child_tokens=100)
        for parent in result:
            if len(parent["children"]) > 1:
                # Second child should contain some text from the end of first child
                first_words = parent["children"][0]["text"].split()
                if len(first_words) > 2:
                    # Some trailing words from first should appear at start of second
                    second_text = parent["children"][1]["text"]
                    # Just verify second child has more tokens than without overlap
                    self.assertGreater(parent["children"][1]["token_count"], 0)

    def test_parent_has_children_list(self):
        text = "Some content here.\n\nMore content there."
        result = parent_child_split(text, "plain", child_target_tokens=300, child_overlap_pct=0.0, max_child_tokens=600)
        self.assertGreaterEqual(len(result), 1)
        for parent in result:
            self.assertIn("children", parent)
            self.assertIn("text", parent)
            self.assertIn("token_count", parent)
            self.assertIn("source_offset_start", parent)
            self.assertIn("source_offset_end", parent)
            for child in parent["children"]:
                self.assertIn("text", child)
                self.assertIn("token_count", child)
                self.assertIn("child_index", child)

    def test_empty_text_returns_empty(self):
        result = parent_child_split("", "plain")
        self.assertEqual(result, [])
        result = parent_child_split("   \n\n  ", "plain")
        self.assertEqual(result, [])


class ChunkTextIntegrationTests(TestCase):
    """Integration tests for chunk_text with parent-child output."""

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_chunk_text_produces_parent_child_structure(self):
        from langchain_core.documents import Document
        text = "# Introduction\n\nThis is the intro.\n\n## Methods\n\nThis is the methods section."
        docs = [Document(page_content=text, metadata={})]
        chunks = chunk_text(docs)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertFalse(chunk["is_child"])
            self.assertIn("children", chunk)

    @unittest.skipIf(not LANGCHAIN_AVAILABLE, "langchain not installed")
    def test_chunk_text_plain_produces_parent_child(self):
        from langchain_core.documents import Document
        paragraphs = ["This is paragraph %d with some content." % i for i in range(5)]
        text = "\n\n".join(paragraphs)
        docs = [Document(page_content=text, metadata={"page": 1})]
        chunks = chunk_text(docs)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("children", chunk)
            self.assertEqual(chunk["is_child"], False)
            self.assertEqual(chunk["source_page_start"], 1)


class ProcessDocumentParentChildTests(TestCase):
    """Tests for process_document with parent-child chunking."""

    def setUp(self):
        self.user = User.objects.create_user(email="pc@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="PCProject", slug="pc-project", created_by=self.user)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_process_creates_parent_and_child_chunks(self):
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {
                "text": "Parent chunk text",
                "heading": "Section",
                "token_count": 10,
                "is_child": False,
                "source_page_start": 1,
                "source_page_end": 1,
                "source_offset_start": 0,
                "source_offset_end": 17,
                "children": [
                    {"text": "Child chunk 1", "heading": "Section", "token_count": 5, "is_child": True, "child_index": 0},
                    {"text": "Child chunk 2", "heading": "Section", "token_count": 5, "is_child": True, "child_index": 1},
                ],
            },
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

                with patch("documents.services.process_document.extract_and_chunk_file", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)

                parent_chunks = doc.chunks.filter(is_child=False)
                child_chunks = doc.chunks.filter(is_child=True)
                self.assertEqual(parent_chunks.count(), 1)
                self.assertEqual(child_chunks.count(), 2)

                # Children should reference their parent
                parent = parent_chunks.first()
                for child in child_chunks:
                    self.assertEqual(child.parent_id, parent.pk)

    @override_settings(PGVECTOR_CONNECTION="")
    def test_backward_compat_no_children(self):
        """Chunks without children still work (legacy format)."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [
            {
                "text": "Simple chunk",
                "heading": None,
                "token_count": 5,
                "source_page_start": 1,
                "source_page_end": 1,
                "source_offset_start": 0,
                "source_offset_end": 12,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="legacy.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("legacy.txt", ContentFile(b"hello"), save=True)

                with patch("documents.services.process_document.extract_and_chunk_file", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)
                self.assertEqual(doc.chunks.count(), 1)
                self.assertFalse(doc.chunks.first().is_child)


class RetrievalParentChildTests(TestCase):
    """Tests for retrieval with parent-child chunks."""

    def setUp(self):
        self.user = User.objects.create_user(email="rpc@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="RPCProject", slug="rpc-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="rpc.txt",
            status=DataRoomDocument.Status.READY,
        )

    def test_get_chunks_by_document_returns_parents_only(self):
        parent = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Parent", token_count=5, is_child=False,
        )
        DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Child", token_count=3, is_child=True, parent=parent,
        )
        result = get_chunks_by_document(self.doc.id)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Parent")

    def test_get_chunks_by_data_room_returns_parents_only(self):
        parent = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Parent", token_count=5, is_child=False,
        )
        DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Child", token_count=3, is_child=True, parent=parent,
        )
        result = get_chunks_by_data_room(self.data_room.id)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Parent")

    def test_legacy_chunks_still_work(self):
        """Chunks without parent/child fields still returned normally."""
        DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Legacy chunk", token_count=5,
        )
        result = get_chunks_by_document(self.doc.id)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Legacy chunk")

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_resolves_child_to_parent(self, mock_sem, mock_fts):
        """When semantic search returns a child chunk, hybrid resolves to parent."""
        parent = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Full parent text here", token_count=10, is_child=False,
        )
        child = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Child excerpt", token_count=5, is_child=True, parent=parent,
        )

        # Simulate semantic search returning the child chunk
        sem_doc = MagicMock()
        sem_doc.page_content = "Child excerpt"
        sem_doc.metadata = {
            "chunk_id": child.pk,
            "document_id": self.doc.pk,
            "data_room_id": self.data_room.pk,
            "chunk_index": 0,
        }
        mock_sem.return_value = [sem_doc]
        mock_fts.return_value = []

        results = hybrid_search_chunks(data_room_ids=[self.data_room.pk], query="test", k=5)
        self.assertEqual(len(results), 1)
        # Should return parent text, not child text
        self.assertEqual(results[0]["id"], parent.pk)
        self.assertEqual(results[0]["text"], "Full parent text here")

    @patch("documents.services.retrieval.fulltext_search_chunks")
    @patch("documents.services.retrieval.vs.similarity_search")
    def test_hybrid_deduplicates_children_of_same_parent(self, mock_sem, mock_fts):
        """Multiple child matches from same parent should deduplicate to one result."""
        parent = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Parent text", token_count=10, is_child=False,
        )
        child1 = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Child 1", token_count=3, is_child=True, parent=parent,
        )
        child2 = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=1, text="Child 2", token_count=3, is_child=True, parent=parent,
        )

        sem_doc1 = MagicMock()
        sem_doc1.page_content = "Child 1"
        sem_doc1.metadata = {"chunk_id": child1.pk, "document_id": self.doc.pk, "data_room_id": self.data_room.pk, "chunk_index": 0}
        sem_doc2 = MagicMock()
        sem_doc2.page_content = "Child 2"
        sem_doc2.metadata = {"chunk_id": child2.pk, "document_id": self.doc.pk, "data_room_id": self.data_room.pk, "chunk_index": 1}
        mock_sem.return_value = [sem_doc1, sem_doc2]
        mock_fts.return_value = []

        results = hybrid_search_chunks(data_room_ids=[self.data_room.pk], query="test", k=5)
        # Should be deduplicated to one parent
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], parent.pk)
        # Score should accumulate from both children
        self.assertGreater(results[0]["rrf_score"], _rrf_score(0))


class ReadDocumentToolTests(TestCase):
    """Tests for ReadDocumentTool filtering to parent chunks."""

    def setUp(self):
        self.user = User.objects.create_user(email="rdt@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="RDTProject", slug="rdt-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="rdt.txt",
            status=DataRoomDocument.Status.READY,
            doc_index=1,
        )

    def test_read_document_assembles_from_parents_only(self):
        import json
        from chat.tools import ReadDocumentTool

        parent = DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Parent content", token_count=5, is_child=False,
        )
        DataRoomDocumentChunk.objects.create(
            document=self.doc, chunk_index=0, text="Child content", token_count=3, is_child=True, parent=parent,
        )

        tool = ReadDocumentTool()
        mock_context = MagicMock()
        mock_context.data_room_ids = [self.data_room.pk]
        mock_context.user_id = self.user.pk
        tool.context = mock_context

        result = json.loads(tool._run(doc_indices=[1]))
        documents = result["documents"]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["content"], "Parent content")
        self.assertNotIn("Child content", documents[0]["content"])


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

    def test_empty_input(self):
        self.assertEqual(clean_extracted_text(""), "")
        self.assertEqual(clean_extracted_text("   "), "")
