import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings

from documents.services.chunking import (
    MIN_CHUNK_TOKENS,
    _count_tokens,
    _looks_like_markdown,
    _merge_small_chunks,
    chunk_text,
    load_documents,
)

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
    def test_chunk_text_multiple_chunks_when_over_max_tokens(self):
        """Documents over max_chunk_tokens are split, then small chunks merged (min 200)."""
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

        similarity_search(project_id=1, query="hello", k=500)

        self.assertEqual(store.similarity_search.call_args.kwargs["k"], 50)
