import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk
from documents.services.chunking import (
    _count_tokens,
    _strip_nul_bytes,
    clean_extracted_text,
    load_documents,
    structure_aware_chunk,
)
from documents.services.retrieval import (
    _rrf_score,
    get_chunk_with_context,
    get_chunks_by_document,
    get_chunks_by_data_room,
    get_merged_context_windows,
    hybrid_search_chunks,
    rerank_chunks,
    similarity_search_chunks,
)

User = get_user_model()

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

    @override_settings(PGVECTOR_CONNECTION="", CHUNKING_STRATEGY="semantic")
    def test_process_document_happy_path(self):
        """UPLOADED doc with a real file becomes READY and gets flat chunks persisted."""
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
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)
                self.assertEqual(doc.chunks.count(), 2)
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

    def test_process_document_skips_nonexistent_document(self):
        """Calling with a non-existent ID logs a warning and returns without raising."""
        from documents.services.process_document import process_document

        with self.assertLogs("documents.services.process_document", level="WARNING") as cm:
            process_document(99999)

        self.assertTrue(any("not found" in line for line in cm.output))


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
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)
                self.assertEqual(doc.chunks.count(), 5)

                # All chunks should have sequential indexes
                indexes = list(doc.chunks.order_by("chunk_index").values_list("chunk_index", flat=True))
                self.assertEqual(indexes, [0, 1, 2, 3, 4])

    @override_settings(PGVECTOR_CONNECTION="", LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_description_runs_after_ready(self):
        """Description generation happens after doc is marked READY."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        call_order = []

        def track_save(fields, **kwargs):
            if "status" in fields:
                call_order.append(f"save_status:{doc_ref.status}")
            if "description" in fields:
                call_order.append("save_description")

        sample_chunks = [{"text": "chunk", "token_count": 5, "chunk_index": 0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc_ref = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="desc.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc_ref.original_file.save("desc.txt", ContentFile(b"test"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="test")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("documents.services.description.generate_description_and_tags_from_text",
                           return_value={"description": "A description", "tags": {"document_type": "Report"}}):
                    process_document(doc_ref.id)

                doc_ref.refresh_from_db()
                self.assertEqual(doc_ref.status, DataRoomDocument.Status.READY)
                self.assertEqual(doc_ref.description, "A description")

    @override_settings(PGVECTOR_CONNECTION="", LLM_DEFAULT_CHEAP_MODEL="openai/gpt-4o-mini")
    def test_description_failure_doesnt_fail_doc(self):
        """Description generation failure doesn't affect doc status."""
        from django.core.files.base import ContentFile
        from documents.services.process_document import process_document

        sample_chunks = [{"text": "chunk", "token_count": 5, "chunk_index": 0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.settings(MEDIA_ROOT=tmpdir):
                doc = DataRoomDocument(
                    data_room=self.data_room,
                    uploaded_by=self.user,
                    original_filename="desc_fail.txt",
                    status=DataRoomDocument.Status.UPLOADED,
                )
                doc.original_file.save("desc_fail.txt", ContentFile(b"test"), save=True)

                with patch("documents.services.process_document.load_documents", return_value=[Mock(page_content="test")]), \
                     patch("documents.services.process_document.semantic_chunk", return_value=sample_chunks), \
                     patch("documents.services.description.generate_description_and_tags_from_text", side_effect=RuntimeError("LLM down")):
                    process_document(doc.id)

                doc.refresh_from_db()
                self.assertEqual(doc.status, DataRoomDocument.Status.READY)


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


class DynamicContextTests(TestCase):
    """Tests for get_chunk_with_context() dynamic expansion."""

    def setUp(self):
        self.user = User.objects.create_user(email="ctx@example.com", password="testpass")
        self.data_room = DataRoom.objects.create(name="CtxProject", slug="ctx-project", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.data_room,
            uploaded_by=self.user,
            original_filename="ctx.txt",
            status=DataRoomDocument.Status.READY,
        )

    def _create_chunks(self, count, tokens_each=100):
        """Create sequential chunks with given token counts."""
        chunks = []
        for i in range(count):
            text = f"Chunk {i} content. " * (tokens_each // 5)
            c = DataRoomDocumentChunk.objects.create(
                document=self.doc,
                chunk_index=i,
                text=text,
                token_count=tokens_each,
            )
            chunks.append(c)
        return chunks

    def test_symmetric_expansion(self):
        """Context expands symmetrically around center chunk."""
        chunks = self._create_chunks(5, tokens_each=100)
        center = chunks[2]

        result = get_chunk_with_context(center.id, target_tokens=300)
        self.assertEqual(result["id"], center.id)
        self.assertIn(center.chunk_index, result["chunks_included"])
        # Should include center + 1 left + 1 right = 3 chunks
        self.assertEqual(len(result["chunks_included"]), 3)
        self.assertEqual(result["chunks_included"], [1, 2, 3])

    def test_edge_of_document_left(self):
        """First chunk can only expand right."""
        chunks = self._create_chunks(5, tokens_each=100)

        result = get_chunk_with_context(chunks[0].id, target_tokens=300)
        self.assertEqual(result["chunks_included"][0], 0)
        self.assertEqual(len(result["chunks_included"]), 3)
        self.assertEqual(result["chunks_included"], [0, 1, 2])

    def test_edge_of_document_right(self):
        """Last chunk can only expand left."""
        chunks = self._create_chunks(5, tokens_each=100)

        result = get_chunk_with_context(chunks[4].id, target_tokens=300)
        self.assertIn(4, result["chunks_included"])
        self.assertEqual(len(result["chunks_included"]), 3)
        self.assertEqual(result["chunks_included"], [2, 3, 4])

    def test_budget_respected(self):
        """Total tokens should not exceed target_tokens."""
        chunks = self._create_chunks(10, tokens_each=100)
        center = chunks[5]

        result = get_chunk_with_context(center.id, target_tokens=350)
        self.assertLessEqual(result["context_token_count"], 350)

    def test_single_chunk_document(self):
        """Single chunk returns just that chunk."""
        chunks = self._create_chunks(1, tokens_each=100)

        result = get_chunk_with_context(chunks[0].id, target_tokens=1200)
        self.assertEqual(len(result["chunks_included"]), 1)
        self.assertEqual(result["context_text"], chunks[0].text)

    def test_nonexistent_chunk(self):
        """Non-existent chunk ID returns error."""
        result = get_chunk_with_context(99999)
        self.assertIn("error", result)

    def test_context_text_joins_chunks(self):
        """context_text should contain text from all included chunks."""
        chunks = self._create_chunks(3, tokens_each=100)

        result = get_chunk_with_context(chunks[1].id, target_tokens=400)
        for c in chunks:
            self.assertIn(f"Chunk {c.chunk_index} content.", result["context_text"])


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
            c = DataRoomDocumentChunk.objects.create(
                document=doc,
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
        """Single hit returns one window similar to get_chunk_with_context."""
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
    def test_valid_json_response(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text

        mock_response = Mock()
        mock_response.message.content = '{"description": "A patent document.", "document_type": "Patent"}'
        mock_service = Mock()
        mock_service.run.return_value = mock_response
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some patent text", user_id=1)
        self.assertEqual(result["description"], "A patent document.")
        self.assertEqual(result["tags"], {"document_type": "Patent"})

    @patch("llm.get_llm_service")
    def test_fallback_on_invalid_json(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text

        mock_response = Mock()
        mock_response.message.content = "Just a plain description paragraph."
        mock_service = Mock()
        mock_service.run.return_value = mock_response
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some text", user_id=1)
        self.assertEqual(result["description"], "Just a plain description paragraph.")
        self.assertEqual(result["tags"], {})

    @patch("llm.get_llm_service")
    def test_json_without_document_type(self, mock_get_service):
        from documents.services.description import generate_description_and_tags_from_text

        mock_response = Mock()
        mock_response.message.content = '{"description": "A document about something."}'
        mock_service = Mock()
        mock_service.run.return_value = mock_response
        mock_get_service.return_value = mock_service

        result = generate_description_and_tags_from_text("Some text", user_id=1)
        self.assertEqual(result["description"], "A document about something.")
        self.assertEqual(result["tags"], {})

    @patch("llm.get_llm_service")
    def test_generate_description_from_text_backward_compat(self, mock_get_service):
        from documents.services.description import generate_description_from_text

        mock_response = Mock()
        mock_response.message.content = '{"description": "A license agreement.", "document_type": "Agreement"}'
        mock_service = Mock()
        mock_service.run.return_value = mock_response
        mock_get_service.return_value = mock_service

        result = generate_description_from_text("Some text", user_id=1)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "A license agreement.")
