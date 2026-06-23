"""Tests for the SearchDocumentsTool and ReadDocumentTool."""

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase


from chat.tools import ReadDocumentTool, SearchDocumentsTool
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentTag


def _doc_version(doc):
    from documents.tests._helpers import make_version
    if doc.current_version_id:
        return doc.current_version
    return make_version(doc, status=doc.status, is_quarantined=doc.is_quarantined)


def _doc_chunk(doc, **kw):
    return DataRoomDocumentChunk.objects.create(version=_doc_version(doc), **kw)


def _doc_tag(doc, **kw):
    return DataRoomDocumentTag.objects.create(version=_doc_version(doc), **kw)
from llm.types.context import RunContext

User = get_user_model()


class SearchDocumentsToolTests(TestCase):
    def setUp(self):
        self.tool = SearchDocumentsTool()
        self.user = User.objects.create_user(email="tooluser@test.com", password="pass")
        self.data_room = DataRoom.objects.create(
            name="Test", slug="test-tools", created_by=self.user,
            description="A test data room with patents",
        )

    def _ctx(self, user_id=None, data_room_pks=None):
        return RunContext.create(
            user_id=user_id or self.user.pk,
            data_room_ids=data_room_pks or [self.data_room.pk],
        )

    def _invoke(self, args, ctx):
        """Set context and invoke the tool."""
        tool = self.tool.model_copy()
        tool.set_context(ctx)
        return tool.invoke(args)

    def _invoke_json(self, args, ctx):
        """Set context, invoke, and parse as JSON (for error cases)."""
        result = self._invoke(args, ctx)
        return json.loads(result)

    def test_has_required_attributes(self):
        self.assertEqual(self.tool.name, "document_search")
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)
        # args_schema should produce a valid JSON schema
        schema = self.tool.args_schema.model_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])

    def test_is_context_aware_tool(self):
        from llm.tools.interfaces import ContextAwareTool
        self.assertIsInstance(self.tool, ContextAwareTool)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_calls_similarity_search_with_correct_args(self, mock_search):
        mock_search.return_value = []
        self._invoke({"query": "test query", "k": 3}, self._ctx())
        mock_search.assert_called_once_with(data_room_ids=[self.data_room.pk], query="test query", k=3)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_default_k(self, mock_search):
        mock_search.return_value = []
        self._invoke({"query": "test"}, self._ctx())
        mock_search.assert_called_once_with(data_room_ids=[self.data_room.pk], query="test", k=5)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_caps_k_at_10(self, mock_search):
        mock_search.return_value = []
        self._invoke({"query": "test", "k": 50}, self._ctx())
        mock_search.assert_called_once_with(data_room_ids=[self.data_room.pk], query="test", k=10)

    def test_empty_query_raises(self):
        with self.assertRaises(Exception):
            self._invoke({"query": ""}, self._ctx())

    def test_empty_data_room_ids_returns_error(self):
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[])
        result = self._invoke_json({"query": "test"}, ctx)
        self.assertIn("error", result)
        self.assertEqual(result["count"], 0)

    @patch("documents.services.retrieval.similarity_search_chunks")
    @patch("documents.services.retrieval.get_merged_context_windows")
    def test_returns_formatted_text(self, mock_windows, mock_search):
        """Search results should be formatted as human-readable text, not JSON."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="License.pdf", description="A license agreement",
            status=DataRoomDocument.Status.READY, doc_index=1,
        )
        _doc_tag(doc, key="document_type", value="Agreement")
        chunk = _doc_chunk(doc, chunk_index=0, text="Grant of license...", token_count=10,
            heading="Grant of License",
        )

        mock_doc = MagicMock()
        mock_doc.page_content = "Grant of license..."
        mock_doc.metadata = {"chunk_id": chunk.id, "doc_index": 1, "data_room_id": self.data_room.pk, "chunk_index": 0}
        mock_search.return_value = [mock_doc]

        mock_windows.return_value = [{
            "chunk_ids": [chunk.id],
            "document_id": doc.pk,
            "context_text": "Grant of license...",
            "context_token_count": 10,
            "chunks_included": [0],
        }]

        result = self._invoke({"query": "license"}, self._ctx())

        # Should be formatted text, not JSON
        self.assertIn("# Search Results", result)
        self.assertIn("License.pdf", result)
        self.assertIn("Agreement", result)
        self.assertIn("Grant of License", result)
        self.assertIn("A license agreement", result)
        self.assertIn("Grant of license...", result)

    @patch("documents.services.retrieval.similarity_search_chunks")
    @patch("documents.services.retrieval.get_merged_context_windows")
    def test_image_document_surfaces_embed_token(self, mock_windows, mock_search):
        """An image-as-document in search results carries a [[image:…]] token."""
        from documents.tests._helpers import make_version

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            description="A bar chart.", status=DataRoomDocument.Status.READY, doc_index=1,
        )
        v = make_version(doc, status=DataRoomDocument.Status.READY)
        v.parser_type = "image"
        v.save(update_fields=["parser_type"])
        chunk = DataRoomDocumentChunk.objects.create(
            version=v, chunk_index=0, text="A bar chart.", token_count=4,
        )

        mock_doc = MagicMock()
        mock_doc.metadata = {"chunk_id": chunk.id}
        mock_search.return_value = [mock_doc]
        mock_windows.return_value = [{
            "chunk_ids": [chunk.id], "document_id": doc.pk,
            "context_text": "A bar chart.", "context_token_count": 4, "chunks_included": [0],
        }]

        result = self._invoke({"query": "chart"}, self._ctx())
        self.assertIn("[[image:", result)
        self.assertIn("This document is an image", result)

    @patch("documents.services.retrieval.similarity_search_chunks")
    @patch("documents.services.retrieval.get_merged_context_windows")
    def test_includes_data_room_context(self, mock_windows, mock_search):
        """Should include data room name and description at the bottom."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="doc.pdf", status=DataRoomDocument.Status.READY, doc_index=1,
        )
        chunk = _doc_chunk(doc, chunk_index=0, text="Some text", token_count=5,
        )

        mock_doc = MagicMock()
        mock_doc.metadata = {"chunk_id": chunk.id}
        mock_search.return_value = [mock_doc]
        mock_windows.return_value = [{
            "chunk_ids": [chunk.id],
            "document_id": doc.pk,
            "context_text": "Some text",
            "context_token_count": 5,
            "chunks_included": [0],
        }]

        result = self._invoke({"query": "test"}, self._ctx())
        self.assertIn("# Data Room Context", result)
        self.assertIn("Test", result)
        self.assertIn("A test data room with patents", result)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_handles_search_exception(self, mock_search):
        mock_search.side_effect = Exception("DB error")
        result = self._invoke_json({"query": "test"}, self._ctx())
        self.assertEqual(result["count"], 0)
        self.assertIn("error", result)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_no_results(self, mock_search):
        mock_search.return_value = []
        result = self._invoke({"query": "nothing"}, self._ctx())
        self.assertIn("No results found", result)

    def test_registered_in_tool_registry(self):
        from llm.tools import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get_tool("document_search")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "document_search")

    def test_denies_access_to_other_users_data_room(self):
        other_user = User.objects.create_user(email="other@test.com", password="pass")
        ctx = self._ctx(user_id=other_user.pk)
        with self.assertRaises(Exception):
            self._invoke({"query": "test"}, ctx)

    def test_denies_access_without_user_id(self):
        ctx = RunContext.create(data_room_ids=[self.data_room.pk])
        # Fail closed: without a user_id the ownership check cannot pass, so access
        # is denied (the tool raises rather than leaking another tenant's rooms).
        with self.assertRaises(Exception):
            self._invoke({"query": "test"}, ctx)

    @patch("documents.services.retrieval.similarity_search_chunks")
    @patch("documents.services.retrieval.get_merged_context_windows")
    def test_records_chunk_usage(self, mock_windows, mock_search):
        """Search should create ThreadChunkUsage records for retrieved chunks."""
        from chat.models import ChatThread, ThreadChunkUsage

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="usage.pdf", status=DataRoomDocument.Status.READY, doc_index=1,
        )
        c1 = _doc_chunk(doc, chunk_index=0, text="A", token_count=1)
        c2 = _doc_chunk(doc, chunk_index=1, text="B", token_count=1)

        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"chunk_id": c1.id, "doc_index": 1, "data_room_id": self.data_room.pk, "chunk_index": 0}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"chunk_id": c2.id, "doc_index": 1, "data_room_id": self.data_room.pk, "chunk_index": 1}
        mock_search.return_value = [mock_doc1, mock_doc2]
        mock_windows.return_value = [{
            "chunk_ids": [c1.id, c2.id], "document_id": doc.pk,
            "context_text": "A\n\nB", "context_token_count": 2, "chunks_included": [0, 1],
        }]

        thread = ChatThread.objects.create(created_by=self.user, title="Usage test")
        ctx = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
            data_room_ids=[self.data_room.pk],
        )
        self._invoke({"query": "test"}, ctx)

        usages = ThreadChunkUsage.objects.filter(thread=thread)
        self.assertEqual(usages.count(), 2)
        self.assertEqual(set(usages.values_list("chunk_id", flat=True)), {c1.id, c2.id})
        self.assertTrue(all(u.document_id == doc.pk for u in usages))

    @patch("documents.services.retrieval.similarity_search_chunks")
    @patch("documents.services.retrieval.get_merged_context_windows")
    def test_duplicate_search_no_duplicate_usage(self, mock_windows, mock_search):
        """Searching same chunks twice should not create duplicate usage rows."""
        from chat.models import ChatThread, ThreadChunkUsage

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="dup.pdf", status=DataRoomDocument.Status.READY, doc_index=1,
        )
        chunk = _doc_chunk(doc, chunk_index=0, text="X", token_count=1)

        mock_doc = MagicMock()
        mock_doc.metadata = {"chunk_id": chunk.id, "doc_index": 1, "data_room_id": self.data_room.pk, "chunk_index": 0}
        mock_search.return_value = [mock_doc]
        mock_windows.return_value = [{
            "chunk_ids": [chunk.id], "document_id": doc.pk,
            "context_text": "X", "context_token_count": 1, "chunks_included": [0],
        }]

        thread = ChatThread.objects.create(created_by=self.user, title="Dup test")
        ctx = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
            data_room_ids=[self.data_room.pk],
        )
        self._invoke({"query": "test"}, ctx)
        self._invoke({"query": "test again"}, ctx)

        self.assertEqual(ThreadChunkUsage.objects.filter(thread=thread).count(), 1)

class ReadDocumentToolTests(TestCase):
    def setUp(self):
        self.tool = ReadDocumentTool()
        self.user = User.objects.create_user(email="readuser@test.com", password="pass")
        self.data_room = DataRoom.objects.create(name="Read", slug="read-tools", created_by=self.user)

    def _ctx(self, user_id=None, data_room_pks=None):
        return RunContext.create(
            user_id=user_id or self.user.pk,
            data_room_ids=data_room_pks or [self.data_room.pk],
        )

    def _invoke(self, args, ctx):
        """Set context and invoke the tool."""
        tool = self.tool.model_copy()
        tool.set_context(ctx)
        result = tool.invoke(args)
        return json.loads(result)

    def test_denies_access_to_other_users_data_room(self):
        other_user = User.objects.create_user(email="other2@test.com", password="pass")
        ctx = self._ctx(user_id=other_user.pk)
        with self.assertRaises(Exception):
            self._invoke({"doc_indices": [1]}, ctx)

    def test_empty_data_room_ids_returns_error(self):
        ctx = RunContext.create(user_id=self.user.pk, data_room_ids=[])
        result = self._invoke({"doc_indices": [1]}, ctx)
        self.assertIn("error", result)

    def test_returns_not_found_for_missing_doc_index(self):
        result = self._invoke({"doc_indices": [999]}, self._ctx())
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("error", result["documents"][0])

    def test_registered_in_tool_registry(self):
        from llm.tools import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get_tool("document_read")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "document_read")

    def test_quarantined_document_returns_error(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="quar.txt", status=DataRoomDocument.Status.READY,
            is_quarantined=True,
            quarantine_reason="Contains GDPR Article 9 (special category) personal data.",
        )
        _doc_chunk(doc, chunk_index=0, text="Sensitive", token_count=2)

        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        self.assertEqual(len(result["documents"]), 1)
        d = result["documents"][0]
        self.assertIn("error", d)
        self.assertIn("quarantined", d["error"].lower())
        self.assertNotIn("content", d)

    def test_image_document_includes_embed_token(self):
        from documents.tests._helpers import make_version

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            description="A bar chart.", status=DataRoomDocument.Status.READY,
        )
        v = make_version(doc, status=DataRoomDocument.Status.READY, chunks=["A bar chart."])
        v.parser_type = "image"
        v.save(update_fields=["parser_type"])

        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        entry = result["documents"][0]
        self.assertIn("image", entry)
        self.assertTrue(entry["image"].startswith("[[image:"))
        # The content is flagged as a vision description, with the real chunk
        # text still present, so the model doesn't re-describe the description.
        self.assertIn("This document is an image", entry["content"])
        self.assertIn("AI-generated description", entry["content"])
        self.assertIn("A bar chart.", entry["content"])

    def test_text_document_has_no_image_note(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="notes.txt", status=DataRoomDocument.Status.READY,
        )
        _doc_chunk(doc, chunk_index=0, text="Plain text.", token_count=2)
        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        entry = result["documents"][0]
        self.assertNotIn("This document is an image", entry["content"])
        self.assertNotIn("image", entry)

    def test_scanning_document_returns_not_found(self):
        """A document still awaiting its PII scan must not be readable (and the
        error must not leak the scan state)."""
        for status in (DataRoomDocument.Status.SCANNING, DataRoomDocument.Status.SCAN_FAILED):
            doc = DataRoomDocument.objects.create(
                data_room=self.data_room, uploaded_by=self.user,
                original_filename=f"{status}.txt", status=status,
            )
            _doc_chunk(doc, chunk_index=0, text="Unscanned", token_count=2)

            result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
            d = result["documents"][0]
            self.assertIn("error", d)
            self.assertIn("No document with index", d["error"])
            self.assertNotIn("content", d)

    def test_full_document_read(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="test.txt", status=DataRoomDocument.Status.READY,
        )
        _doc_chunk(doc, chunk_index=0, text="Chunk 0", token_count=2)
        _doc_chunk(doc, chunk_index=1, text="Chunk 1", token_count=2)
        _doc_chunk(doc, chunk_index=2, text="Chunk 2", token_count=2)

        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        self.assertEqual(len(result["documents"]), 1)
        d = result["documents"][0]
        self.assertIn("Chunk 0", d["content"])
        self.assertIn("Chunk 1", d["content"])
        self.assertIn("Chunk 2", d["content"])
        self.assertEqual(d["total_chunks"], 3)
        self.assertNotIn("chunk_range", d)

    def test_chunk_range_read(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="range.txt", status=DataRoomDocument.Status.READY,
        )
        for i in range(5):
            _doc_chunk(doc, chunk_index=i, text=f"Chunk {i}", token_count=2,
                heading=f"Section {i}" if i % 2 == 0 else None,
            )

        result = self._invoke({
            "doc_indices": [doc.doc_index],
            "chunk_start": 1,
            "chunk_end": 3,
        }, self._ctx())

        self.assertEqual(len(result["documents"]), 1)
        d = result["documents"][0]
        self.assertIn("Chunk 1", d["content"])
        self.assertIn("Chunk 2", d["content"])
        self.assertIn("Chunk 3", d["content"])
        self.assertNotIn("Chunk 0", d["content"])
        self.assertNotIn("Chunk 4", d["content"])
        self.assertEqual(d["chunk_range"], "1-3")
        self.assertEqual(d["chunks_returned"], 3)
        self.assertEqual(d["total_chunks"], 5)

    def test_chunk_range_with_headings(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="headings.txt", status=DataRoomDocument.Status.READY,
        )
        _doc_chunk(doc, chunk_index=0, text="text0", token_count=2, heading="Intro",
        )
        _doc_chunk(doc, chunk_index=1, text="text1", token_count=2, heading="Body",
        )

        result = self._invoke({
            "doc_indices": [doc.doc_index],
            "chunk_start": 0,
            "chunk_end": 1,
        }, self._ctx())

        d = result["documents"][0]
        self.assertIn("Intro", d["headings"])
        self.assertIn("Body", d["headings"])

    def test_chunk_range_only_start_ignored(self):
        """When only chunk_start is provided without chunk_end, full doc is returned."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="partial.txt", status=DataRoomDocument.Status.READY,
        )
        _doc_chunk(doc, chunk_index=0, text="A", token_count=1)
        _doc_chunk(doc, chunk_index=1, text="B", token_count=1)

        result = self._invoke({
            "doc_indices": [doc.doc_index],
            "chunk_start": 1,
        }, self._ctx())

        d = result["documents"][0]
        # Without chunk_end, should return full document
        self.assertIn("A", d["content"])
        self.assertIn("B", d["content"])
        self.assertNotIn("chunk_range", d)

    def test_quarantined_chunks_excluded(self):
        """Quarantined chunks must not be returned by document_read."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="guarded.txt", status=DataRoomDocument.Status.READY,
        )
        _doc_chunk(doc, chunk_index=0, text="Safe content", token_count=2,
        )
        _doc_chunk(doc, chunk_index=1, text="Dangerous injected content",
            token_count=3, is_quarantined=True,
        )
        _doc_chunk(doc, chunk_index=2, text="More safe content", token_count=2,
        )

        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        d = result["documents"][0]
        self.assertIn("Safe content", d["content"])
        self.assertIn("More safe content", d["content"])
        self.assertNotIn("Dangerous injected content", d["content"])
        self.assertEqual(d["total_chunks"], 2)

    def test_records_chunk_usage(self):
        """Reading a document should create ThreadChunkUsage records."""
        from chat.models import ChatThread, ThreadChunkUsage

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="read-usage.txt", status=DataRoomDocument.Status.READY,
        )
        c1 = _doc_chunk(doc, chunk_index=0, text="AA", token_count=1)
        c2 = _doc_chunk(doc, chunk_index=1, text="BB", token_count=1)

        thread = ChatThread.objects.create(created_by=self.user, title="Read usage test")
        ctx = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(thread.id),
            data_room_ids=[self.data_room.pk],
        )
        self._invoke({"doc_indices": [doc.doc_index]}, ctx)

        usages = ThreadChunkUsage.objects.filter(thread=thread)
        self.assertEqual(usages.count(), 2)
        self.assertEqual(set(usages.values_list("chunk_id", flat=True)), {c1.id, c2.id})


class CanvasSaveToDocumentToolTests(TestCase):
    """Tests for CanvasSaveToDocumentTool (mode='new') — saving a canvas as a .md document."""

    def setUp(self):
        from django.utils import timezone

        from chat.models import ChatCanvas, ChatThread
        from chat.tools import CanvasSaveToDocumentTool

        self.tool = CanvasSaveToDocumentTool()
        self.user = User.objects.create_user(email="canvassaver@test.com", password="pass")
        self.data_room = DataRoom.objects.create(
            name="Owner Room", slug="save-canvas", created_by=self.user,
        )
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Memo", content="# Hello\n\nBody text.",
            is_active=True, last_activated_at=timezone.now(),
        )

    def _ctx(self, user_id=None, data_room_pks=None):
        return RunContext.create(
            user_id=user_id or self.user.pk,
            conversation_id=str(self.thread.id),
            data_room_ids=data_room_pks if data_room_pks is not None else [self.data_room.pk],
        )

    def _invoke_json(self, args, ctx):
        tool = self.tool.model_copy()
        tool.set_context(ctx)
        return json.loads(tool.invoke(args))

    def test_has_required_attributes(self):
        self.assertEqual(self.tool.name, "canvas_save_to_document")
        schema = self.tool.args_schema.model_json_schema()
        self.assertIn("mode", schema["properties"])
        self.assertIn("canvas_name", schema["properties"])
        self.assertIn("data_room_name", schema["properties"])

    def test_invalid_mode_returns_error(self):
        result = self._invoke_json({"mode": "sideways"}, self._ctx())
        self.assertIn("error", result)
        self.assertFalse(DataRoomDocument.objects.exists())

    def test_saves_active_canvas_as_md_document(self):
        result = self._invoke_json({"mode": "new"}, self._ctx())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["verdict"], "clean")
        self.assertEqual(result["filename"], "Memo.md")
        self.assertEqual(result["data_room_name"], "Owner Room")

        doc = DataRoomDocument.objects.get(data_room=self.data_room)
        self.assertEqual(doc.original_filename, "Memo.md")
        self.assertEqual(doc.mime_type, "text/markdown")
        # Sync scan-at-save processes v0 inline and releases it READY on a clean scan
        # (no LLM models configured in the test env → the scan no-ops and clears).
        self.assertEqual(doc.status, DataRoomDocument.Status.READY)
        self.assertTrue(
            DataRoomDocumentTag.objects.filter(
                version__document=doc, key="source", value="canvas_export"
            ).exists()
        )

    def test_no_data_room_returns_error(self):
        result = self._invoke_json({"mode": "new"}, self._ctx(data_room_pks=[]))
        self.assertIn("error", result)
        self.assertFalse(DataRoomDocument.objects.exists())

    def test_blank_canvas_returns_error(self):
        self.canvas.content = "   "
        self.canvas.save(update_fields=["content"])
        result = self._invoke_json({"mode": "new"}, self._ctx())
        self.assertIn("error", result)
        self.assertIn("empty", result["error"].lower())
        self.assertFalse(DataRoomDocument.objects.exists())

    def test_unknown_canvas_name_returns_error(self):
        result = self._invoke_json({"mode": "new", "canvas_name": "Nope"}, self._ctx())
        self.assertIn("error", result)
        self.assertFalse(DataRoomDocument.objects.exists())

    def test_multiple_rooms_without_name_returns_error(self):
        second = DataRoom.objects.create(
            name="Second Room", slug="save-canvas-2", created_by=self.user,
        )
        result = self._invoke_json(
            {"mode": "new"}, self._ctx(data_room_pks=[self.data_room.pk, second.pk])
        )
        self.assertIn("error", result)
        self.assertIn("Second Room", result["error"])
        self.assertFalse(DataRoomDocument.objects.exists())

    @patch("documents.tasks.process_document_task.delay")
    def test_multiple_rooms_with_name_saves_to_named_room(self, mock_delay):
        second = DataRoom.objects.create(
            name="Second Room", slug="save-canvas-2", created_by=self.user,
        )
        result = self._invoke_json(
            {"mode": "new", "data_room_name": "Second Room"},
            self._ctx(data_room_pks=[self.data_room.pk, second.pk]),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(DataRoomDocument.objects.filter(data_room=second).exists())
        self.assertFalse(DataRoomDocument.objects.filter(data_room=self.data_room).exists())

    def test_access_denied_for_unowned_room(self):
        other = User.objects.create_user(email="intruder@test.com", password="pass")
        other_room = DataRoom.objects.create(
            name="Other Room", slug="other-room", created_by=other,
        )
        result = self._invoke_json({"mode": "new"}, self._ctx(data_room_pks=[other_room.pk]))
        self.assertIn("error", result)
        self.assertFalse(DataRoomDocument.objects.exists())


class ImageTokenSurfacingTests(TestCase):
    """document_list surfaces an embed token for image docs; the token helper
    is idempotent (one reference asset per version)."""

    def setUp(self):
        from documents.tests._helpers import make_version

        self.user = User.objects.create_user(email="imgtok@test.com", password="pw")
        self.room = DataRoom.objects.create(name="R", slug="r-imgtok", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="chart.png", mime_type="image/png",
            description="A bar chart.", status=DataRoomDocument.Status.READY,
        )
        self.version = make_version(
            self.doc, status=DataRoomDocument.Status.READY, chunks=["A bar chart."],
        )
        self.version.parser_type = "image"
        self.version.save(update_fields=["parser_type"])
        self.doc.refresh_from_db()

    def _ctx(self):
        return RunContext.create(
            user_id=self.user.pk, conversation_id="t", data_room_ids=[self.room.pk],
        )

    def test_document_list_includes_image_token(self):
        from chat.tools import ListDocumentsTool

        tool = ListDocumentsTool()
        tool.set_context(self._ctx())
        out = json.loads(tool.invoke({}))
        row = next(d for d in out["documents"] if d["doc_index"] == self.doc.doc_index)
        self.assertIn("image", row)
        self.assertTrue(row["image"].startswith("[[image:"))

    def test_text_document_has_no_image_token(self):
        from chat.tools import ListDocumentsTool

        text_doc = DataRoomDocument.objects.create(
            data_room=self.room, uploaded_by=self.user,
            original_filename="notes.txt", mime_type="text/plain",
            status=DataRoomDocument.Status.READY,
        )
        from documents.tests._helpers import make_version

        v = make_version(text_doc, status=DataRoomDocument.Status.READY, chunks=["hi"])
        v.parser_type = "text"
        v.save(update_fields=["parser_type"])

        tool = ListDocumentsTool()
        tool.set_context(self._ctx())
        out = json.loads(tool.invoke({}))
        row = next(d for d in out["documents"] if d["doc_index"] == text_doc.doc_index)
        self.assertNotIn("image", row)

    def test_token_helper_is_idempotent(self):
        from chat.assets import get_or_create_version_image_token
        from chat.models import Asset

        t1 = get_or_create_version_image_token(
            version_id=self.version.id, mime="image/png", description="A bar chart.",
        )
        t2 = get_or_create_version_image_token(
            version_id=self.version.id, mime="image/png", description="A bar chart.",
        )
        self.assertEqual(t1, t2)
        self.assertTrue(t1.startswith("[[image:"))
        # The caption is left empty — the model authors it if it wants one.
        self.assertTrue(t1.endswith("|]]"))
        self.assertEqual(
            Asset.objects.filter(version=self.version, blob="").count(), 1
        )
