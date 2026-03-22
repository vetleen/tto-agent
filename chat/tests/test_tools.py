"""Tests for the SearchDocumentsTool and ReadDocumentTool."""

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from chat.tools import ReadDocumentTool, SearchDocumentsTool
from documents.models import DataRoom, DataRoomDocument, DataRoomDocumentChunk, DataRoomDocumentTag
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
        self.assertEqual(self.tool.name, "search_documents")
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
        DataRoomDocumentTag.objects.create(document=doc, key="document_type", value="Agreement")
        chunk = DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=0, text="Grant of license...", token_count=10,
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
    def test_includes_data_room_context(self, mock_windows, mock_search):
        """Should include data room name and description at the bottom."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="doc.pdf", status=DataRoomDocument.Status.READY, doc_index=1,
        )
        chunk = DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=0, text="Some text", token_count=5,
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
        tool = registry.get_tool("search_documents")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "search_documents")

    def test_denies_access_to_other_users_data_room(self):
        other_user = User.objects.create_user(email="other@test.com", password="pass")
        ctx = self._ctx(user_id=other_user.pk)
        with self.assertRaises(Exception):
            self._invoke({"query": "test"}, ctx)

    def test_denies_access_without_user_id(self):
        ctx = RunContext.create(data_room_ids=[self.data_room.pk])
        # Without a user_id the tool skips the ownership check and proceeds
        result = self._invoke({"query": "test"}, ctx)
        self.assertIsInstance(result, str)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_allows_access_to_shared_data_room_via_org(self, mock_search):
        """Users in the same org as the owner should access shared data rooms."""
        mock_search.return_value = []
        org = Organization.objects.create(name="Acme", slug="acme-tools-search")
        other_user = User.objects.create_user(email="colleague-s@test.com", password="pass")
        Membership.objects.create(user=self.user, org=org)
        Membership.objects.create(user=other_user, org=org)
        self.data_room.is_shared = True
        self.data_room.save(update_fields=["is_shared"])

        ctx = self._ctx(user_id=other_user.pk)
        result = self._invoke({"query": "test"}, ctx)
        self.assertIsInstance(result, str)
        mock_search.assert_called_once()

    def test_denies_shared_room_if_not_in_same_org(self):
        """Shared room should be denied if user is not in the owner's org."""
        org1 = Organization.objects.create(name="Org1", slug="org1-tools-search")
        org2 = Organization.objects.create(name="Org2", slug="org2-tools-search")
        other_user = User.objects.create_user(email="outsider-s@test.com", password="pass")
        Membership.objects.create(user=self.user, org=org1)
        Membership.objects.create(user=other_user, org=org2)
        self.data_room.is_shared = True
        self.data_room.save(update_fields=["is_shared"])

        ctx = self._ctx(user_id=other_user.pk)
        with self.assertRaises(Exception):
            self._invoke({"query": "test"}, ctx)


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
        tool = registry.get_tool("read_document")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "read_document")

    def test_full_document_read(self):
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="test.txt", status=DataRoomDocument.Status.READY,
        )
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=0, text="Chunk 0", token_count=2)
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=1, text="Chunk 1", token_count=2)
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=2, text="Chunk 2", token_count=2)

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
            DataRoomDocumentChunk.objects.create(
                document=doc, chunk_index=i, text=f"Chunk {i}", token_count=2,
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
        DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=0, text="text0", token_count=2, heading="Intro",
        )
        DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=1, text="text1", token_count=2, heading="Body",
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
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=0, text="A", token_count=1)
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=1, text="B", token_count=1)

        result = self._invoke({
            "doc_indices": [doc.doc_index],
            "chunk_start": 1,
        }, self._ctx())

        d = result["documents"][0]
        # Without chunk_end, should return full document
        self.assertIn("A", d["content"])
        self.assertIn("B", d["content"])
        self.assertNotIn("chunk_range", d)

    def test_allows_access_to_shared_data_room_via_org(self):
        """Users in the same org as the owner should access shared data rooms."""
        org = Organization.objects.create(name="Acme", slug="acme-tools-read")
        other_user = User.objects.create_user(email="colleague-r@test.com", password="pass")
        Membership.objects.create(user=self.user, org=org)
        Membership.objects.create(user=other_user, org=org)
        self.data_room.is_shared = True
        self.data_room.save(update_fields=["is_shared"])

        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="shared.txt", status=DataRoomDocument.Status.READY,
        )
        DataRoomDocumentChunk.objects.create(document=doc, chunk_index=0, text="Shared content", token_count=2)

        ctx = self._ctx(user_id=other_user.pk)
        result = self._invoke({"doc_indices": [doc.doc_index]}, ctx)
        self.assertEqual(len(result["documents"]), 1)
        self.assertIn("Shared content", result["documents"][0]["content"])

    def test_denies_shared_room_if_not_in_same_org(self):
        """Shared room should be denied if user is not in the owner's org."""
        org1 = Organization.objects.create(name="Org1", slug="org1-tools-read")
        org2 = Organization.objects.create(name="Org2", slug="org2-tools-read")
        other_user = User.objects.create_user(email="outsider-r@test.com", password="pass")
        Membership.objects.create(user=self.user, org=org1)
        Membership.objects.create(user=other_user, org=org2)
        self.data_room.is_shared = True
        self.data_room.save(update_fields=["is_shared"])

        ctx = self._ctx(user_id=other_user.pk)
        with self.assertRaises(Exception):
            self._invoke({"doc_indices": [1]}, ctx)

    def test_quarantined_chunks_excluded(self):
        """Quarantined chunks must not be returned by read_document."""
        doc = DataRoomDocument.objects.create(
            data_room=self.data_room, uploaded_by=self.user,
            original_filename="guarded.txt", status=DataRoomDocument.Status.READY,
        )
        DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=0, text="Safe content", token_count=2,
        )
        DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=1, text="Dangerous injected content",
            token_count=3, is_quarantined=True,
        )
        DataRoomDocumentChunk.objects.create(
            document=doc, chunk_index=2, text="More safe content", token_count=2,
        )

        result = self._invoke({"doc_indices": [doc.doc_index]}, self._ctx())
        d = result["documents"][0]
        self.assertIn("Safe content", d["content"])
        self.assertIn("More safe content", d["content"])
        self.assertNotIn("Dangerous injected content", d["content"])
        self.assertEqual(d["total_chunks"], 2)
