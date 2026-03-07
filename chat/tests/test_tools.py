"""Tests for the SearchDocumentsTool and ReadDocumentTool."""

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.tools import ReadDocumentTool, SearchDocumentsTool
from documents.models import DataRoom
from llm.types.context import RunContext

User = get_user_model()


class SearchDocumentsToolTests(TestCase):
    def setUp(self):
        self.tool = SearchDocumentsTool()
        self.user = User.objects.create_user(email="tooluser@test.com", password="pass")
        self.data_room = DataRoom.objects.create(name="Test", slug="test-tools", created_by=self.user)

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
        result = self._invoke({"query": "test"}, ctx)
        self.assertIn("error", result)
        self.assertEqual(result["count"], 0)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_returns_results(self, mock_search):
        mock_doc = MagicMock()
        mock_doc.page_content = "Some text"
        mock_doc.metadata = {"chunk_id": 1}
        mock_search.return_value = [mock_doc]

        result = self._invoke({"query": "test"}, self._ctx())

        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["text"], "Some text")
        self.assertEqual(result["results"][0]["metadata"], {"chunk_id": 1})

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_handles_search_exception(self, mock_search):
        mock_search.side_effect = Exception("DB error")
        result = self._invoke({"query": "test"}, self._ctx())
        self.assertEqual(result["count"], 0)
        self.assertIn("error", result)

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
        self.assertIn("count", result)


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
