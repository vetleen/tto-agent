"""Tests for the SearchDocumentsTool."""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from chat.tools import SearchDocumentsTool
from llm.types.context import RunContext


class SearchDocumentsToolTests(TestCase):
    def setUp(self):
        self.tool = SearchDocumentsTool()

    def test_has_required_attributes(self):
        self.assertEqual(self.tool.name, "search_documents")
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)
        self.assertIsInstance(self.tool.parameters, dict)
        self.assertEqual(self.tool.parameters["type"], "object")
        self.assertIn("query", self.tool.parameters["properties"])
        self.assertIn("query", self.tool.parameters["required"])

    def test_has_run_method(self):
        self.assertTrue(callable(getattr(self.tool, "run", None)))

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_calls_similarity_search_with_correct_args(self, mock_search):
        mock_search.return_value = []
        ctx = RunContext.create(user_id=1, conversation_id=42)
        self.tool.run({"query": "test query", "k": 3}, ctx)
        mock_search.assert_called_once_with(project_id=42, query="test query", k=3)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_default_k(self, mock_search):
        mock_search.return_value = []
        ctx = RunContext.create(user_id=1, conversation_id=42)
        self.tool.run({"query": "test"}, ctx)
        mock_search.assert_called_once_with(project_id=42, query="test", k=5)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_caps_k_at_10(self, mock_search):
        mock_search.return_value = []
        ctx = RunContext.create(user_id=1, conversation_id=42)
        self.tool.run({"query": "test", "k": 50}, ctx)
        mock_search.assert_called_once_with(project_id=42, query="test", k=10)

    def test_empty_query_raises(self):
        ctx = RunContext.create(user_id=1, conversation_id=42)
        with self.assertRaises(ValueError):
            self.tool.run({"query": ""}, ctx)

    def test_missing_conversation_id_raises(self):
        ctx = RunContext.create(user_id=1)
        with self.assertRaises(ValueError):
            self.tool.run({"query": "test"}, ctx)

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_returns_results(self, mock_search):
        mock_doc = MagicMock()
        mock_doc.page_content = "Some text"
        mock_doc.metadata = {"chunk_id": 1}
        mock_search.return_value = [mock_doc]

        ctx = RunContext.create(user_id=1, conversation_id=42)
        result = self.tool.run({"query": "test"}, ctx)

        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["text"], "Some text")
        self.assertEqual(result["results"][0]["metadata"], {"chunk_id": 1})

    @patch("documents.services.retrieval.similarity_search_chunks")
    def test_handles_search_exception(self, mock_search):
        mock_search.side_effect = Exception("DB error")
        ctx = RunContext.create(user_id=1, conversation_id=42)
        result = self.tool.run({"query": "test"}, ctx)
        self.assertEqual(result["count"], 0)
        self.assertIn("error", result)

    def test_registered_in_tool_registry(self):
        from llm.tools import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get_tool("search_documents")
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "search_documents")
