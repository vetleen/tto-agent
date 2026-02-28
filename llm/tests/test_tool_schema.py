"""Tests for tools_to_langchain_schemas."""

from django.test import TestCase

from llm.tools import tools_to_langchain_schemas
from llm.tools.builtins import AddNumberTool


class ToolsToLangchainSchemasTests(TestCase):
    """Test tools_to_langchain_schemas output structure."""

    def test_single_tool_output_structure(self):
        tools = [AddNumberTool()]
        result = tools_to_langchain_schemas(tools)
        self.assertEqual(len(result), 1)
        schema = result[0]
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        fn = schema["function"]
        self.assertEqual(fn["name"], "add_number")
        self.assertIn("description", fn)
        self.assertIn("parameters", fn)
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertIn("a", fn["parameters"]["properties"])
        self.assertIn("b", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["a", "b"])

    def test_multiple_tools(self):
        tools = [AddNumberTool(), AddNumberTool()]
        result = tools_to_langchain_schemas(tools)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "add_number")
        self.assertEqual(result[1]["function"]["name"], "add_number")

    def test_empty_list_returns_empty(self):
        result = tools_to_langchain_schemas([])
        self.assertEqual(result, [])
