"""Tests for tools_to_langchain_schemas."""

from django.test import TestCase

from llm.tools import tools_to_langchain_schemas


class _MockTool:
    """Inline mock tool for schema tests."""

    name = "mock_tool"
    description = "A mock tool."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "A number"},
            "y": {"type": "number", "description": "Another number"},
        },
        "required": ["x", "y"],
    }


class ToolsToLangchainSchemasTests(TestCase):
    """Test tools_to_langchain_schemas output structure."""

    def test_single_tool_output_structure(self):
        tools = [_MockTool()]
        result = tools_to_langchain_schemas(tools)
        self.assertEqual(len(result), 1)
        schema = result[0]
        self.assertEqual(schema["type"], "function")
        self.assertIn("function", schema)
        fn = schema["function"]
        self.assertEqual(fn["name"], "mock_tool")
        self.assertIn("description", fn)
        self.assertIn("parameters", fn)
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertIn("x", fn["parameters"]["properties"])
        self.assertIn("y", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["x", "y"])

    def test_multiple_tools(self):
        tools = [_MockTool(), _MockTool()]
        result = tools_to_langchain_schemas(tools)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "mock_tool")
        self.assertEqual(result[1]["function"]["name"], "mock_tool")

    def test_empty_list_returns_empty(self):
        result = tools_to_langchain_schemas([])
        self.assertEqual(result, [])
