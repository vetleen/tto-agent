"""Tests for BaseTool schema generation (replaces old manual schema conversion tests)."""

from django.test import TestCase
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool
from llm.tools.schema import tools_to_langchain_schemas


class _MockInput(BaseModel):
    x: float = Field(description="A number")
    y: float = Field(description="Another number")


class _MockTool(ContextAwareTool):
    name: str = "mock_tool"
    description: str = "A mock tool."
    args_schema: type[BaseModel] = _MockInput

    def _run(self, x: float, y: float) -> str:
        return f'{{"result": {x + y}}}'


class ToolSchemaTests(TestCase):
    """Test that BaseTool generates correct schema for bind_tools()."""

    def test_tool_has_correct_name_and_description(self):
        tool = _MockTool()
        self.assertEqual(tool.name, "mock_tool")
        self.assertEqual(tool.description, "A mock tool.")

    def test_tool_args_schema_produces_json_schema(self):
        tool = _MockTool()
        schema = tool.args_schema.model_json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("x", schema["properties"])
        self.assertIn("y", schema["properties"])

    def test_tools_to_langchain_schemas_passes_through(self):
        tools = [_MockTool(), _MockTool()]
        result = tools_to_langchain_schemas(tools)
        self.assertEqual(len(result), 2)
        # Pass-through: same objects
        self.assertIs(result[0], tools[0])
        self.assertIs(result[1], tools[1])

    def test_empty_list_returns_empty(self):
        result = tools_to_langchain_schemas([])
        self.assertEqual(result, [])
