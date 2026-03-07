"""Tests for ToolRegistry and ContextAwareTool."""

from django.test import TestCase

from llm.tools.interfaces import ContextAwareTool
from llm.tools.registry import ToolRegistry, get_tool_registry
from llm.types.context import RunContext
from pydantic import BaseModel, Field


class _MockInput(BaseModel):
    x: float = Field(description="A number")


class _MockTool(ContextAwareTool):
    """Simple mock tool for testing the registry."""

    name: str = "mock_tool"
    description: str = "A mock tool for testing."
    args_schema: type[BaseModel] = _MockInput

    def _run(self, x: float) -> str:
        return f'{{"result": {x}}}'


class ContextAwareToolTests(TestCase):
    """Test ContextAwareTool base class."""

    def test_set_context_returns_self(self):
        tool = _MockTool()
        ctx = RunContext.create(user_id=1)
        result = tool.set_context(ctx)
        self.assertIs(result, tool)
        self.assertEqual(tool.context.user_id, "1")

    def test_context_defaults_to_none(self):
        tool = _MockTool()
        self.assertIsNone(tool.context)

    def test_invoke_calls_run(self):
        tool = _MockTool()
        result = tool.invoke({"x": 42})
        self.assertIn("42", result)

    def test_model_copy_preserves_fields(self):
        tool = _MockTool()
        ctx = RunContext.create(user_id=1)
        tool.set_context(ctx)
        copy = tool.model_copy()
        self.assertEqual(copy.name, "mock_tool")
        # Context is copied by reference
        self.assertIsNotNone(copy.context)


class ToolRegistryTests(TestCase):
    """Test ToolRegistry register, get_tool, list_tools."""

    def setUp(self):
        super().setUp()
        self.registry = ToolRegistry()

    def test_register_tool_and_get_tool(self):
        tool = _MockTool()
        self.registry.register_tool(tool)
        self.assertIs(self.registry.get_tool("mock_tool"), tool)
        self.assertIsNone(self.registry.get_tool("nonexistent"))

    def test_register_tool_empty_name_raises(self):
        class EmptyNameTool(ContextAwareTool):
            name: str = ""
            description: str = "empty"
            def _run(self) -> str:
                return "{}"
        with self.assertRaises(ValueError) as ctx:
            self.registry.register_tool(EmptyNameTool())
        self.assertIn("non-empty", str(ctx.exception))

    def test_list_tools_returns_copy(self):
        tool = _MockTool()
        self.registry.register_tool(tool)
        listed = self.registry.list_tools()
        self.assertEqual(listed, {"mock_tool": tool})
        listed["mock_tool"] = None
        self.assertIs(self.registry.get_tool("mock_tool"), tool)

    def test_clear_empties_all_tools(self):
        tool = _MockTool()
        self.registry.register_tool(tool)
        self.registry.clear()
        self.assertIsNone(self.registry.get_tool("mock_tool"))
        self.assertEqual(self.registry.list_tools(), {})

    def test_get_tool_registry_returns_singleton(self):
        a = get_tool_registry()
        b = get_tool_registry()
        self.assertIs(a, b)
