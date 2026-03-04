"""Tests for ToolRegistry."""

from django.test import TestCase

from llm.tools.registry import ToolRegistry, get_tool_registry
from llm.types.context import RunContext


class _MockTool:
    """Simple mock tool for testing the registry."""

    name = "mock_tool"
    description = "A mock tool for testing."
    parameters = {
        "type": "object",
        "properties": {"x": {"type": "number"}},
        "required": ["x"],
    }

    def run(self, args, context):
        return {"result": args.get("x", 0)}


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
        class EmptyNameTool:
            name = ""
            def run(self, args, context):
                return {}
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
