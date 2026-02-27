"""Tests for tools: ToolRegistry and AddNumberTool."""

from django.test import TestCase

from llm.tools.builtins import AddNumberTool
from llm.tools.registry import ToolRegistry, get_tool_registry
from llm.types.context import RunContext


class ToolRegistryTests(TestCase):
    """Test ToolRegistry register, get_tool, list_tools."""

    def setUp(self):
        super().setUp()
        self.registry = ToolRegistry()

    def test_register_tool_and_get_tool(self):
        tool = AddNumberTool()
        self.registry.register_tool(tool)
        self.assertIs(self.registry.get_tool("add_number"), tool)
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
        tool = AddNumberTool()
        self.registry.register_tool(tool)
        listed = self.registry.list_tools()
        self.assertEqual(listed, {"add_number": tool})
        listed["add_number"] = None
        self.assertIs(self.registry.get_tool("add_number"), tool)

    def test_get_tool_registry_returns_singleton(self):
        a = get_tool_registry()
        b = get_tool_registry()
        self.assertIs(a, b)


class AddNumberToolTests(TestCase):
    """Test AddNumberTool with valid and invalid inputs."""

    def setUp(self):
        super().setUp()
        self.tool = AddNumberTool()
        self.context = RunContext.create()

    def test_run_valid_ints(self):
        result = self.tool.run({"a": 2, "b": 3}, self.context)
        self.assertEqual(result, {"result": 5})

    def test_run_valid_floats(self):
        result = self.tool.run({"a": 1.5, "b": 2.5}, self.context)
        self.assertEqual(result, {"result": 4.0})

    def test_run_string_numbers_parsed(self):
        result = self.tool.run({"a": "10", "b": "20"}, self.context)
        self.assertEqual(result, {"result": 30.0})

    def test_run_missing_a_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.tool.run({"b": 1}, self.context)
        self.assertIn("a", str(ctx.exception).lower())
        self.assertIn("b", str(ctx.exception).lower())

    def test_run_missing_b_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.tool.run({"a": 1}, self.context)
        self.assertIn("requires", str(ctx.exception))

    def test_run_non_numeric_a_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.tool.run({"a": "not a number", "b": 1}, self.context)
        self.assertIn("numeric", str(ctx.exception).lower())

    def test_run_non_numeric_b_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.tool.run({"a": 1, "b": []}, self.context)
        self.assertIn("numeric", str(ctx.exception).lower())
