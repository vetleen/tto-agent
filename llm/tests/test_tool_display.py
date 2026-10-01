"""Tests for llm.tools.display — the completion label shared by the live
``tool_end`` event and the reloaded chat history."""

from types import SimpleNamespace

from django.test import SimpleTestCase

from llm.tools.display import parse_tool_result, tool_end_display


class _Tool(SimpleNamespace):
    def end_label_for_result(self, result):
        return self.dynamic(result) if self.dynamic else None


def _tool(end_label="Did it", dynamic=None):
    return _Tool(end_label=end_label, dynamic=dynamic)


class ToolEndDisplayTests(SimpleTestCase):
    def test_static_end_label(self):
        self.assertEqual(tool_end_display(_tool(), '{"ok": 1}'), ("Did it", False))

    def test_dynamic_label_wins(self):
        tool = _tool(dynamic=lambda r: f"Found {len(r['results'])}")
        self.assertEqual(tool_end_display(tool, '{"results": [1, 2]}'), ("Found 2", False))

    def test_dynamic_none_falls_back_to_end_label(self):
        tool = _tool(dynamic=lambda r: None)
        self.assertEqual(tool_end_display(tool, '{"a": 1}')[0], "Did it")

    def test_non_json_result_uses_end_label_without_calling_dynamic(self):
        def boom(_):
            raise AssertionError("must not be called for non-JSON results")
        self.assertEqual(tool_end_display(_tool(dynamic=boom), "markdown text"), ("Did it", False))

    def test_unknown_tool_is_done(self):
        self.assertEqual(tool_end_display(None, '{"a": 1}'), ("Done", False))

    def test_error_status_flags_failure(self):
        self.assertEqual(
            tool_end_display(_tool(), '{"status": "error", "message": "x"}'),
            ("Did it", True),
        )
        self.assertTrue(tool_end_display(None, '{"status": "error"}')[1])

    def test_missing_result(self):
        self.assertEqual(tool_end_display(_tool(), ""), ("Did it", False))
        self.assertEqual(tool_end_display(_tool(), None), ("Did it", False))

    def test_parse_tool_result(self):
        self.assertEqual(parse_tool_result('{"a": 1}'), {"a": 1})
        self.assertIsNone(parse_tool_result("[1]"))
        self.assertIsNone(parse_tool_result(None))
