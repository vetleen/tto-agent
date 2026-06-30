"""Tools return a friendly validation-error observation instead of raising.

Regression for the recurring langchain ``_parse_input`` ValidationError class
(WILFRED-40 / WILFRED-6A): when a model calls a tool with arguments that fail the
``args_schema`` — a list arg passed as a JSON string, malformed JSON, a missing
required field — the tool must hand an instructive message back to the model (so
it can self-correct and retry) rather than raising and storming Sentry. Wired via
``handle_validation_error`` on ContextAwareTool.
"""
from __future__ import annotations

from typing import List

from django.test import TestCase
from pydantic import BaseModel

from llm.tools.interfaces import ContextAwareTool


class _ListInput(BaseModel):
    items: List[str]


class _EchoListTool(ContextAwareTool):
    name: str = "echo_list"
    description: str = "Echo a list of strings."
    args_schema: type = _ListInput

    def _run(self, items, **kwargs):
        return "ran:" + ",".join(items)


class ToolValidationErrorTests(TestCase):
    def test_valid_input_still_runs(self):
        self.assertEqual(_EchoListTool().invoke({"items": ["a", "b"]}), "ran:a,b")

    def test_list_passed_as_malformed_json_string_returns_message(self):
        # WILFRED-6A: the model emitted a list arg as a (broken) JSON string.
        out = _EchoListTool().invoke({"items": '["a", "b"}'})
        self.assertIsInstance(out, str)
        self.assertIn("invalid", out.lower())
        self.assertNotIn("ran:", out)  # the tool body must NOT have executed

    def test_missing_required_field_returns_message(self):
        out = _EchoListTool().invoke({})
        self.assertIsInstance(out, str)
        self.assertIn("invalid", out.lower())
        self.assertNotIn("ran:", out)
