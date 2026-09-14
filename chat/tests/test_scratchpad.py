"""Tests for the agent-only scratchpad: append tool, cap, and prompt injection.

The scratchpad is the assistant's private working memory. It is append-only,
capped, injected into every turn's dynamic context, and NEVER surfaced to the
user (no ``canvas.updated``-style event, not in ``CANVAS_UPDATED_TOOLS``).
"""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from chat.models import ChatThread
from chat.prompts import build_dynamic_context
from chat.scratchpad_tools import ScratchpadAppendTool
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(args, ctx):
    tool = ScratchpadAppendTool()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


class ScratchpadAppendToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="pad@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_appends_note_to_empty_scratchpad(self):
        result = _invoke({"note": "First finding: X = 42"}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "ok")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.scratchpad, "First finding: X = 42")

    def test_appends_are_concatenated(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke({"note": "one"}, ctx)
        _invoke({"note": "two"}, ctx)
        self.thread.refresh_from_db()
        self.assertIn("one", self.thread.scratchpad)
        self.assertIn("two", self.thread.scratchpad)
        # Order preserved, "one" before "two".
        self.assertLess(
            self.thread.scratchpad.index("one"),
            self.thread.scratchpad.index("two"),
        )

    def test_empty_note_is_rejected(self):
        result = _invoke({"note": "   "}, _ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.scratchpad, "")

    def test_no_context_returns_error(self):
        tool = ScratchpadAppendTool()
        result = json.loads(tool.invoke({"note": "hi"}))
        self.assertEqual(result["status"], "error")

    @override_settings(SCRATCHPAD_MAX_CHARS=50)
    def test_cap_tail_truncates_keeping_recent(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke({"note": "A" * 40}, ctx)
        result = _invoke({"note": "B" * 40}, ctx)
        self.assertEqual(result["status"], "ok")
        self.thread.refresh_from_db()
        # Capped to 50 chars, and the most recent note's content survives.
        self.assertLessEqual(len(self.thread.scratchpad), 50)
        self.assertIn("B", self.thread.scratchpad)
        # The result flags that oldest notes were dropped.
        self.assertIn("full", result.get("note", "").lower())

    def test_thread_not_found_returns_error(self):
        import uuid

        result = _invoke({"note": "hi"}, _ctx(self.user.pk, uuid.uuid4()))
        self.assertEqual(result["status"], "error")


class ScratchpadToolMetadataTests(TestCase):
    def test_tool_is_main_audience_and_always_on(self):
        tool = ScratchpadAppendTool()
        self.assertEqual(tool.audience, "main")
        self.assertEqual(tool.section, "chat")

    def test_tool_not_in_canvas_updated_tools(self):
        """The scratchpad must never emit a user-visible content event."""
        from chat.consumers import CANVAS_UPDATED_TOOLS

        self.assertNotIn("scratchpad_append", CANVAS_UPDATED_TOOLS)


class ScratchpadInjectionTests(TestCase):
    def test_scratchpad_injected_into_dynamic_context(self):
        out = build_dynamic_context(scratchpad="Key fact: the deadline is Friday.")
        self.assertIn("Scratchpad", out)
        self.assertIn("Key fact: the deadline is Friday.", out)

    def test_empty_scratchpad_not_injected(self):
        out = build_dynamic_context(scratchpad="")
        self.assertNotIn("# Scratchpad", out)

    def test_whitespace_scratchpad_not_injected(self):
        out = build_dynamic_context(scratchpad="   \n  ")
        self.assertNotIn("# Scratchpad", out)

    def test_no_scratchpad_arg_not_injected(self):
        out = build_dynamic_context()
        self.assertNotIn("# Scratchpad", out)
