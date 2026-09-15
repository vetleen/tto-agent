"""Tests for the agent-only scratchpad: append tool, cap, and prompt injection.

The scratchpad is the assistant's private working memory. It is append-only,
capped, injected into every turn's dynamic context, and NEVER surfaced to the
user (no ``canvas.updated``-style event, not in ``CANVAS_UPDATED_TOOLS``).
"""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from chat.models import ChatThread, SubAgentRun
from chat.prompts import build_dynamic_context
from chat.scratchpad_tools import ScratchpadAppendTool, SubagentScratchpadAppendTool
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(args, ctx):
    tool = ScratchpadAppendTool()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


def _sub_ctx(user_id, thread_id, run_id):
    ctx = RunContext.create(user_id=user_id, conversation_id=str(thread_id))
    ctx.run_id = str(run_id)
    ctx.agent_kind = "subagent"
    return ctx


def _invoke_sub(args, ctx):
    tool = SubagentScratchpadAppendTool()
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


class SubagentScratchpadAppendToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="subpad@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="do the thing"
        )

    def test_appends_note_to_run(self):
        ctx = _sub_ctx(self.user.pk, self.thread.id, self.run.id)
        result = _invoke_sub({"note": "Finding: X = 42"}, ctx)
        self.assertEqual(result["status"], "ok")
        self.run.refresh_from_db()
        self.assertEqual(self.run.scratchpad, "Finding: X = 42")

    def test_updates_in_memory_context(self):
        """The tool must update context.scratchpad so the tool loop can re-inject
        it next iteration without a DB read."""
        ctx = _sub_ctx(self.user.pk, self.thread.id, self.run.id)
        _invoke_sub({"note": "remember this"}, ctx)
        self.assertEqual(ctx.scratchpad, "remember this")

    def test_appends_are_concatenated(self):
        ctx = _sub_ctx(self.user.pk, self.thread.id, self.run.id)
        _invoke_sub({"note": "one"}, ctx)
        _invoke_sub({"note": "two"}, ctx)
        self.run.refresh_from_db()
        self.assertLess(
            self.run.scratchpad.index("one"), self.run.scratchpad.index("two")
        )

    def test_empty_note_is_rejected(self):
        ctx = _sub_ctx(self.user.pk, self.thread.id, self.run.id)
        result = _invoke_sub({"note": "  "}, ctx)
        self.assertEqual(result["status"], "error")
        self.run.refresh_from_db()
        self.assertEqual(self.run.scratchpad, "")

    def test_no_run_context_returns_error(self):
        # A context with no run_id (e.g. the main agent) can't resolve a run.
        ctx = _ctx(self.user.pk, self.thread.id)
        ctx.run_id = ""
        result = _invoke_sub({"note": "hi"}, ctx)
        self.assertEqual(result["status"], "error")

    def test_run_not_found_returns_error(self):
        import uuid

        ctx = _sub_ctx(self.user.pk, self.thread.id, uuid.uuid4())
        result = _invoke_sub({"note": "hi"}, ctx)
        self.assertEqual(result["status"], "error")

    @override_settings(SCRATCHPAD_MAX_CHARS=50)
    def test_cap_tail_truncates_keeping_recent(self):
        ctx = _sub_ctx(self.user.pk, self.thread.id, self.run.id)
        _invoke_sub({"note": "A" * 40}, ctx)
        result = _invoke_sub({"note": "B" * 40}, ctx)
        self.run.refresh_from_db()
        self.assertLessEqual(len(self.run.scratchpad), 50)
        self.assertIn("B", self.run.scratchpad)
        self.assertIn("full", result.get("note", "").lower())


class SubagentScratchpadMetadataTests(TestCase):
    def test_tool_is_subagent_audience_and_always_on(self):
        tool = SubagentScratchpadAppendTool()
        self.assertEqual(tool.audience, "subagent")
        self.assertEqual(tool.section, "chat")

    def test_tool_not_in_canvas_updated_tools(self):
        from chat.consumers import CANVAS_UPDATED_TOOLS

        self.assertNotIn("subagent_scratchpad_append", CANVAS_UPDATED_TOOLS)

    def test_main_scratchpad_stays_main_only(self):
        # The two tools must not blur: the main scratchpad is main-audience and
        # thread-scoped; the sub-agent one is subagent-audience and run-scoped.
        self.assertEqual(ScratchpadAppendTool().audience, "main")


class SubagentScratchpadInjectionTests(TestCase):
    """The pipeline re-injects the run scratchpad as a single trailing block each
    iteration so it survives mid-run pruning (see _refresh_subagent_scratchpad)."""

    def _pipeline(self):
        from llm.pipelines.simple_chat import SimpleChatPipeline

        return SimpleChatPipeline()

    def _req(self, agent_kind, scratchpad):
        from llm.types import ChatRequest, Message

        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
        ctx.scratchpad = scratchpad
        return ChatRequest(
            messages=[Message(role="system", content="sys")],
            model="gpt-4o",
            context=ctx,
        )

    def _marker(self):
        from llm.pipelines.simple_chat import _SUBAGENT_SCRATCHPAD_MARKER

        return _SUBAGENT_SCRATCHPAD_MARKER

    def test_notes_injected_as_trailing_block(self):
        from llm.types import Message

        pipe = self._pipeline()
        req = self._req("subagent", "Fact A\nFact B")
        messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        pipe._refresh_subagent_scratchpad(messages, req)
        self.assertTrue(messages[-1].content.startswith(self._marker()))
        self.assertIn("Fact A", messages[-1].content)

    def test_reinjection_is_idempotent(self):
        from llm.types import Message

        pipe = self._pipeline()
        req = self._req("subagent", "Fact A")
        messages = [Message(role="system", content="sys")]
        pipe._refresh_subagent_scratchpad(messages, req)
        # Simulate a next round: scratchpad grew, re-inject.
        req.context.scratchpad = "Fact A\nFact B"
        pipe._refresh_subagent_scratchpad(messages, req)
        blocks = [
            m for m in messages
            if isinstance(m.content, str) and m.content.startswith(self._marker())
        ]
        self.assertEqual(len(blocks), 1)
        self.assertIn("Fact B", blocks[0].content)

    def test_main_agent_is_noop(self):
        from llm.types import Message

        pipe = self._pipeline()
        req = self._req("main", "should be ignored")
        messages = [Message(role="user", content="hi")]
        pipe._refresh_subagent_scratchpad(messages, req)
        self.assertEqual(len(messages), 1)
        self.assertNotIn(self._marker(), messages[0].content)
