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




class ScratchpadEditPointTests(TestCase):
    """The tool loop never re-injects the scratchpad between edit points (the
    notes are visible as the append calls' own arguments); at an edit point it
    appends ONE consolidated block and shrinks the old note arguments."""

    LONG = "Fact: " + "x" * 300  # over the scratchpad tools' trim threshold

    def _req(self, agent_kind, messages, scratchpad=""):
        from llm.types import ChatRequest

        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
        ctx.scratchpad = scratchpad
        return ChatRequest(
            messages=messages, model="gpt-5.4", context=ctx,
            params={"max_context_tokens": 50_000, "thinking_level": "low"},
        )

    def _mgr(self, req):
        from llm.pipelines.edit_points import EditPointManager
        from llm.pipelines.simple_chat import SimpleChatPipeline

        return EditPointManager(SimpleChatPipeline(), req, {})

    def _note_round(self, cid, tool, note):
        from llm.types import Message, ToolCall

        return [
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id=cid, name=tool, arguments={"note": note})]),
            Message(role="tool", content='{"status": "ok"}', tool_call_id=cid),
        ]

    def _blocks(self, messages):
        from llm.pipelines.edit_points import SCRATCHPAD_BLOCK_HEADER

        return [
            m for m in messages
            if isinstance(m.content, str) and m.content.startswith(SCRATCHPAD_BLOCK_HEADER)
        ]

    def _finish(self, mgr, req, new_messages, *, real_input_tokens, cid):
        out, _tools = mgr.finish_round(
            new_messages, req, [], prior_len=len(req.messages),
            real_input_tokens=real_input_tokens, round_call_ids={cid}, notice=None,
        )
        return out

    def test_quiet_round_adds_no_block(self):
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        req = self._req("subagent", base)
        mgr = self._mgr(req)
        req.context.scratchpad = self.LONG  # the tool updated it this round
        out = self._finish(
            mgr, req, base + self._note_round("c1", "subagent_scratchpad_append", self.LONG),
            real_input_tokens=100, cid="c1",
        )
        self.assertEqual(self._blocks(out), [])
        self.assertEqual(mgr.edit_rounds, [])

    def test_edit_point_appends_one_block_and_shrinks_old_notes(self):
        from chat.scratchpad_tools import SAVED_NOTE_MARKER
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        history = base + self._note_round("c1", "subagent_scratchpad_append", self.LONG)
        req = self._req("subagent", history, scratchpad=self.LONG)
        mgr = self._mgr(req)
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out = self._finish(
                mgr, req, history + self._note_round("c2", "web_search", "q"),
                real_input_tokens=40_000, cid="c2",  # over the ceiling: edit point
            )
            # A second edit point replaces (not duplicates) the block.
            req2 = req.model_copy(update={"messages": out})
            req2.context.scratchpad = self.LONG + "\n\nFact B"
            out2 = self._finish(
                mgr, req2, out + self._note_round("c3", "web_search", "q2"),
                real_input_tokens=40_000, cid="c3",
            )
        self.assertEqual(mgr.edit_rounds, [0, 1])
        self.assertIs(out[-1], self._blocks(out)[-1])  # block is last
        blocks = self._blocks(out2)
        self.assertEqual(len(blocks), 1)
        self.assertIn("Fact B", blocks[0].content)
        note_call = out2[2].tool_calls[0]
        self.assertEqual(note_call.arguments["note"], SAVED_NOTE_MARKER)
        self.assertEqual(req.context.observability["edit_points_total"], 2)

    def test_main_agent_block_has_only_this_turns_notes(self):
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        base += self._note_round("c0", "web_search", "old")  # something to prune
        req = self._req("main", base, scratchpad="")
        req.context.scratchpad_turn_notes.append("NEW THIS TURN")
        mgr = self._mgr(req)
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out = self._finish(
                mgr, req, base + self._note_round("c1", "web_search", "q"),
                real_input_tokens=40_000, cid="c1",
            )
        blocks = self._blocks(out)
        self.assertEqual(len(blocks), 1)
        self.assertIn("NEW THIS TURN", blocks[0].content)

    def test_over_ceiling_with_nothing_to_change_is_not_an_edit(self):
        """Still over the ceiling but nothing left to prune/trim: stay
        append-only (no block churn every round, no edit point counted)."""
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        req = self._req("main", base, scratchpad="")
        req.context.scratchpad_turn_notes.append("NOTE")
        mgr = self._mgr(req)
        new = base + self._note_round("c1", "web_search", "q")
        out = self._finish(mgr, req, new, real_input_tokens=40_000, cid="c1")
        self.assertEqual(out, new)
        self.assertEqual(mgr.edit_rounds, [])
        self.assertNotIn("edit_points_total", req.context.observability)

    def test_main_agent_without_turn_notes_gets_no_block(self):
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        req = self._req("main", base)
        out = self._finish(
            self._mgr(req), req, base + self._note_round("c1", "web_search", "q"),
            real_input_tokens=40_000, cid="c1",
        )
        self.assertEqual(self._blocks(out), [])

    def test_seeded_subagent_scratchpad_shown_once_append_only(self):
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        req = self._req("subagent", base, scratchpad="Seeded from a retried run")
        mgr = self._mgr(req)
        out = self._finish(
            mgr, req, base + self._note_round("c1", "web_search", "q"),
            real_input_tokens=100, cid="c1",
        )
        self.assertEqual(len(self._blocks(out)), 1)
        self.assertEqual(out[:len(base)], base)  # nothing earlier touched
        req2 = req.model_copy(update={"messages": out})
        out2 = self._finish(
            mgr, req2, out + self._note_round("c2", "web_search", "q2"),
            real_input_tokens=100, cid="c2",
        )
        self.assertEqual(len(self._blocks(out2)), 1)  # not re-appended

    def test_subagent_nudge_appended_once_when_filling(self):
        from llm.types import Message

        base = [Message(role="system", content="sys"), Message(role="user", content="hi")]
        req = self._req("subagent", base)
        mgr = self._mgr(req)
        out = self._finish(
            mgr, req, base + self._note_round("c1", "web_search", "q"),
            real_input_tokens=20_000, cid="c1",  # >= 0.7 x ceiling, under it
        )
        req2 = req.model_copy(update={"messages": out})
        out2 = self._finish(
            mgr, req2, out + self._note_round("c2", "web_search", "q2"),
            real_input_tokens=21_000, cid="c2",
        )
        nudges = [m for m in out2 if isinstance(m.content, str) and m.content.startswith("# Context notice")]
        self.assertEqual(len(nudges), 1)
        self.assertEqual(out2[:len(out)], out)  # append-only

    def test_append_tools_record_turn_notes(self):
        user = User.objects.create_user(email="notes@example.com", password="x")
        thread = ChatThread.objects.create(created_by=user, title="t")
        ctx = _ctx(user.pk, thread.pk)
        _invoke({"note": "remember this"}, ctx)
        self.assertEqual(ctx.scratchpad_turn_notes, ["remember this"])
