"""Append-only tool loop with edit points (llm/pipelines/edit_points.py).

Covers the per-tool argument trimming hook, native-asset eviction hysteresis,
edit-point triggers/counters, the provider opt-ins for Anthropic preserved
thinking, and — end to end through SimpleChatPipeline — the invariant that each
request extends the previous one unchanged except on recorded edit points.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, TestCase
from pydantic import BaseModel, Field

from llm.pipelines.edit_points import (
    RESULT_NOTE_FIELD,
    TRIM_NOTES_KEY,
    EditPointManager,
    TRIMMED_ARGS_KEY,
    attach_trim_notes,
    record_thinking_drops,
    trim_history_call_args,
    trim_old_tool_call_args,
)
from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.tools.interfaces import ArgTrim, ContextAwareTool, find_echoed_marker
from llm.types.context import RunContext
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent


def _asst(cid, name, args):
    return Message(role="assistant", content="", tool_calls=[ToolCall(id=cid, name=name, arguments=args)])


def _tool(cid, body):
    return Message(role="tool", content=body, tool_call_id=cid)


class _TrimAll:
    """Minimal stand-in tool whose hook shrinks ``content``."""

    trim_args_min_chars = 50

    def __init__(self):
        self.seen_later = []

    def trim_args_at_edit_point(self, args, *, later_calls):
        self.seen_later.append([c.id for c in later_calls])
        return {**args, "content": "[trimmed]"}


class _TrimAllRecent(_TrimAll):
    """Same hook, but opts out of recency protection (like the slide tools)."""

    trim_despite_recency = True


class _TrimResult:
    """Stand-in tool whose RESULT hook shrinks ``deck_json``; opts out of recency."""

    trim_despite_recency = True
    trim_result_min_chars = 50

    def __init__(self):
        self.seen_later = []

    def trim_result_at_edit_point(self, result, args, *, later_calls):
        self.seen_later.append([c.id for c in later_calls])
        return '{"deck_json": "[trimmed]"}'


class TrimOldToolCallArgsTests(SimpleTestCase):
    def test_trims_old_calls_but_not_protected_or_small(self):
        big = "x" * 200
        msgs = [
            Message(role="user", content="q"),
            _asst("c1", "writer", {"content": big}),
            _tool("c1", "ok"),
            _asst("c2", "writer", {"content": "tiny"}),
            _tool("c2", "ok"),
            _asst("c3", "writer", {"content": big}),
            _tool("c3", "ok"),
        ]
        tool = _TrimAll()
        out, n = trim_old_tool_call_args(
            msgs, protected_call_ids={"c3"}, tool_lookup=lambda name: tool,
        )
        self.assertEqual(n, 1)
        self.assertEqual(out[1].tool_calls[0].arguments["content"], "[trimmed]")
        self.assertEqual(out[1].metadata[TRIMMED_ARGS_KEY], ["c1"])
        self.assertIs(out[3], msgs[3])  # under the size threshold
        self.assertIs(out[5], msgs[5])  # protected
        self.assertEqual(tool.seen_later[0], ["c2", "c3"])  # later calls passed
        self.assertEqual(msgs[1].tool_calls[0].arguments["content"], big)  # not mutated

    def test_already_trimmed_is_skipped(self):
        msg = _asst("c1", "writer", {"content": "x" * 200})
        msg = msg.model_copy(update={"metadata": {TRIMMED_ARGS_KEY: ["c1"]}})
        out, n = trim_old_tool_call_args(
            [msg], protected_call_ids=set(), tool_lookup=lambda name: _TrimAll(),
        )
        self.assertEqual(n, 0)
        self.assertIs(out[0], msg)

    def test_unknown_tool_and_default_hook_keep_args(self):
        class _Keeps(ContextAwareTool):
            name: str = "keeps"
            description: str = "d"

            def _run(self, **kw):
                return ""

        msgs = [_asst("c1", "keeps", {"content": "x" * 5000}), _asst("c2", "gone", {"content": "x" * 5000})]
        lookup = {"keeps": _Keeps()}.get
        out, n = trim_old_tool_call_args(msgs, protected_call_ids=set(), tool_lookup=lookup)
        self.assertEqual(n, 0)

    def test_history_variant_returns_new_args_by_id(self):
        tool = _TrimAll()
        calls = [
            {"id": "a", "name": "writer", "arguments": {"content": "x" * 200}},
            {"id": "b", "name": "writer", "arguments": {"content": "x" * 200}},
        ]
        out = trim_history_call_args(calls, {"a"}, tool_lookup=lambda name: tool)
        self.assertEqual(out, {"a": ArgTrim(args={"content": "[trimmed]"}, note="")})

    def test_arg_trim_note_recorded_on_the_assistant_message(self):
        class _Drops(_TrimAll):
            def trim_args_at_edit_point(self, args, *, later_calls):
                return ArgTrim(args={k: v for k, v in args.items() if k != "content"}, note="NOTE-c1")

        msgs = [
            Message(role="user", content="q"),
            _asst("c1", "writer", {"title": "T", "content": "x" * 200}), _tool("c1", '{"status": "ok"}'),
        ]
        out, n = trim_old_tool_call_args(msgs, protected_call_ids=set(), tool_lookup=lambda name: _Drops())
        self.assertEqual(n, 1)
        self.assertEqual(out[1].tool_calls[0].arguments, {"title": "T"})
        self.assertEqual(out[1].metadata[TRIM_NOTES_KEY], {"c1": "NOTE-c1"})
        self.assertIs(out[2], msgs[2])  # the edit point attaches the note later

    def test_small_slide_edit_untouched(self):
        """Below trim_args_min_chars (2000) a slide edit keeps its args even though
        the hook would trim it — small edits aren't worth an edit of history."""
        from chat.slide_tools import EditDeckTool

        msgs = [
            Message(role="user", content="q"),
            _asst("e1", "slide_canvas_edit", {"edits": [{"old_text": "a" * 100, "new_text": "b" * 100}]}),
            _tool("e1", "ok"),
            _asst("e2", "web_search", {"q": 1}), _tool("e2", "r"),
        ]
        out, n = trim_old_tool_call_args(
            msgs, protected_call_ids=set(), tool_lookup=lambda name: EditDeckTool(),
        )
        self.assertEqual(n, 0)
        self.assertEqual(out, msgs)

    def test_recent_calls_kept_unless_tool_opts_out(self):
        """Recency protects a call's arguments unless its tool sets
        trim_despite_recency; the current round (protected ids) always wins."""
        big = "x" * 200
        msgs = [
            Message(role="user", content="q"),
            _asst("c1", "writer", {"content": big}), _tool("c1", "r1"),
            _asst("c2", "writer", {"content": big}), _tool("c2", "r2"),
            _asst("c3", "writer", {"content": big}), _tool("c3", "r3"),
        ]
        plain = _TrimAll()
        out, n = trim_old_tool_call_args(
            msgs, protected_call_ids={"c3"}, recent_call_ids={"c2", "c3"},
            tool_lookup=lambda name: plain,
        )
        self.assertEqual(n, 1)
        self.assertEqual(out[1].tool_calls[0].arguments["content"], "[trimmed]")
        self.assertEqual(out[3].tool_calls[0].arguments["content"], big)
        flagged = _TrimAllRecent()
        out, n = trim_old_tool_call_args(
            msgs, protected_call_ids={"c3"}, recent_call_ids={"c2", "c3"},
            tool_lookup=lambda name: flagged,
        )
        self.assertEqual(n, 2)
        self.assertEqual(out[3].tool_calls[0].arguments["content"], "[trimmed]")
        self.assertEqual(out[5].tool_calls[0].arguments["content"], big)  # current round
        self.assertEqual(out[1].metadata[TRIMMED_ARGS_KEY], ["c1"])
        self.assertEqual(out[3].metadata[TRIMMED_ARGS_KEY], ["c2"])

    def test_history_variant_recent_ids_only_for_flagged_tools(self):
        calls = [
            {"id": "a", "name": "writer", "arguments": {"content": "x" * 200}},
            {"id": "b", "name": "writer", "arguments": {"content": "y" * 200}},
        ]
        plain = trim_history_call_args(calls, {"a"}, tool_lookup=lambda n: _TrimAll(), recent_ids={"b"})
        self.assertEqual(set(plain), {"a"})
        flagged = trim_history_call_args(calls, {"a"}, tool_lookup=lambda n: _TrimAllRecent(), recent_ids={"b"})
        self.assertEqual(set(flagged), {"a", "b"})


class TrimOldToolResultsTests(SimpleTestCase):
    """The result-side twin of the argument trimming."""

    def _msgs(self, body='{"deck_json": "' + "d" * 200 + '", "deck_id": "7"}'):
        return [
            Message(role="user", content="q"),
            _asst("c1", "deck_tool", {"title": "T"}), _tool("c1", body),
            _asst("c2", "deck_tool", {"title": "T"}), _tool("c2", body),
        ]

    def test_shrinks_recent_flagged_result_marks_metadata_and_keeps_current_round(self):
        from llm.pipelines.edit_points import TRIMMED_RESULT_KEY, trim_old_tool_results

        msgs = self._msgs()
        tool = _TrimResult()
        out, n = trim_old_tool_results(
            msgs, protected_call_ids={"c2"}, recent_call_ids={"c1", "c2"},
            tool_lookup=lambda name: tool,
        )
        self.assertEqual(n, 1)
        self.assertEqual(out[2].content, '{"deck_json": "[trimmed]"}')
        self.assertTrue(out[2].metadata[TRIMMED_RESULT_KEY])
        self.assertIs(out[4], msgs[4])  # current round untouched
        self.assertEqual(tool.seen_later, [["c2"]])
        self.assertTrue(msgs[2].content.startswith('{"deck_json": "ddd'))  # originals unmutated

    def test_skips_stubbed_small_marked_unflagged_and_unchanged(self):
        from llm.pipelines.edit_points import TRIMMED_RESULT_KEY, trim_old_tool_results

        class _Unflagged(_TrimResult):
            trim_despite_recency = False

        class _Same(_TrimResult):
            def trim_result_at_edit_point(self, result, args, *, later_calls):
                return result

        marked = self._msgs()
        marked[2] = marked[2].model_copy(update={"metadata": {TRIMMED_RESULT_KEY: True}})
        cases = {
            "stubbed": (self._msgs(body="[Earlier result of deck_tool(...) was cleared]"), _TrimResult(), set()),
            "small": (self._msgs(body='{"deck_json": "d"}'), _TrimResult(), set()),
            "marked": (marked, _TrimResult(), set()),
            "unflagged_recent": (self._msgs(), _Unflagged(), {"c1", "c2"}),
            "unchanged": (self._msgs(), _Same(), set()),
        }
        for label, (msgs, tool, recent) in cases.items():
            with self.subTest(label):
                out, n = trim_old_tool_results(
                    msgs, protected_call_ids={"c2"}, recent_call_ids=recent,
                    tool_lookup=lambda name, t=tool: t,
                )
                self.assertEqual(n, 0)
                self.assertIs(out, msgs)

    def test_hook_exception_logged_and_skipped(self):
        from llm.pipelines.edit_points import trim_old_tool_results

        class _Boom(_TrimResult):
            def trim_result_at_edit_point(self, result, args, *, later_calls):
                raise RuntimeError("bad hook")

        msgs = self._msgs()
        with self.assertLogs("llm.pipelines.edit_points", level="ERROR"):
            out, n = trim_old_tool_results(
                msgs, protected_call_ids=set(), recent_call_ids=set(), tool_lookup=lambda name: _Boom(),
            )
        self.assertEqual(n, 0)
        self.assertIs(out, msgs)

    def test_history_variant_returns_new_content_by_id(self):
        from llm.pipelines.edit_points import trim_history_results

        body = '{"deck_json": "' + "d" * 200 + '"}'
        calls = [
            {"id": "a", "name": "deck_tool", "arguments": {}},
            {"id": "b", "name": "other", "arguments": {}},
        ]
        tool = _TrimResult()
        out = trim_history_results(
            [("a", body), ("b", body)], calls,
            tool_lookup=lambda name: tool if name == "deck_tool" else None,
        )
        self.assertEqual(out, {"a": '{"deck_json": "[trimmed]"}'})
        self.assertEqual(tool.seen_later, [["b"]])


class ToolTrimHookTests(SimpleTestCase):
    """The per-tool trim rules: big fields are REMOVED (never replaced by text the
    model could copy back, WILFRED-9Q) and the note says where the content lives."""

    def assertDropped(self, trim, key, *in_note):
        self.assertIsInstance(trim, ArgTrim)
        self.assertNotIn(key, trim.args)
        self.assertIsNone(find_echoed_marker(trim.args))
        self.assertTrue(trim.note.startswith("[Omitted from this transcript"), trim.note)
        for text in in_note:
            self.assertIn(text, trim.note)

    def test_canvas_write_and_edit_point_to_canvas_read(self):
        from chat.canvas_tools import EditCanvasTool, WriteCanvasTool

        w = WriteCanvasTool().trim_args_at_edit_point({"title": "T", "content": "y" * 3000}, later_calls=[])
        self.assertDropped(w, "content", "canvas_read", "3,000-char document")
        self.assertEqual(w.args["title"], "T")
        e = EditCanvasTool().trim_args_at_edit_point(
            {"edits": [{"old_text": "a", "new_text": "b"}], "canvas_name": "C"}, later_calls=[],
        )
        self.assertDropped(e, "edits", "canvas_read", "list of 1 edit(s)")
        self.assertEqual(e.args["canvas_name"], "C")
        # Idempotent: an already-trimmed call is left alone.
        self.assertIsNone(WriteCanvasTool().trim_args_at_edit_point(w.args, later_calls=[]))
        self.assertIsNone(EditCanvasTool().trim_args_at_edit_point(e.args, later_calls=[]))

    def test_subagent_canvas_args_trim_to_read_note(self):
        """Since subagent_canvas_read exists, old sub-agent canvas write/edit
        arguments are recoverable and trim like the main agent's."""
        from chat.subagent_canvas_tools import SubagentCanvasEditTool, SubagentCanvasWriteTool

        out = SubagentCanvasWriteTool().trim_args_at_edit_point({"content": "z" * 3000}, later_calls=[])
        self.assertDropped(out, "content", "subagent_canvas_read")
        self.assertIsNone(SubagentCanvasWriteTool().trim_args_at_edit_point(out.args, later_calls=[]))
        edit = SubagentCanvasEditTool().trim_args_at_edit_point(
            {"edits": [{"old_text": "a", "new_text": "b"}]}, later_calls=[],
        )
        self.assertDropped(edit, "edits", "subagent_canvas_read")

    def test_scratchpad_note_dropped(self):
        from chat.scratchpad_tools import ScratchpadAppendTool, SubagentScratchpadAppendTool

        for tool in (ScratchpadAppendTool(), SubagentScratchpadAppendTool()):
            with self.subTest(tool=tool.name):
                out = tool.trim_args_at_edit_point({"note": "n" * 300, "reason": "r"}, later_calls=[])
                self.assertDropped(out, "note", "scratchpad")
                self.assertEqual(out.args, {"reason": "r"})
                self.assertIsNone(tool.trim_args_at_edit_point(out.args, later_calls=[]))

    def test_slide_write_trimmed_while_deck_stays_active(self):
        """A deck write is trimmed whenever its deck is still the active one (its
        current state is in the Active Slide Deck section / edit-point block) or a
        later full rewrite replaced it; a later write of ANOTHER named deck keeps it."""
        from chat.slide_tools import WriteDeckTool

        args = {"content": {"slides": []}, "deck_name": "Pitch"}
        out = WriteDeckTool().trim_args_at_edit_point(args, later_calls=[])
        self.assertDropped(out, "content", "Active Slide Deck")
        self.assertEqual(out.args["deck_name"], "Pitch")
        same = [ToolCall(id="y", name="slide_canvas_write", arguments={"deck_name": "Pitch"})]
        self.assertIn("replaced", WriteDeckTool().trim_args_at_edit_point(args, later_calls=same).note)
        other = [ToolCall(id="x", name="slide_canvas_write", arguments={"deck_name": "Other"})]
        self.assertIsNone(WriteDeckTool().trim_args_at_edit_point(args, later_calls=other))
        # Unnamed calls target the then-active deck; an unnamed later edit keeps it active.
        unnamed = {"content": {"slides": []}}
        later_edit = [ToolCall(id="e", name="slide_canvas_edit", arguments={"edits": []})]
        self.assertIn(
            "Active Slide Deck",
            WriteDeckTool().trim_args_at_edit_point(unnamed, later_calls=later_edit).note,
        )
        # Idempotent: an already-trimmed write is left alone.
        self.assertIsNone(WriteDeckTool().trim_args_at_edit_point(out.args, later_calls=[]))

    def test_slide_write_kept_after_switch(self):
        """Anything that may have changed which deck is active keeps the JSON (the
        section / block only ever shows the active deck); irrelevant later calls don't."""
        from chat.slide_tools import WriteDeckTool

        args = {"content": {"slides": []}, "deck_name": "Pitch"}
        for switch in (
            ToolCall(id="a", name="slide_canvas_activate", arguments={"deck_names": ["Other"]}),
            ToolCall(id="c", name="slides_create_deck", arguments={"title": "New", "layouts": []}),
            ToolCall(id="d", name="slide_canvas_delete", arguments={"deck_name": "Pitch"}),
            ToolCall(id="l", name="skill_resource_load", arguments={"template_name": "t", "target": "deck"}),
            ToolCall(id="t", name="slides_set_theme", arguments={"theme": "slate", "deck_name": "Other"}),
        ):
            with self.subTest(switch=switch.name):
                self.assertIsNone(WriteDeckTool().trim_args_at_edit_point(args, later_calls=[switch]))
        for harmless in (
            ToolCall(id="p", name="slides_preview_slide", arguments={"deck_name": "Other"}),
            ToolCall(id="d2", name="slide_canvas_delete", arguments={"deck_name": "Other"}),
            ToolCall(id="l2", name="skill_resource_load", arguments={"template_name": "t", "target": "canvas"}),
            ToolCall(id="w", name="web_search", arguments={"query": "x"}),
            ToolCall(id="a2", name="slide_canvas_activate", arguments={"deck_names": ["Pitch"]}),
        ):
            with self.subTest(harmless=harmless.name):
                out = WriteDeckTool().trim_args_at_edit_point(args, later_calls=[harmless])
                self.assertDropped(out, "content", "Active Slide Deck")

    def test_slide_write_not_superseded_across_a_switch(self):
        """write("") -> activate("Other") -> write(""): the second write hit another
        deck, so the first is neither replaced nor safely in the context — kept."""
        from chat.slide_tools import WriteDeckTool

        args = {"content": {"slides": [{"id": "s1"}]}}
        later = [
            ToolCall(id="a", name="slide_canvas_activate", arguments={"deck_names": ["Other"]}),
            ToolCall(id="w", name="slide_canvas_write", arguments={"content": {}}),
        ]
        self.assertIsNone(WriteDeckTool().trim_args_at_edit_point(args, later_calls=later))

    def test_slide_edit_hook_mirrors_write(self):
        from chat.slide_tools import EditDeckTool

        args = {"edits": [{"old_text": "a", "new_text": "b"}, {"old_text": "c", "new_text": "d"}], "deck_name": ""}
        out = EditDeckTool().trim_args_at_edit_point(args, later_calls=[])
        self.assertDropped(out, "edits", "list of 2 edit(s)", "Active Slide Deck")
        self.assertIsNone(EditDeckTool().trim_args_at_edit_point(out.args, later_calls=[]))  # idempotent
        as_string = {"edits": '[{"old_text": "a", "new_text": "b"}]'}
        self.assertDropped(EditDeckTool().trim_args_at_edit_point(as_string, later_calls=[]), "edits", "the edits")
        superseded = [ToolCall(id="w", name="slide_canvas_write", arguments={"content": {}})]
        self.assertIn("replaced", EditDeckTool().trim_args_at_edit_point(args, later_calls=superseded).note)
        switched = [ToolCall(id="a", name="slide_canvas_activate", arguments={"deck_names": ["Other"]})]
        self.assertIsNone(EditDeckTool().trim_args_at_edit_point(args, later_calls=switched))

    def test_slide_tools_opt_out_of_recency(self):
        from agent_skills.tools import LoadTemplateToCanvasTool
        from chat.canvas_tools import WriteCanvasTool
        from chat.slide_tools import CreateDeckTool, EditDeckTool, WriteDeckTool

        for tool in (WriteDeckTool(), EditDeckTool(), CreateDeckTool(), LoadTemplateToCanvasTool()):
            self.assertTrue(tool.trim_despite_recency, tool.name)
        self.assertFalse(WriteCanvasTool().trim_despite_recency)
        self.assertFalse(ContextAwareTool.model_fields["trim_despite_recency"].default)

    def assertFieldDropped(self, data, key, *in_note):
        self.assertNotIn(key, data)
        note = data["omitted_fields_note"]
        self.assertTrue(note.startswith("[Omitted from this transcript"), note)
        for text in in_note:
            self.assertIn(text, note)

    def test_create_deck_result_hook(self):
        from chat.slide_tools import CreateDeckTool

        result = json.dumps({
            "status": "ok", "deck_id": "7", "title": "Pitch", "slide_ids": ["s1", "s2"],
            "changed_slide_ids": ["s1", "s2"], "slide_count": 2, "deck_json": "{" + "x" * 3000 + "}",
        })
        args = {"title": "Pitch", "layouts": ["title", "bullets"]}
        tool = CreateDeckTool()
        out = tool.trim_result_at_edit_point(result, args, later_calls=[])
        data = json.loads(out)
        self.assertFieldDropped(data, "deck_json", "Active Slide Deck", "seeded deck JSON")
        self.assertEqual(data["slide_ids"], ["s1", "s2"])
        self.assertEqual(data["deck_id"], "7")
        self.assertLess(len(out), len(result))
        superseded = [ToolCall(id="w", name="slide_canvas_write", arguments={"deck_name": "Pitch"})]
        replaced = json.loads(tool.trim_result_at_edit_point(result, args, later_calls=superseded))
        self.assertFieldDropped(replaced, "deck_json", "replaced")
        switched = [ToolCall(id="a", name="slide_canvas_activate", arguments={"deck_names": ["Other"]})]
        self.assertIsNone(tool.trim_result_at_edit_point(result, args, later_calls=switched))
        error = json.dumps({"status": "error", "deck_json": "x" * 3000})
        self.assertIsNone(tool.trim_result_at_edit_point(error, args, later_calls=[]))
        self.assertIsNone(tool.trim_result_at_edit_point("not json", args, later_calls=[]))
        self.assertIsNone(tool.trim_result_at_edit_point(out, args, later_calls=[]))  # idempotent

    def test_skill_resource_load_result_hook_deck_slide_and_canvas(self):
        from agent_skills.tools import LoadTemplateToCanvasTool

        tool = LoadTemplateToCanvasTool()
        args = {"template_name": "Full Deck", "target": "deck", "canvas_name": "Full Deck"}
        deck = json.dumps({
            "status": "ok", "loaded_as": "deck", "deck_id": "9", "title": "Full Deck (2)",
            "slide_ids": ["s1"], "deck_json": "{" + "x" * 3000 + "}",
        })
        out = json.loads(tool.trim_result_at_edit_point(deck, args, later_calls=[]))
        self.assertFieldDropped(out, "deck_json", "loaded deck JSON")
        self.assertEqual(out["title"], "Full Deck (2)")
        # A later write naming "Full Deck" is ANOTHER deck: the stored title is the de-duplicated one.
        other = [ToolCall(id="w", name="slide_canvas_write", arguments={"deck_name": "Full Deck"})]
        self.assertIsNone(tool.trim_result_at_edit_point(deck, args, later_calls=other))
        same = [ToolCall(id="w", name="slide_canvas_write", arguments={"deck_name": "Full Deck (2)"})]
        self.assertFieldDropped(
            json.loads(tool.trim_result_at_edit_point(deck, args, later_calls=same)), "deck_json", "replaced",
        )
        slide = json.dumps({
            "status": "ok", "loaded_as": "slide", "deck_id": "9", "title": "QA Deck",
            "slide_id": "s3", "slide_json": "{" + "x" * 3000 + "}",
        })
        shrunk = json.loads(tool.trim_result_at_edit_point(slide, args, later_calls=[]))
        self.assertFieldDropped(shrunk, "slide_json", "loaded slide JSON")
        self.assertEqual(shrunk["slide_id"], "s3")
        canvas = json.dumps({"status": "ok", "title": "Notes", "canvas_id": "c1", "note": "n" * 3000})
        self.assertIsNone(tool.trim_result_at_edit_point(canvas, args, later_calls=[]))

    def test_slide_hooks_keep_payloads_when_deck_block_failed(self):
        from chat.slide_tools import CreateDeckTool, WriteDeckTool

        ctx = RunContext.create(user_id="1")
        ctx.deck_block_failed = True
        write = WriteDeckTool().set_context(ctx)
        args = {"content": {"slides": [{"id": "s1", "name": "x" * 3000}]}}
        self.assertIsNone(write.trim_args_at_edit_point(args, later_calls=[]))
        create = CreateDeckTool().set_context(ctx)
        result = json.dumps({"status": "ok", "title": "P", "deck_id": "1", "deck_json": "x" * 3000})
        self.assertIsNone(create.trim_result_at_edit_point(result, {"title": "P"}, later_calls=[]))
        ctx.deck_block_failed = False
        self.assertDropped(write.trim_args_at_edit_point(args, later_calls=[]), "content", "Active Slide Deck")
        shrunk = json.loads(create.trim_result_at_edit_point(result, {"title": "P"}, later_calls=[]))
        self.assertFieldDropped(shrunk, "deck_json", "Active Slide Deck")

    def test_document_edit_points_to_document_read(self):
        from llm.tools.registry import get_tool_registry

        tool = get_tool_registry().get_tool("document_edit")
        out = tool.trim_args_at_edit_point(
            {"doc_index": 1, "mode": "rewrite", "content": "w" * 3000}, later_calls=[],
        )
        self.assertDropped(out, "content", "document_read")
        self.assertEqual(out.args["doc_index"], 1)
        both = tool.trim_args_at_edit_point(
            {"doc_index": 1, "content": "w" * 300, "edits": [{"old_text": "a", "new_text": "b"}]},
            later_calls=[],
        )
        self.assertDropped(both, "content", "300-char content", "list of 1 edit(s)")
        self.assertNotIn("edits", both.args)
        self.assertIsNone(tool.trim_args_at_edit_point(both.args, later_calls=[]))

    def test_copied_trimmed_call_never_writes(self):
        """A model that copies a trimmed call back gets an error, never a write:
        each dropped field is required, or the tool rejects it as empty."""
        from chat.canvas_tools import EditCanvasTool, WriteCanvasTool
        from chat.scratchpad_tools import ScratchpadAppendTool
        from chat.slide_tools import EditDeckTool, WriteDeckTool

        cases = (
            (WriteCanvasTool(), {"title": "T", "content": "y" * 3000}),
            (EditCanvasTool(), {"edits": [{"old_text": "a", "new_text": "b"}]}),
            (EditDeckTool(), {"edits": [{"old_text": "a", "new_text": "b"}]}),
            (ScratchpadAppendTool(), {"note": "n" * 300}),
        )
        for tool, args in cases:
            with self.subTest(tool=tool.name):
                trimmed = tool.trim_args_at_edit_point(args, later_calls=[]).args
                with patch.object(type(tool), "_run", side_effect=AssertionError("ran")):
                    out = tool.invoke(trimmed)
                self.assertIn("NOT run", out)
        write_deck = WriteDeckTool()
        trimmed = write_deck.trim_args_at_edit_point({"content": {"slides": []}}, later_calls=[]).args
        out = json.loads(write_deck.invoke(trimmed))
        self.assertEqual(out["status"], "error")


class TrimNoteTests(TestCase):
    """The note for a trimmed call goes on its RESULT, never in an argument slot
    (WILFRED-9Q: Gemini copied ``"edits": "[Omitted…]"`` back as its edits)."""

    def test_json_result_gets_a_field_and_stays_parseable(self):
        msgs = [_tool("c1", '{"status": "ok", "applied": 3}')]
        out = attach_trim_notes(msgs, {"c1": "[Omitted from this transcript: x. y]"})
        data = json.loads(out[0].content)
        self.assertEqual(data["applied"], 3)
        self.assertEqual(data[RESULT_NOTE_FIELD], "[Omitted from this transcript: x. y]")
        self.assertIs(attach_trim_notes(out, {"c1": "[Omitted from this transcript: x. y]"}), out)  # idempotent

    def test_stub_and_text_results_get_the_note_appended(self):
        stub = "[Earlier result of canvas_edit(canvas_name=C) was cleared to save context.]"
        msgs = [_tool("c1", stub), _tool("c2", "plain"), _tool("c3", "other")]
        out = attach_trim_notes(msgs, {"c1": "NOTE1", "c2": "NOTE2 \"quoted\""})
        self.assertTrue(out[0].content.startswith(stub))  # prefix checks still hold
        self.assertTrue(out[0].content.endswith("\nNOTE1"))
        self.assertEqual(out[1].content, 'plain\nNOTE2 "quoted"')
        self.assertIs(out[2], msgs[2])
        twice = attach_trim_notes(out, {"c1": "NOTE1", "c2": "NOTE2 \"quoted\""})
        self.assertEqual([m.content for m in twice], [m.content for m in out])

    def test_history_dicts(self):
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": '{"status": "ok"}'}, {"role": "user", "content": "q"}]
        out = attach_trim_notes(msgs, {"c1": "N"})
        self.assertEqual(json.loads(out[0]["content"])[RESULT_NOTE_FIELD], "N")
        self.assertEqual(msgs[0]["content"], '{"status": "ok"}')  # not mutated

    def _req(self, messages):
        ctx = RunContext.create(user_id="1")
        return ChatRequest(
            messages=messages, model="gpt-5.4", context=ctx,
            params={"max_context_tokens": 50_000, "thinking_level": "low"},
        )

    def test_edit_point_leaves_no_marker_in_any_argument(self):
        """Regression for WILFRED-9Q: after an edit point no replayed call carries
        placeholder text in an argument, and each trimmed call's result says where
        the content lives."""
        from chat.canvas_tools import EditCanvasTool, WriteCanvasTool
        from chat.scratchpad_tools import ScratchpadAppendTool

        base = [
            Message(role="user", content="q"),
            _asst("w1", "canvas_write", {"title": "T", "content": "c" * 5000}),
            _tool("w1", '{"status": "ok", "title": "T"}'),
            _asst("e1", "canvas_edit", {"edits": [{"old_text": "c", "new_text": "d" * 3000}]}),
            _tool("e1", '{"status": "ok", "applied": 1, "failed": []}'),
            _asst("n1", "scratchpad_append", {"note": "n" * 400}),
            _tool("n1", '{"status": "ok"}'),
        ]
        req = self._req(base)
        tools = {
            "canvas_write": WriteCanvasTool(), "canvas_edit": EditCanvasTool(),
            "scratchpad_append": ScratchpadAppendTool(),
        }
        mgr = EditPointManager(SimpleChatPipeline(), req, tools)
        new = base + [_asst("s2", "web_search", {"q": 1}), _tool("s2", "R")]
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out, _ = mgr.finish_round(new, req, [], prior_len=len(base), real_input_tokens=40_000,
                                      round_call_ids={"s2"}, notice=None)
        self.assertEqual(mgr.edit_rounds, [0])
        for m in out:
            for tc in m.tool_calls or []:
                self.assertIsNone(find_echoed_marker(tc.arguments), tc.name)
        results = {m.tool_call_id: m.content for m in out if m.role == "tool"}
        self.assertIn("canvas_read", results["w1"])
        self.assertIn("list of 1 edit(s)", results["e1"])
        self.assertIn("scratchpad", results["n1"])
        self.assertEqual(results["s2"], "R")  # current round untouched

    def test_note_survives_a_later_prune_of_the_result(self):
        """A call trimmed at an earlier edit point (args dropped, note recorded on
        the assistant message) whose result a later prune turns into a stub: the
        note is re-attached to the stub."""
        note = "[Omitted from this transcript to save space: the doc. Call canvas_read.]"
        trimmed = _asst("w1", "canvas_write", {"title": "T"}).model_copy(update={"metadata": {
            TRIMMED_ARGS_KEY: ["w1"], TRIM_NOTES_KEY: {"w1": note},
        }})
        base = [
            Message(role="user", content="q"),
            trimmed, _tool("w1", json.dumps({"status": "ok", "title": "T", RESULT_NOTE_FIELD: note})),
            _asst("s1", "web_search", {"q": 1}), _tool("s1", "R1"),
        ]
        req = self._req(base)
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        new = base + [_asst("s2", "web_search", {"q": 2}), _tool("s2", "R2")]
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out, _ = mgr.finish_round(new, req, [], prior_len=len(base), real_input_tokens=40_000,
                                      round_call_ids={"s2"}, notice=None)
        self.assertEqual(mgr.edit_rounds, [0])
        w1 = next(m for m in out if m.tool_call_id == "w1")
        self.assertTrue(w1.content.startswith("[Earlier result of canvas_write(title=T)"))
        self.assertTrue(w1.content.endswith(note))
        self.assertEqual(w1.content.count("[Omitted from this transcript"), 1)


class NativeEvictionPlanTests(SimpleTestCase):
    def _img(self, size, pathway="dataroom"):
        return {"type": "image", "source": {"type": "base64", "data": "a" * size}, "_wf_pathway": pathway}

    def test_evicts_to_target_oldest_first_and_protects(self):
        from llm.core.native_limits import apply_native_evictions, native_b64_total, plan_native_evictions

        msgs = [
            Message(role="user", content=[self._img(100)]),
            Message(role="user", content=[self._img(100)]),
            Message(role="user", content=[self._img(100)]),
        ]
        self.assertEqual(native_b64_total(msgs), 300)
        plan = plan_native_evictions(msgs, 150, protect_msg_indices={0})
        self.assertEqual(plan, {(1, 0), (2, 0)})  # msg 0 protected, so the next two go
        out = apply_native_evictions(msgs, plan)
        self.assertEqual(out[1].content[0]["type"], "text")
        self.assertIs(out[0], msgs[0])

    def test_skill_assets_evicted_last(self):
        from llm.core.native_limits import plan_native_evictions

        msgs = [
            Message(role="user", content=[self._img(100, "skill")]),
            Message(role="user", content=[self._img(100)]),
        ]
        self.assertEqual(plan_native_evictions(msgs, 100), {(1, 0)})


class EditPointManagerTests(TestCase):
    def _req(self, messages, *, model="gpt-5.4", agent_kind="main"):
        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
        return ChatRequest(
            messages=messages, model=model, context=ctx,
            params={"max_context_tokens": 50_000, "thinking_level": "low"},
        )

    def test_quiet_round_is_append_only(self):
        base = [Message(role="user", content="q"), _asst("c1", "web_search", {"q": 1}), _tool("c1", "R1")]
        req = self._req(base)
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        new = base + [_asst("c2", "web_search", {"q": 2}), _tool("c2", "R2")]
        out, _ = mgr.finish_round(new, req, [], prior_len=len(base), real_input_tokens=500,
                                  round_call_ids={"c2"}, notice=None)
        self.assertEqual(out[:len(base)], base)
        self.assertEqual(mgr.edit_rounds, [])
        self.assertNotIn("edit_points_total", req.context.observability)

    def test_tool_addition_is_an_edit_point(self):
        class _Late(ContextAwareTool):
            name: str = "late_tool"
            description: str = "d"

            def _run(self, **kw):
                return ""

        base = [Message(role="user", content="q")]
        req = self._req(base)
        req.context.added_tool_names.append("late_tool")
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        registry = MagicMock()
        registry.return_value.get_tool.side_effect = lambda n: _Late() if n == "late_tool" else None
        with patch("llm.pipelines.simple_chat.get_tool_registry", registry):
            out, tools = mgr.finish_round(
                base + [_asst("c1", "x", {}), _tool("c1", "r")], req, [],
                prior_len=1, real_input_tokens=100, round_call_ids={"c1"}, notice=None,
            )
        self.assertEqual([t.name for t in tools], ["late_tool"])
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertEqual(req.context.observability["edit_point:tools"], 1)
        self.assertEqual(req.context.observability["edit_points_total"], 1)

    def test_native_overflow_evicts_to_low_watermark(self):
        img = lambda n: {"type": "image", "source": {"type": "base64", "data": "a" * n}}  # noqa: E731
        base = [Message(role="user", content=[img(400)]), Message(role="user", content=[img(400)])]
        req = self._req(base, model="anthropic/claude-opus-5")
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        new = base + [_asst("c1", "x", {}), _tool("c1", "r"), Message(role="user", content=[img(400)])]
        with patch("llm.core.native_limits.provider_native_b64_ceiling", return_value=1000):
            out, _ = mgr.finish_round(new, req, [], prior_len=len(base), real_input_tokens=100,
                                      round_call_ids={"c1"}, notice=None)
        from llm.core.native_limits import native_b64_total

        self.assertLessEqual(native_b64_total(out), 600)  # 0.6 x ceiling
        self.assertEqual(req.context.observability["edit_point:native_evict"], 1)

    def test_prune_edit_point_also_normalizes_dedup_and_trims(self):
        from chat.canvas_tools import WriteCanvasTool

        search = (
            "# Search Results\n\n## 1.\n**Document:** \"d.pdf\" [doc #1]\n"
            "**Chunk #3 of 10:**\nTHE CHUNK\n"
        )
        base = [
            Message(role="user", content="q"),
            _asst("w1", "canvas_write", {"title": "T", "content": "c" * 5000}), _tool("w1", "ok"),
            _asst("s1", "document_search", {}), _tool("s1", search),
        ]
        req = self._req(base)
        mgr = EditPointManager(SimpleChatPipeline(), req, {"canvas_write": WriteCanvasTool()})
        new = base + [_asst("s2", "document_search", {}), _tool("s2", search)]
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=1):
            out, _ = mgr.finish_round(new, req, [], prior_len=len(base), real_input_tokens=40_000,
                                      round_call_ids={"s2"}, notice=None)
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertIn("THE CHUNK", out[-1].content)  # newest copy full (keep-newest)
        self.assertNotIn("content", out[1].tool_calls[0].arguments)
        self.assertTrue(out[2].content.startswith("[Earlier result of"))  # pruned
        self.assertIn("canvas_read", out[2].content)  # the note rides on the stub
        self.assertEqual(out[2].content.count("[Omitted from this transcript"), 1)

    def test_record_thinking_drops_counts_only_known_type(self):
        ctx = RunContext.create()
        record_thinking_drops(ctx, {"input_transformations": [
            {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"},
            {"type": "something_new", "path": "x"},
        ]})
        record_thinking_drops(ctx, {"input_transformations": []})
        record_thinking_drops(ctx, None)
        self.assertEqual(ctx.observability["thinking_dropped"], 1)

    # -- the main agent's current-deck block ------------------------------

    def _deck_req(self, base, *, provider, notes=(), agent_kind="main"):
        req = self._req(base, agent_kind=agent_kind)
        req.context.deck_block_provider = provider
        req.context.scratchpad_turn_notes.extend(notes)
        return req

    def _round(self, mgr, req, base, cid, *, real_input_tokens=40_000):
        out, _ = mgr.finish_round(
            base + [_asst(cid, "web_search", {"q": cid}), _tool(cid, f"R-{cid}")], req, [],
            prior_len=len(base), real_input_tokens=real_input_tokens,
            round_call_ids={cid}, notice=None,
        )
        return out

    def test_deck_block_appended_before_scratchpad_block(self):
        from llm.pipelines.edit_points import DECK_BLOCK_KEY, SCRATCHPAD_BLOCK_KEY

        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: "# Active Slide Deck (current state)\nDECK", notes=["N"])
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out = self._round(mgr, req, base, "c1")
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertTrue(out[-2].metadata.get(DECK_BLOCK_KEY))
        self.assertIn("DECK", out[-2].content)
        self.assertTrue(out[-1].metadata.get(SCRATCHPAD_BLOCK_KEY))  # scratchpad stays last
        self.assertEqual(req.context.observability["deck_block_refreshes"], 1)
        self.assertNotIn("edit_point:deck_block", req.context.observability)

    def test_deck_block_replaced_on_next_edit_point(self):
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        state = {"v": 1}
        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: f"DECK v{state['v']}")
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out = self._round(mgr, req, base, "c1")
            state["v"] = 2
            req2 = req.model_copy(update={"messages": out})
            out2 = self._round(mgr, req2, out, "c2")
        self.assertEqual(mgr.edit_rounds, [0, 1])
        blocks = [m for m in out2 if (m.metadata or {}).get(DECK_BLOCK_KEY)]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].content, "DECK v2")
        self.assertIs(out2[-1], blocks[0])  # no scratchpad note this turn → the deck block is last
        self.assertEqual(req.context.observability["deck_block_refreshes"], 2)

    def test_deck_block_provider_empty_or_raising_adds_nothing(self):
        from contextlib import nullcontext

        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        def boom():
            raise RuntimeError("db down")

        cases = (
            ("empty", lambda: "", False),
            ("raising", boom, True),
            ("returns_none", lambda: None, True),
            ("no_provider", None, False),
        )
        for label, provider, failed in cases:
            with self.subTest(label):
                base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
                req = self._deck_req(base, provider=provider)
                mgr = EditPointManager(SimpleChatPipeline(), req, {})
                quiet = (
                    self.assertLogs("llm.pipelines.edit_points", level="WARNING")
                    if failed else nullcontext()
                )
                with quiet, self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
                    out = self._round(mgr, req, base, "c1")
                self.assertEqual(mgr.edit_rounds, [0])  # the edit point itself still happened
                self.assertFalse(any((m.metadata or {}).get(DECK_BLOCK_KEY) for m in out))
                self.assertNotIn("deck_block_refreshes", req.context.observability)
                self.assertIs(req.context.deck_block_failed, failed)

    def test_deck_block_not_added_without_an_edit(self):
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: "DECK")
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        # A quiet round is append-only: no block.
        out = self._round(mgr, req, base, "c1", real_input_tokens=500)
        self.assertFalse(any((m.metadata or {}).get(DECK_BLOCK_KEY) for m in out))
        # Over the ceiling but nothing left to change (everything protected): still no edit, no block.
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=10):
            out2 = self._round(mgr, req, base, "c1")
        self.assertFalse(any((m.metadata or {}).get(DECK_BLOCK_KEY) for m in out2))
        self.assertEqual(mgr.edit_rounds, [])
        self.assertNotIn("deck_block_refreshes", req.context.observability)

    def test_prune_trims_old_slide_write_and_appends_block(self):
        from chat.slide_tools import WriteDeckTool
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        base = [
            Message(role="user", content="q"),
            _asst("w1", "slide_canvas_write", {"content": {"slides": [{"id": "s1", "name": "x" * 5000}]}}),
            _tool("w1", "ok"),
            _asst("s1", "web_search", {"q": 1}), _tool("s1", "R1"),
        ]
        req = self._deck_req(base, provider=lambda: "# Active Slide Deck (current state)\nDECK")
        mgr = EditPointManager(SimpleChatPipeline(), req, {"slide_canvas_write": WriteDeckTool()})
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=1):
            out = self._round(mgr, req, base, "s2")
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertNotIn("content", out[1].tool_calls[0].arguments)
        self.assertIn("Active Slide Deck", out[2].content)
        self.assertTrue(out[-1].metadata.get(DECK_BLOCK_KEY))

    def test_subagent_ignores_deck_block_provider(self):
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: "DECK", agent_kind="subagent")
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out = self._round(mgr, req, base, "c1")
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertFalse(any((m.metadata or {}).get(DECK_BLOCK_KEY) for m in out))
        self.assertNotIn("deck_block_refreshes", req.context.observability)

    # -- coverage beats recency for deck payloads ---------------------------

    def test_recent_slide_write_trimmed_but_recent_canvas_write_kept(self):
        """Recency protects a recent canvas write (exact-match edits need the text)
        but not a recent deck write — the block appended here covers it."""
        from chat.canvas_tools import WriteCanvasTool
        from chat.slide_tools import WriteDeckTool
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        big_deck = {"slides": [{"id": "s1", "name": "x" * 5000}]}
        base = [
            Message(role="user", content="q"),
            _asst("d1", "slide_canvas_write", {"content": big_deck}), _tool("d1", "ok"),
            _asst("w1", "canvas_write", {"title": "T", "content": "c" * 5000}), _tool("w1", "ok"),
        ]
        req = self._deck_req(base, provider=lambda: "# Active Slide Deck (current state)\nDECK")
        mgr = EditPointManager(
            SimpleChatPipeline(), req,
            {"slide_canvas_write": WriteDeckTool(), "canvas_write": WriteCanvasTool()},
        )
        out = self._round(mgr, req, base, "s1")  # default keep_recent=6: both writes are "recent"
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertNotIn("content", out[1].tool_calls[0].arguments)
        self.assertIn("Active Slide Deck", out[2].content)
        self.assertEqual(out[3].tool_calls[0].arguments["content"], "c" * 5000)
        self.assertEqual(out[4].content, "ok")  # untrimmed call: no note
        self.assertTrue(out[-1].metadata.get(DECK_BLOCK_KEY))

    def test_create_result_kept_in_its_round_then_trimmed_next_edit_point(self):
        from chat.slide_tools import CreateDeckTool
        from llm.pipelines.edit_points import TRIMMED_RESULT_KEY

        create_result = json.dumps({
            "status": "ok", "deck_id": "7", "title": "Pitch", "slide_ids": ["s1"],
            "changed_slide_ids": ["s1"], "slide_count": 1, "deck_json": "{" + "x" * 5000 + "}",
        })
        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: "# Active Slide Deck (current state)\nDECK")
        mgr = EditPointManager(SimpleChatPipeline(), req, {"slides_create_deck": CreateDeckTool()})
        # Edit point 1: the create call IS the current round → its result stays whole.
        round1 = base + [
            _asst("k1", "slides_create_deck", {"title": "Pitch", "layouts": ["title"]}),
            _tool("k1", create_result),
        ]
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            out, _ = mgr.finish_round(round1, req, [], prior_len=len(base), real_input_tokens=40_000,
                                      round_call_ids={"k1"}, notice=None)
        self.assertEqual(mgr.edit_rounds, [0])
        self.assertEqual(out[4].content, create_result)
        # Edit point 2: now merely recent (default keep 6) → shrunk by the hook, not stubbed.
        req2 = req.model_copy(update={"messages": out})
        out2 = self._round(mgr, req2, out, "c2")
        self.assertEqual(mgr.edit_rounds, [0, 1])
        shrunk = out2[4]
        self.assertTrue(shrunk.metadata.get(TRIMMED_RESULT_KEY))
        data = json.loads(shrunk.content)
        self.assertNotIn("deck_json", data)
        self.assertIn("Active Slide Deck", data["omitted_fields_note"])
        self.assertEqual(data["slide_ids"], ["s1"])

    def test_deck_block_provider_called_once_per_edit_point(self):
        calls = {"n": 0}

        def provider():
            calls["n"] += 1
            return "DECK"

        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=provider)
        mgr = EditPointManager(SimpleChatPipeline(), req, {})
        self._round(mgr, req, base, "c1", real_input_tokens=500)  # quiet round: no edit point
        self.assertEqual(calls["n"], 0)
        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            self._round(mgr, req, base, "c1")
        self.assertEqual(calls["n"], 1)
        self.assertFalse(req.context.deck_block_failed)

    def test_deck_block_failure_keeps_previous_block_and_slide_args(self):
        """EP1 ok → EP2 the re-read fails (flag set, the last good block stays, the
        write made since keeps its JSON) → EP3 recovers (flag cleared, block
        replaced, everything trimmable trimmed)."""
        from chat.slide_tools import WriteDeckTool
        from llm.pipelines.edit_points import DECK_BLOCK_KEY

        state = {"text": "DECK v1"}
        base = [Message(role="user", content="q"), _asst("c0", "web_search", {"q": 0}), _tool("c0", "R0")]
        req = self._deck_req(base, provider=lambda: state["text"])
        tool = WriteDeckTool().set_context(req.context)
        mgr = EditPointManager(SimpleChatPipeline(), req, {"slide_canvas_write": tool})

        def deck(n):
            return {"slides": [{"id": "s1", "name": n * 5000}]}

        def call(msg_list, cid):
            return next(m for m in msg_list if m.tool_calls and m.tool_calls[0].id == cid)

        with self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=0):
            r1 = base + [_asst("d1", "slide_canvas_write", {"content": deck("a")}), _tool("d1", "ok")]
            out1, _ = mgr.finish_round(r1, req, [], prior_len=len(base), real_input_tokens=40_000,
                                       round_call_ids={"d1"}, notice=None)
            state["text"] = None
            r2 = out1 + [_asst("d2", "slide_canvas_write", {"content": deck("b")}), _tool("d2", "ok")]
            req2 = req.model_copy(update={"messages": out1})
            with self.assertLogs("llm.pipelines.edit_points", level="WARNING"):
                out2, _ = mgr.finish_round(r2, req2, [], prior_len=len(out1), real_input_tokens=40_000,
                                           round_call_ids={"d2"}, notice=None)
            self.assertTrue(req.context.deck_block_failed)
            blocks = [m for m in out2 if (m.metadata or {}).get(DECK_BLOCK_KEY)]
            self.assertEqual([b.content for b in blocks], ["DECK v1"])
            self.assertEqual(call(out2, "d1").tool_calls[0].arguments["content"], deck("a"))  # kept
            state["text"] = "DECK v3"
            r3 = out2 + [_asst("c3", "web_search", {"q": 3}), _tool("c3", "R3")]
            req3 = req.model_copy(update={"messages": out2})
            out3, _ = mgr.finish_round(r3, req3, [], prior_len=len(out2), real_input_tokens=40_000,
                                       round_call_ids={"c3"}, notice=None)
        self.assertFalse(req.context.deck_block_failed)
        blocks = [m for m in out3 if (m.metadata or {}).get(DECK_BLOCK_KEY)]
        self.assertEqual([b.content for b in blocks], ["DECK v3"])
        self.assertNotIn("content", call(out3, "d1").tool_calls[0].arguments)
        self.assertNotIn("content", call(out3, "d2").tool_calls[0].arguments)
        results = {m.tool_call_id: m.content for m in out3 if m.role == "tool"}
        self.assertIn("replaced", results["d1"])  # d2 rewrote the same deck
        self.assertIn("Active Slide Deck", results["d2"])
        self.assertEqual(mgr.edit_rounds, [0, 1, 2])


# ---------------------------------------------------------------------------
# End-to-end: each request extends the previous one except at edit points
# ---------------------------------------------------------------------------


class _SearchInput(BaseModel):
    query: str = Field(default="", description="q")


class _DupSearchTool(ContextAwareTool):
    """Always returns the same chunk, so every round after the first is a duplicate."""

    name: str = "document_search"
    description: str = "search"
    args_schema: type[BaseModel] = _SearchInput

    def _run(self, query: str = "", **kw) -> str:
        return (
            "# Search Results\n\n## 1.\n**Document:** \"d.pdf\" [doc #1]\n"
            "**Chunk #3 of 10:**\nTHE SAME CHUNK " + ("filler " * 50) + "\n"
        )


class _NoteInput(BaseModel):
    note: str = Field(default="", description="n")


class _NoteTool(ContextAwareTool):
    name: str = "scratchpad_append"
    description: str = "note"
    args_schema: type[BaseModel] = _NoteInput

    def _run(self, note: str = "", **kw) -> str:
        if self.context is not None:
            self.context.scratchpad = (self.context.scratchpad + "\n" + note).strip()
            self.context.scratchpad_turn_notes.append(note)
        return '{"status": "ok"}'


class _FlaggedNoteTool(_NoteTool):
    """A note tool that opts out of recency protection and trims its note."""

    trim_despite_recency: bool = True
    trim_args_min_chars: int = 50

    def trim_args_at_edit_point(self, args, *, later_calls):
        return {**args, "note": "[trimmed]"}


def _snapshot(req: ChatRequest):
    """What the provider would be sent: system/messages (content, calls, ids) + tool set."""
    msgs = [
        (m.role, json.dumps(m.content, sort_keys=True, default=str), m.tool_call_id,
         json.dumps([tc.model_dump() for tc in (m.tool_calls or [])], sort_keys=True))
        for m in req.messages
    ]
    return sorted(t.name for t in (req.tool_schemas or [])), msgs


class PrefixInvariantTests(TestCase):
    ROUNDS = 5

    def _registry(self, *, flagged_note: bool = False):
        note_tool = _FlaggedNoteTool() if flagged_note else _NoteTool()
        tools = {"document_search": _DupSearchTool(), "scratchpad_append": note_tool,
                 "subagent_scratchpad_append": _NoteTool(name="subagent_scratchpad_append")}
        reg = MagicMock()
        reg.return_value.get_tool.side_effect = lambda n: tools.get(n)
        return patch("llm.pipelines.simple_chat.get_tool_registry", reg)

    def _calls(self, i):
        note_tool = "subagent_scratchpad_append" if self._kind == "subagent" else "scratchpad_append"
        return [
            {"id": f"s{i}", "name": "document_search", "arguments": {"query": "x"}},
            {"id": f"n{i}", "name": note_tool, "arguments": {"note": f"finding {i} " + "y" * 300}},
        ]

    def _run(
        self, *, stream: bool, agent_kind: str, pressure_round: int | None = None,
        truncated_round: int | None = None, deck_block: bool = False,
        flagged_note: bool = False, keep_recent: int = 2,
    ):
        self._kind = agent_kind
        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
        if deck_block:
            ctx.deck_block_provider = lambda: "# Active Slide Deck (current state)\nDECK"
        tool_names = ["document_search", "subagent_scratchpad_append" if agent_kind == "subagent" else "scratchpad_append"]
        request = ChatRequest(
            messages=[Message(role="system", content="sys"), Message(role="user", content="go")],
            model="gpt-5.4", stream=stream, tools=tool_names, context=ctx,
            params={"max_context_tokens": 50_000, "thinking_level": "low"},
        )
        seen: list = []
        counter = {"i": 0}

        def input_tokens(i):
            return 40_000 if i == pressure_round else 1_000

        def next_turn(req):
            seen.append(_snapshot(req))
            i = counter["i"]
            counter["i"] += 1
            return i

        if stream:
            def fake_stream(req):
                i = next_turn(req)
                data = {"content": "", "input_tokens": input_tokens(i)}
                if i < self.ROUNDS:
                    data["tool_calls"] = self._calls(i)
                else:
                    data["content"] = "final"
                if i == truncated_round:
                    data["output_truncated"] = True
                    data["stop_reason"] = "repetition"

                def gen():
                    yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id="")
                    yield StreamEvent(event_type="message_end", data=data, sequence=2, run_id="")
                return gen()

            fake_model = MagicMock()
            fake_model.stream.side_effect = fake_stream
        else:
            def fake_generate(req):
                i = next_turn(req)
                calls = [ToolCall(**c) for c in self._calls(i)] if i < self.ROUNDS else None
                return ChatResponse(
                    message=Message(role="assistant", content="" if calls else "final", tool_calls=calls),
                    model="gpt-5.4",
                    usage=Usage(prompt_tokens=input_tokens(i), completion_tokens=10, total_tokens=0),
                    metadata=(
                        {"output_truncated": True, "stop_reason": "repetition"}
                        if i == truncated_round else {}
                    ),
                )

            fake_model = MagicMock()
            fake_model.generate.side_effect = fake_generate

        with patch("llm.pipelines.simple_chat.create_chat_model", return_value=fake_model), \
             self._registry(flagged_note=flagged_note), \
             self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=keep_recent):
            pipe = SimpleChatPipeline(max_tool_iterations=self.ROUNDS + 1)
            if stream:
                list(pipe.stream(request))
            else:
                pipe.run(request)
        return seen, ctx

    def _assert_extends(self, seen, edit_after: set):
        for i in range(1, len(seen)):
            prev_tools, prev_msgs = seen[i - 1]
            tools, msgs = seen[i]
            if (i - 1) in edit_after:
                continue  # round i-1 ended in an edit point
            self.assertEqual(tools, prev_tools, f"tool set changed before call {i}")
            self.assertEqual(msgs[:len(prev_msgs)], prev_msgs, f"prefix edited before call {i}")

    def test_quiet_loop_never_edits_history(self):
        for stream in (True, False):
            for kind in ("main", "subagent"):
                with self.subTest(stream=stream, agent_kind=kind):
                    seen, ctx = self._run(stream=stream, agent_kind=kind)
                    self.assertEqual(len(seen), self.ROUNDS + 1)
                    self._assert_extends(seen, set())
                    self.assertEqual(ctx.observability.get("edit_points_total", 0), 0)
                    # Duplicates were still deduped — append-only, in the new copy.
                    last_msgs = seen[-1][1]
                    self.assertTrue(any("earlier tool result above" in m[1] for m in last_msgs))
                    # The final call kept its tools bound (tool_choice="none").
                    self.assertEqual(seen[-1][0], seen[-2][0])

    def test_truncated_round_stays_append_only(self):
        # A cut-off round appends an assistant message, one error result per
        # skipped call and a user notice — never an edit of earlier history.
        for stream in (True, False):
            for kind in ("main", "subagent"):
                with self.subTest(stream=stream, agent_kind=kind):
                    seen, ctx = self._run(stream=stream, agent_kind=kind, truncated_round=1)
                    self.assertEqual(len(seen), self.ROUNDS + 1)
                    self._assert_extends(seen, set())
                    self.assertEqual(ctx.observability.get("edit_points_total", 0), 0)
                    self.assertEqual(ctx.observability["output_truncated_rounds"], 1)
                    after = seen[2][1]
                    self.assertTrue(any("NOT run" in m[1] for m in after))
                    self.assertTrue(any("cut off" in m[1] for m in after))

    def test_pressure_round_is_the_only_edit(self):
        for stream in (True, False):
            for kind in ("main", "subagent"):
                with self.subTest(stream=stream, agent_kind=kind):
                    seen, ctx = self._run(stream=stream, agent_kind=kind, pressure_round=2)
                    self._assert_extends(seen, {2})
                    self.assertEqual(ctx.observability["edit_points_total"], 1)
                    self.assertEqual(ctx.observability["edit_point:prune"], 1)
                    # The edit really rewrote history (otherwise the test is vacuous).
                    before, after = seen[2][1], seen[3][1]
                    self.assertNotEqual(after[:len(before)], before)

    def test_pressure_round_with_deck_block_is_the_only_edit(self):
        """The current-deck block is appended INSIDE the edit point (never on a
        quiet round) and then rides along unchanged — one edit, one cache break."""
        for stream in (True, False):
            with self.subTest(stream=stream):
                seen, ctx = self._run(stream=stream, agent_kind="main", pressure_round=2, deck_block=True)
                self._assert_extends(seen, {2})
                self.assertEqual(ctx.observability["edit_points_total"], 1)
                self.assertEqual(ctx.observability["deck_block_refreshes"], 1)
                has_block = [any("DECK" in m[1] for m in msgs) for _tools, msgs in seen]
                self.assertEqual(has_block, [False, False, False, True, True, True])

    def test_pressure_round_with_flagged_tool_is_the_only_edit(self):
        """A tool that opts out of recency protection still only ever edits history
        INSIDE the edit point: one edit, where its recent note args are trimmed
        (keep_recent=4 makes round 1's note "recent" at the round-2 edit)."""
        for stream in (True, False):
            with self.subTest(stream=stream):
                seen, ctx = self._run(
                    stream=stream, agent_kind="main", pressure_round=2,
                    flagged_note=True, keep_recent=4,
                )
                self._assert_extends(seen, {2})
                self.assertEqual(ctx.observability["edit_points_total"], 1)
                self.assertFalse(any("[trimmed]" in m[3] for m in seen[2][1]))
                after = seen[3][1]
                trimmed_notes = [m for m in after if "[trimmed]" in m[3]]
                self.assertEqual(len(trimmed_notes), 2)  # n0 (old) and n1 (recent, flagged)


# ---------------------------------------------------------------------------
# Provider opt-ins
# ---------------------------------------------------------------------------


class ToolChoiceKwargsTests(SimpleTestCase):
    def _req(self, choice=None):
        params = {"_tool_choice": choice} if choice else {}
        return ChatRequest(messages=[Message(role="user", content="x")], model="m", params=params)

    def test_anthropic_uses_dict_form(self):
        from llm.core.providers.anthropic import AnthropicChatModel

        m = AnthropicChatModel("anthropic/claude-opus-5", MagicMock())
        self.assertEqual(m._tool_choice_kwargs(self._req("none")), {"tool_choice": {"type": "none"}})
        self.assertEqual(m._tool_choice_kwargs(self._req()), {})

    def test_openai_and_gemini_use_string(self):
        from llm.core.providers.gemini import GeminiChatModel
        from llm.core.providers.openai import OpenAIChatModel

        for cls, name in ((OpenAIChatModel, "gpt-5.4"), (GeminiChatModel, "gemini/gemini-3.1-pro-preview")):
            m = cls(name, MagicMock())
            self.assertEqual(m._tool_choice_kwargs(self._req("none")), {"tool_choice": "none"})

    def test_bind_tools_receives_tool_choice(self):
        from llm.core.providers.openai import OpenAIChatModel

        client = MagicMock()
        m = OpenAIChatModel("gpt-5.4", client)
        req = ChatRequest(messages=[Message(role="user", content="x")], model="gpt-5.4",
                          tool_schemas=[MagicMock()], params={"_tool_choice": "none"})
        m._get_streaming_client(req)
        self.assertEqual(client.bind_tools.call_args.kwargs.get("tool_choice"), "none")


class PreservedThinkingOptInTests(SimpleTestCase):
    def test_only_post_2026_10_01_models_are_flagged(self):
        from llm.model_registry import _MODELS

        self.assertEqual(
            [k for k, v in _MODELS.items() if v.binds_thinking_to_prefix],
            ["anthropic/claude-haiku-5-5"],
        )

    def test_flag_only_valid_on_adaptive_anthropic(self):
        from llm.model_registry import _MODELS

        for model_id, info in _MODELS.items():
            if info.binds_thinking_to_prefix:
                self.assertEqual(info.provider, "anthropic", model_id)
                self.assertEqual(info.thinking_mode, "adaptive", model_id)

    def test_flagged_model_gets_beta_header_and_drop_block(self):
        from llm.core import model_factory
        from llm.core.providers.anthropic import AnthropicChatModel

        with patch("llm.core.model_factory.anthropic_binds_thinking", return_value=True), \
             patch("llm.core.providers.anthropic.anthropic_binds_thinking", return_value=True), \
             patch("llm.core.providers.anthropic.create_variant_client") as variant:
            kwargs = model_factory._get_provider_kwargs("anthropic", "claude-opus-5-5")
            self.assertEqual(kwargs["betas"], [model_factory.ANTHROPIC_THINKING_BINDING_BETA])
            m = AnthropicChatModel("anthropic/claude-opus-5-5", MagicMock())
            m._get_reasoning_client(ChatRequest(
                messages=[], model="anthropic/claude-opus-5-5", params={"thinking_level": "high"},
            ))
            thinking = variant.call_args.kwargs["thinking"]
            self.assertEqual(thinking["block_binding"], {"prefix_mismatch_behavior": "drop_block"})
            # No level: still sends thinking explicitly with the binding.
            m._get_reasoning_client(ChatRequest(messages=[], model="anthropic/claude-opus-5-5", params={}))
            self.assertIn("block_binding", variant.call_args.kwargs["thinking"])

    def test_unflagged_model_unchanged(self):
        from llm.core import model_factory

        kwargs = model_factory._get_provider_kwargs("anthropic", "claude-opus-5-5")
        self.assertNotIn("betas", kwargs)

    def test_stream_chunk_keeps_input_transformations(self):
        from llm.core.providers.anthropic_client import WilfredChatAnthropic

        # Guard the private hook we override (pinned to langchain-anthropic 1.5.x).
        self.assertTrue(hasattr(WilfredChatAnthropic, "_make_message_chunk_from_anthropic_event"))
        client = WilfredChatAnthropic(model="claude-opus-5", api_key="test")
        event = SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(model="claude-opus-5", input_transformations=[
                {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"},
            ]),
        )
        msg, _ = client._make_message_chunk_from_anthropic_event(
            event, stream_usage=True, coerce_content_to_string=False,
        )
        self.assertEqual(msg.response_metadata["input_transformations"][0]["type"], "thinking_dropped")


class ObservabilityEditPointFieldTests(SimpleTestCase):
    def test_fields_mapped(self):
        from llm.service.logger import _observability_fields

        ctx = RunContext.create()
        ctx.bump_stat("tool_calls", 3)
        ctx.bump_stat("edit_points_total", 2)
        ctx.bump_stat("edit_point:prune", 2)
        ctx.bump_stat("edit_point:tools", 1)
        ctx.bump_stat("thinking_dropped", 4)
        out = _observability_fields(ctx)
        self.assertEqual(out["edit_point_count"], 2)
        self.assertEqual(out["edit_points"], {"prune": 2, "tools": 1})
        self.assertEqual(out["thinking_dropped_count"], 4)

    def test_no_tools_means_null(self):
        from llm.service.logger import _observability_fields

        out = _observability_fields(RunContext.create())
        self.assertIsNone(out["edit_point_count"])
        self.assertIsNone(out["edit_points"])
        self.assertIsNone(out["thinking_dropped_count"])
