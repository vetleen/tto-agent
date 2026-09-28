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
    EditPointManager,
    TRIMMED_ARGS_KEY,
    record_thinking_drops,
    trim_history_call_args,
    trim_old_tool_call_args,
)
from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.tools.interfaces import ContextAwareTool
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
        self.assertEqual(out, {"a": {"content": "[trimmed]"}})


class ToolTrimHookTests(SimpleTestCase):
    """The per-tool trim rules (content must stay reachable)."""

    def test_canvas_write_and_edit_point_to_canvas_read(self):
        from chat.canvas_tools import EditCanvasTool, WriteCanvasTool

        w = WriteCanvasTool().trim_args_at_edit_point({"title": "T", "content": "y" * 3000}, later_calls=[])
        self.assertIn("canvas_read", w["content"])
        self.assertIn("Omitted from this transcript", w["content"])
        self.assertEqual(w["title"], "T")
        e = EditCanvasTool().trim_args_at_edit_point(
            {"edits": [{"old_text": "a", "new_text": "b"}]}, later_calls=[],
        )
        self.assertIn("canvas_read", e["edits"])
        # Idempotent: an already-trimmed write is left alone.
        self.assertIsNone(WriteCanvasTool().trim_args_at_edit_point(w, later_calls=[]))

    def test_subagent_canvas_only_when_superseded(self):
        from chat.subagent_canvas_tools import SubagentCanvasEditTool, SubagentCanvasWriteTool

        args = {"content": "z" * 3000}
        self.assertIsNone(SubagentCanvasWriteTool().trim_args_at_edit_point(args, later_calls=[]))
        later = [ToolCall(id="w2", name="subagent_canvas_write", arguments={})]
        self.assertIn("replaced", SubagentCanvasWriteTool().trim_args_at_edit_point(args, later_calls=later)["content"])
        edit_args = {"edits": [{"old_text": "a", "new_text": "b"}]}
        self.assertIsNone(SubagentCanvasEditTool().trim_args_at_edit_point(edit_args, later_calls=[]))

    def test_slide_write_only_when_same_deck_rewritten(self):
        from chat.slide_tools import WriteDeckTool

        args = {"content": {"slides": []}, "deck_name": "Pitch"}
        other = [ToolCall(id="x", name="slide_canvas_write", arguments={"deck_name": "Other"})]
        same = [ToolCall(id="y", name="slide_canvas_write", arguments={"deck_name": "Pitch"})]
        self.assertIsNone(WriteDeckTool().trim_args_at_edit_point(args, later_calls=other))
        self.assertIn("replaced", WriteDeckTool().trim_args_at_edit_point(args, later_calls=same)["content"])

    def test_document_edit_points_to_document_read(self):
        from llm.tools.registry import get_tool_registry

        tool = get_tool_registry().get_tool("document_edit")
        out = tool.trim_args_at_edit_point(
            {"doc_index": 1, "mode": "rewrite", "content": "w" * 3000}, later_calls=[],
        )
        self.assertIn("document_read", out["content"])
        self.assertEqual(out["doc_index"], 1)


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
        self.assertIn("canvas_read", out[1].tool_calls[0].arguments["content"])
        self.assertTrue(out[2].content.startswith("[Earlier result of"))  # pruned

    def test_record_thinking_drops_counts_only_known_type(self):
        ctx = RunContext.create()
        record_thinking_drops(ctx, {"input_transformations": [
            {"type": "thinking_dropped", "path": "messages.1.content.0", "reason": "prefix_binding_mismatch"},
            {"type": "something_new", "path": "x"},
        ]})
        record_thinking_drops(ctx, {"input_transformations": []})
        record_thinking_drops(ctx, None)
        self.assertEqual(ctx.observability["thinking_dropped"], 1)


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

    def _registry(self):
        tools = {"document_search": _DupSearchTool(), "scratchpad_append": _NoteTool(),
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

    def _run(self, *, stream: bool, agent_kind: str, pressure_round: int | None = None):
        self._kind = agent_kind
        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
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
                    metadata={},
                )

            fake_model = MagicMock()
            fake_model.generate.side_effect = fake_generate

        with patch("llm.pipelines.simple_chat.create_chat_model", return_value=fake_model), \
             self._registry(), self.settings(CONTEXT_MIDTURN_KEEP_TOOL_RESULTS=2):
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
    def test_no_current_model_is_flagged(self):
        from llm.model_registry import _MODELS

        self.assertEqual([k for k, v in _MODELS.items() if v.binds_thinking_to_prefix], [])

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
