"""Per-tool timeouts in the sub-agent tool loop (llm/pipelines/simple_chat.py).

A sub-agent's round waits while any call is within its own ``timeout_seconds``;
unfinished calls then get a timeout result and are abandoned. A call that
returns later is discarded — only *when* it returned is recorded, for timeout
calibration. The main agent never times out.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from pydantic import BaseModel

from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.service.logger import _observability_fields
from llm.tools.interfaces import ContextAwareTool
from llm.types import ChatRequest, ChatResponse, Message, RunContext, ToolCall


class _NoArgs(BaseModel):
    pass


class _SleepyTool(ContextAwareTool):
    """Sleeps ``delay`` seconds (or until ``release`` is set), then returns its name."""

    name: str = "slow_tool"
    description: str = "sleeps"
    args_schema: type[BaseModel] = _NoArgs
    delay: float = 0.0
    release: threading.Event | None = None

    def _run(self) -> str:
        if self.release is not None:
            self.release.wait(timeout=5)
        else:
            time.sleep(self.delay)
        return json.dumps({"ok": self.name})


def _ctx(agent_kind="subagent"):
    ctx = RunContext.create()
    ctx.agent_kind = agent_kind
    return ctx


def _call(cid, name):
    return ToolCall(id=cid, name=name, arguments={})


class RoundWithTimeoutsTests(SimpleTestCase):
    def test_fast_timeout_waits_for_slower_sibling_within_its_timeout(self):
        """A (timeout 0.1 s) is still running when B (timeout 2 s) needs 0.4 s:
        the round waits for B, then abandons A — not at 0.1 s."""
        a = _SleepyTool(name="a_tool", delay=1.5, timeout_seconds=0.1)
        b = _SleepyTool(name="b_tool", delay=0.4, timeout_seconds=2)
        ctx = _ctx()
        t0 = time.monotonic()
        results = SimpleChatPipeline._execute_tool_calls(
            [_call("1", "a_tool"), _call("2", "b_tool")], {"a_tool": a, "b_tool": b}, ctx,
        )
        elapsed = time.monotonic() - t0
        self.assertGreaterEqual(elapsed, 0.35)
        self.assertLess(elapsed, 1.4)
        a_result = json.loads(results[0][1])
        self.assertEqual(a_result["error"], "tool_timeout")
        self.assertIn("timed out after 0 s", a_result["message"])
        self.assertEqual(json.loads(results[1][1]), {"ok": "b_tool"})
        self.assertEqual(ctx.observability["tool_timeouts"][0]["tool"], "a_tool")

    def test_single_slow_call_times_out(self):
        tool = _SleepyTool(delay=1.0, timeout_seconds=0.2)
        ctx = _ctx()
        with self.assertLogs("llm.pipelines.simple_chat", level="WARNING") as logs:
            results = SimpleChatPipeline._execute_tool_calls(
                [_call("1", "slow_tool")], {"slow_tool": tool}, ctx,
            )
        self.assertEqual(json.loads(results[0][1])["error"], "tool_timeout")
        self.assertTrue(any("timed out" in r.getMessage() for r in logs.records))
        fields = _observability_fields(ctx)
        self.assertEqual(fields["tool_timeout_count"], 1)
        self.assertEqual(fields["tool_timeouts"][0]["tool"], "slow_tool")
        self.assertIsNone(fields["tool_timeouts"][0]["late_after_s"])

    def test_main_agent_has_no_timeout(self):
        tool = _SleepyTool(delay=0.3, timeout_seconds=0.05)
        ctx = _ctx("main")
        results = SimpleChatPipeline._execute_tool_calls(
            [_call("1", "slow_tool")], {"slow_tool": tool}, ctx,
        )
        self.assertEqual(json.loads(results[0][1]), {"ok": "slow_tool"})
        self.assertNotIn("tool_timeouts", ctx.observability)

    def test_no_timeouts_when_calls_finish(self):
        tool = _SleepyTool(delay=0.0, timeout_seconds=5)
        ctx = _ctx()
        results = SimpleChatPipeline._execute_tool_calls(
            [_call("1", "slow_tool"), _call("2", "slow_tool")], {"slow_tool": tool}, ctx,
        )
        self.assertEqual([json.loads(r)["ok"] for _, r in results], ["slow_tool", "slow_tool"])
        self.assertEqual(_observability_fields(ctx)["tool_timeout_count"], None)

    def test_subagent_call_runs_under_its_timeout_as_the_call_deadline(self):
        """The tool's retry/throttle waits see the per-tool timeout as their
        deadline on the worker thread; the main agent's calls see none."""
        from llm.tools._throttle import call_deadline_remaining

        seen = []

        class _Peek(_SleepyTool):
            def _run(self) -> str:
                seen.append(call_deadline_remaining())
                return json.dumps({"ok": self.name})

        tool = _Peek(timeout_seconds=7)
        SimpleChatPipeline._execute_tool_calls([_call("1", "slow_tool")], {"slow_tool": tool}, _ctx())
        self.assertIsNotNone(seen[0])
        self.assertGreater(seen[0], 6.0)
        self.assertLessEqual(seen[0], 7.0)
        SimpleChatPipeline._execute_tool_calls(
            [_call("1", "slow_tool")], {"slow_tool": tool}, _ctx("main"),
        )
        self.assertIsNone(seen[1])
        self.assertIsNone(call_deadline_remaining())  # never leaks onto this thread


def _wait_for_late_return(record, seconds=2.0):
    deadline = time.monotonic() + seconds
    while record.get("late_after_s") is None and time.monotonic() < deadline:
        time.sleep(0.01)


class LateReturnTests(SimpleTestCase):
    def test_late_return_is_recorded_and_discarded(self):
        """An abandoned call that returns later fills in ``late_after_s`` on its
        timeout record (calibration) and nothing else — its result is dropped."""
        release = threading.Event()
        tool = _SleepyTool(release=release, timeout_seconds=0.1)
        ctx = _ctx()
        results = SimpleChatPipeline._execute_tool_calls(
            [_call("1", "slow_tool")], {"slow_tool": tool}, ctx,
        )
        self.assertEqual(json.loads(results[0][1])["error"], "tool_timeout")
        self.assertIn("discarded", json.loads(results[0][1])["message"])
        record = ctx.observability["tool_timeouts"][0]
        self.assertIsNone(record["late_after_s"])
        release.set()
        _wait_for_late_return(record)
        self.assertIsInstance(record["late_after_s"], int)
        self.assertFalse(hasattr(ctx, "pending_late_tool_results"))

    def test_late_return_never_reaches_the_history(self):
        """Round 1's slow call times out and returns during round 2; round 3's
        request carries no trace of it (append-only, and no late text)."""
        release = threading.Event()
        slow = _SleepyTool(name="slow_tool", release=release, timeout_seconds=0.1)
        fast = _SleepyTool(name="fast_tool", delay=0.0)
        ctx = _ctx()
        request = ChatRequest(
            messages=[Message(role="user", content="Go")],
            stream=False,
            model="gpt-4o-mini",
            tools=["slow_tool", "fast_tool"],
            context=ctx,
        )

        def _round(name, cid):
            return ChatResponse(
                message=Message(role="assistant", content="", tool_calls=[_call(cid, name)]),
                model="gpt-4o-mini", usage=None, metadata={},
            )

        def _generate(req):
            n = fake_model.generate.call_count
            if n == 2:
                # The abandoned slow call returns while round 2 is being decided.
                release.set()
                _wait_for_late_return(ctx.observability["tool_timeouts"][0])
            return [
                _round("slow_tool", "c1"),
                _round("fast_tool", "c2"),
                ChatResponse(message=Message(role="assistant", content="Done."),
                             model="gpt-4o-mini", usage=None, metadata={}),
            ][n - 1]

        fake_model = MagicMock()
        fake_model.generate.side_effect = _generate
        registry = MagicMock()
        registry.return_value.get_tool.side_effect = {"slow_tool": slow, "fast_tool": fast}.get
        with patch("llm.pipelines.simple_chat.create_chat_model", return_value=fake_model), \
             patch("llm.pipelines.simple_chat.get_tool_registry", registry):
            response = SimpleChatPipeline(max_tool_iterations=5).run(request)

        self.assertEqual(response.message.content, "Done.")
        round2 = fake_model.generate.call_args_list[1][0][0].messages
        round3 = fake_model.generate.call_args_list[2][0][0].messages
        self.assertEqual(round3[:len(round2)], round2)
        self.assertFalse(any('{"ok": "slow_tool"}' in str(m.content) for m in round3))
        self.assertFalse(any("Late tool result" in str(m.content) for m in round3))
        self.assertIsNotNone(ctx.observability["tool_timeouts"][0]["late_after_s"])
