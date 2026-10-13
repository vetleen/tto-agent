"""Tool-loop handling of a truncated round (provider output cap hit, or the
stream guard aborted a degenerate response).

A truncated round's tool calls are never executed — their arguments may have
been silently repaired from cut-off JSON — each is paired with a synthetic
error result, the model gets a short notice, and the loop continues. Two in a
row fall through to the final tool-less call. A main-agent round whose visible
answer was cut ends the turn as usual.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase
from pydantic import BaseModel, Field

from llm.pipelines.simple_chat import SimpleChatPipeline, _TRUNCATED_TOOL_LABEL
from llm.tools.interfaces import ContextAwareTool
from llm.types.context import RunContext
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

CALLS: list = []


class _Input(BaseModel):
    a: float = Field(description="a")
    b: float = Field(description="b")


class _RecordingTool(ContextAwareTool):
    name: str = "document_search"
    description: str = "records calls"
    args_schema: type[BaseModel] = _Input

    def _run(self, a: float, b: float) -> str:
        CALLS.append((a, b))
        return '{"result": 5}'


def _call(i: int = 1) -> dict:
    return {"id": f"c{i}", "name": "document_search", "arguments": {"a": 1, "b": 2}}


def _stream_of(data: dict):
    def gen():
        yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id="")
        yield StreamEvent(event_type="message_end", data=dict(data), sequence=2, run_id="")
    return gen()


class TruncatedRoundTests(TestCase):
    def setUp(self):
        CALLS.clear()
        self.tool = _RecordingTool()
        registry = MagicMock()
        registry.return_value.get_tool.side_effect = (
            lambda n: self.tool if n == "document_search" else None
        )
        self._registry = patch("llm.pipelines.simple_chat.get_tool_registry", registry)

    def _request(self, *, stream: bool, agent_kind: str = "main") -> ChatRequest:
        ctx = RunContext.create(user_id="1")
        ctx.agent_kind = agent_kind
        return ChatRequest(
            messages=[Message(role="user", content="go")],
            stream=stream, model="gpt-5.4", tools=["document_search"], context=ctx,
            params={"max_context_tokens": 200_000},
        )

    def _run_stream(self, request, datas, max_iterations=50):
        seen = []

        def fake_stream(req):
            seen.append(req)
            return _stream_of(datas[len(seen) - 1])

        fake_model = MagicMock()
        fake_model.stream.side_effect = fake_stream
        with patch("llm.pipelines.simple_chat.create_chat_model", return_value=fake_model), \
             self._registry:
            events = list(SimpleChatPipeline(max_tool_iterations=max_iterations).stream(request))
        return events, seen

    # -- streaming -------------------------------------------------------

    def test_truncated_tool_round_is_not_executed_and_pairs_results(self):
        request = self._request(stream=True)
        events, seen = self._run_stream(request, [
            {"content": "", "output_truncated": True, "stop_reason": "repetition",
             "tool_calls": [_call()], "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            {"content": "final", "input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
        ])
        self.assertEqual(CALLS, [])
        self.assertEqual(len(seen), 2)

        starts = [e for e in events if e.event_type == "tool_start"]
        ends = [e for e in events if e.event_type == "tool_end"]
        self.assertEqual([e.data["tool_call_id"] for e in starts], ["c1"])
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0].data["is_error"])
        self.assertEqual(ends[0].data["display_label"], _TRUNCATED_TOOL_LABEL)
        self.assertIn("NOT run", ends[0].data["result"])

        msgs = seen[1].messages
        asst = next(i for i, m in enumerate(msgs) if m.role == "assistant" and m.tool_calls)
        self.assertEqual(msgs[asst].tool_calls[0].id, "c1")
        self.assertEqual(msgs[asst + 1].role, "tool")
        self.assertEqual(msgs[asst + 1].tool_call_id, "c1")
        self.assertIn("NOT run", msgs[asst + 1].content)
        self.assertEqual(msgs[asst + 2].role, "user")
        self.assertIn("cut off", msgs[asst + 2].content)
        self.assertIn("repetition", msgs[asst + 2].content)

        obs = request.context.observability
        self.assertEqual(obs["output_truncated_rounds"], 1)
        self.assertEqual(obs["tool_calls"], 1)
        end = [e for e in events if e.event_type == "message_end"][-1]
        self.assertEqual(end.data["output_tokens"], 8)

    def test_subagent_truncated_round_without_calls_appends_notice_only(self):
        request = self._request(stream=True, agent_kind="subagent")
        _events, seen = self._run_stream(request, [
            {"content": "garbage", "output_truncated": True, "stop_reason": "max_tokens",
             "input_tokens": 10, "output_tokens": 5},
            {"content": "final", "input_tokens": 20, "output_tokens": 3},
        ])
        self.assertEqual(len(seen), 2)
        msgs = seen[1].messages
        self.assertFalse(any(m.role == "assistant" for m in msgs))
        self.assertEqual(msgs[-1].role, "user")
        self.assertIn("cut off", msgs[-1].content)
        self.assertIn("[Progress:", msgs[-1].content)
        self.assertEqual(request.context.observability["output_truncated_rounds"], 1)

    def test_main_agent_cut_visible_answer_ends_the_turn(self):
        request = self._request(stream=True)
        events, seen = self._run_stream(request, [
            {"content": "partial answer", "output_truncated": True, "stop_reason": "length",
             "input_tokens": 10, "output_tokens": 5},
        ])
        self.assertEqual(len(seen), 1)
        end = [e for e in events if e.event_type == "message_end"][-1]
        self.assertTrue(end.data["output_truncated"])
        self.assertEqual(end.data["stop_reason"], "length")
        self.assertEqual(request.context.observability["output_truncated_rounds"], 1)

    def test_main_agent_cut_thinking_round_continues(self):
        request = self._request(stream=True)
        _events, seen = self._run_stream(request, [
            {"content": "", "output_truncated": True, "stop_reason": "max_output_tokens"},
            {"content": "final", "input_tokens": 1, "output_tokens": 1},
        ])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1].messages[-1].role, "user")
        self.assertIn("cut off", seen[1].messages[-1].content)

    def test_two_consecutive_truncated_rounds_break_to_final_stream(self):
        request = self._request(stream=True)
        with self.assertLogs("llm.pipelines.simple_chat", level="WARNING") as logs:
            _events, seen = self._run_stream(request, [
                {"content": "", "output_truncated": True, "stop_reason": "repetition",
                 "tool_calls": [_call(1)]},
                {"content": "", "output_truncated": True, "stop_reason": "repetition",
                 "tool_calls": [_call(2)]},
                {"content": "final", "input_tokens": 1, "output_tokens": 1},
            ])
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[2].params.get("_tool_choice"), "none")
        self.assertIn("tool-use limit", seen[2].messages[-1].content)
        self.assertTrue(any("consecutive truncated" in r.getMessage() for r in logs.records))
        self.assertEqual(request.context.observability["output_truncated_rounds"], 2)
        self.assertEqual(CALLS, [])

    def test_normal_round_resets_the_streak(self):
        request = self._request(stream=True)
        _events, seen = self._run_stream(request, [
            {"content": "", "output_truncated": True, "stop_reason": "repetition",
             "tool_calls": [_call(1)]},
            {"content": "", "tool_calls": [_call(2)]},
            {"content": "", "output_truncated": True, "stop_reason": "repetition",
             "tool_calls": [_call(3)]},
            {"content": "final", "input_tokens": 1, "output_tokens": 1},
        ])
        self.assertEqual(len(seen), 4)
        self.assertNotEqual(seen[3].params.get("_tool_choice"), "none")
        self.assertEqual(CALLS, [(1, 2)])

    def test_calls_without_a_name_are_dropped(self):
        request = self._request(stream=True)
        _events, seen = self._run_stream(request, [
            {"content": "", "output_truncated": True, "stop_reason": "repetition",
             "tool_calls": [{"id": "c1", "name": "", "arguments": {}}]},
            {"content": "final", "input_tokens": 1, "output_tokens": 1},
        ])
        self.assertEqual(len(seen), 2)
        msgs = seen[1].messages
        self.assertFalse(any(m.role in ("assistant", "tool") for m in msgs))
        self.assertIn("cut off", msgs[-1].content)

    # -- non-streaming ---------------------------------------------------

    def _run_generate(self, request, responses):
        seen = []

        def fake_generate(req):
            seen.append(req)
            return responses[len(seen) - 1]

        fake_model = MagicMock()
        fake_model.generate.side_effect = fake_generate
        with patch("llm.pipelines.simple_chat.create_chat_model", return_value=fake_model), \
             self._registry:
            response = SimpleChatPipeline().run(request)
        return response, seen

    def test_generate_truncated_tool_round_is_not_executed(self):
        request = self._request(stream=False)
        truncated = ChatResponse(
            message=Message(role="assistant", content="", tool_calls=[ToolCall(**_call())]),
            model="gpt-5.4",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            metadata={"output_truncated": True, "stop_reason": "max_tokens"},
        )
        final = ChatResponse(
            message=Message(role="assistant", content="final"), model="gpt-5.4",
            usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23), metadata={},
        )
        response, seen = self._run_generate(request, [truncated, final])
        self.assertEqual(CALLS, [])
        self.assertEqual(len(seen), 2)
        msgs = seen[1].messages
        asst = next(i for i, m in enumerate(msgs) if m.role == "assistant" and m.tool_calls)
        self.assertEqual(msgs[asst + 1].role, "tool")
        self.assertIn("NOT run", msgs[asst + 1].content)
        self.assertEqual(msgs[asst + 2].role, "user")
        self.assertIn("cut off", msgs[asst + 2].content)
        self.assertEqual(response.message.content, "final")
        self.assertEqual(response.usage.completion_tokens, 8)
        self.assertEqual(request.context.observability["output_truncated_rounds"], 1)

    def test_generate_main_agent_cut_visible_answer_ends_the_turn(self):
        request = self._request(stream=False)
        truncated = ChatResponse(
            message=Message(role="assistant", content="partial"), model="gpt-5.4",
            usage=None, metadata={"output_truncated": True, "stop_reason": "length"},
        )
        response, seen = self._run_generate(request, [truncated])
        self.assertEqual(len(seen), 1)
        self.assertEqual(response.message.content, "partial")
        self.assertTrue(response.metadata["output_truncated"])
