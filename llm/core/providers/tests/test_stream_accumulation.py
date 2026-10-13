"""Streaming chunk accumulation in BaseLangChainChatModel.

The stream loop collects chunks and merges them once at the end (via
``add_ai_message_chunks``) instead of ``accumulated + chunk`` per token. These
tests lock in that the single-pass merge still yields the correct final content,
tool calls, and usage in the ``message_end`` event.
"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from llm.core.providers.base import BaseLangChainChatModel
from llm.types.messages import Message
from llm.types.requests import ChatRequest


def _make_model(chunks):
    client = MagicMock()
    client.stream = MagicMock(return_value=list(chunks))
    client.bind_tools = MagicMock(return_value=client)
    model = BaseLangChainChatModel(model_name="test-model", client=client)
    model._provider_label = "Test"
    return model


def _request():
    return ChatRequest(messages=[Message(role="user", content="hi")], model="test-model")


class StreamAccumulationTests(TestCase):
    def test_multi_chunk_text_merges_into_message_end(self):
        chunks = [
            AIMessageChunk(content="Hel"),
            AIMessageChunk(content="lo "),
            AIMessageChunk(
                content="world",
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            ),
        ]
        events = list(_make_model(chunks).stream(_request()))

        # Live UI stream: one token event per chunk (independent of the merge).
        tokens = [e.data.get("text") for e in events if e.event_type == "token"]
        self.assertEqual(tokens, ["Hel", "lo ", "world"])

        end = next(e for e in events if e.event_type == "message_end")
        self.assertEqual(end.data["content"], "Hello world")
        self.assertEqual(end.data["output_tokens"], 4)
        self.assertEqual(end.data["input_tokens"], 10)

    def test_tool_call_chunks_merge(self):
        # Tool-call args split across chunks, as providers stream them.
        chunks = [
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "get_weather", "args": '{"ci', "id": "call_1", "index": 0}
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": 'ty": "Paris"}', "id": None, "index": 0}
                ],
                usage_metadata={"input_tokens": 7, "output_tokens": 9, "total_tokens": 16},
            ),
        ]
        events = list(_make_model(chunks).stream(_request()))

        end = next(e for e in events if e.event_type == "message_end")
        tool_calls = end.data.get("tool_calls")
        self.assertTrue(tool_calls, "merged tool_calls should be present in message_end")
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["arguments"], {"city": "Paris"})

    def test_normal_stream_is_not_truncated(self):
        chunks = [
            AIMessageChunk(
                content="done",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            ),
        ]
        end = next(
            e for e in _make_model(chunks).stream(_request()) if e.event_type == "message_end"
        )
        self.assertFalse(end.data["output_truncated"])
        self.assertEqual(end.data["stop_reason"], "stop")

    def test_provider_cap_hit_marks_output_truncated(self):
        cases = {
            "anthropic": ({"stop_reason": "max_tokens"}, "max_tokens"),
            "chat_completions": ({"finish_reason": "length"}, "length"),
            "gemini_concatenated": ({"finish_reason": "STOPMAX_TOKENS"}, "STOPMAX_TOKENS"),
            "responses_api": (
                {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
                "max_output_tokens",
            ),
        }
        for label, (meta, expected_stop) in cases.items():
            with self.subTest(label=label):
                chunks = [
                    AIMessageChunk(
                        content="partial",
                        response_metadata=meta,
                        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    ),
                ]
                end = next(
                    e for e in _make_model(chunks).stream(_request())
                    if e.event_type == "message_end"
                )
                self.assertTrue(end.data["output_truncated"])
                self.assertEqual(end.data["stop_reason"], expected_stop)
                # The provider's own usage is kept (nothing was estimated).
                self.assertEqual(end.data["output_tokens"], 2)
                self.assertNotIn("usage_estimated", end.data)

    def test_content_filter_incomplete_is_not_truncated(self):
        chunks = [
            AIMessageChunk(
                content="",
                response_metadata={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                },
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            ),
        ]
        end = next(
            e for e in _make_model(chunks).stream(_request()) if e.event_type == "message_end"
        )
        self.assertFalse(end.data["output_truncated"])


class _RecordingStream:
    """A provider stream generator that records how far it was consumed and
    whether the pipeline closed it (as closing the SDK's HTTP stream would)."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.consumed = 0
        self.closed = False

    def __call__(self, *args, **kwargs):
        return self._gen()

    def _gen(self):
        try:
            for chunk in self.chunks:
                self.consumed += 1
                yield chunk
        finally:
            self.closed = True


def _make_generator_model(chunks):
    stream = _RecordingStream(chunks)
    client = MagicMock()
    client.stream = stream
    client.bind_tools = MagicMock(return_value=client)
    model = BaseLangChainChatModel(model_name="test-model", client=client)
    model._provider_label = "Test"
    return model, stream


def _pipe_loop_chunks(n_fragments: int = 80, frag_chars: int = 400):
    head = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": "canvas_edit", "args": '{"edits": [{"old_text": "x", "new_text": "',
             "id": "call_1", "index": 0}
        ],
    )
    frags = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": " |" * (frag_chars // 2), "id": None, "index": 0}],
        )
        for _ in range(n_fragments)
    ]
    tail = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": None, "args": '"}]}', "id": None, "index": 0}],
        usage_metadata={"input_tokens": 50, "output_tokens": 9_999, "total_tokens": 10_049},
        response_metadata={"finish_reason": "stop"},
    )
    return [head, *frags, tail]


class StreamOutputGuardTests(TestCase):
    def test_degenerate_tool_args_abort_emits_truncated_message_end(self):
        chunks = _pipe_loop_chunks()
        model, stream = _make_generator_model(chunks)

        events = list(model.stream(_request()))

        self.assertFalse([e for e in events if e.event_type == "error"])
        end = next(e for e in events if e.event_type == "message_end")
        self.assertTrue(end.data["output_truncated"])
        self.assertEqual(end.data["stop_reason"], "repetition")
        # Provider usage never arrived (the final chunk was not consumed), so it
        # is estimated rather than logged as NULL.
        self.assertTrue(end.data["usage_estimated"])
        self.assertGreater(end.data["output_tokens"], 0)
        self.assertGreater(end.data["input_tokens"], 0)
        self.assertNotEqual(end.data["output_tokens"], 9_999)
        # The partial call is still reported (the pipeline decides not to run it).
        self.assertEqual(end.data["tool_calls"][0]["name"], "canvas_edit")
        # The provider stream was closed early and never drained.
        self.assertTrue(stream.closed)
        self.assertLess(stream.consumed, len(chunks))

    def test_abort_logs_one_warning(self):
        model, _stream = _make_generator_model(_pipe_loop_chunks())
        with self.assertLogs("llm.core.providers.base", level="WARNING") as logs:
            list(model.stream(_request()))
        aborted = [r for r in logs.records if "aborted for degenerate output" in r.getMessage()]
        self.assertEqual(len(aborted), 1)
        quiet = [r for r in logs.records if "No usage_metadata" in r.getMessage()]
        self.assertEqual(quiet, [])

    def test_clean_stream_is_fully_consumed_and_not_closed_early(self):
        chunks = [
            AIMessageChunk(content="Hel"),
            AIMessageChunk(
                content="lo",
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
        ]
        model, stream = _make_generator_model(chunks)
        end = next(e for e in model.stream(_request()) if e.event_type == "message_end")
        self.assertEqual(stream.consumed, 2)
        self.assertFalse(end.data["output_truncated"])
        self.assertEqual(end.data["output_tokens"], 2)
