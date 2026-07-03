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
