"""OpenAI Responses: several text messages in one response (commentary, then
final_answer) stream a ``text_break`` and join with a blank line."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase
from langchain_core.messages import AIMessageChunk

from llm.core.providers.openai import OpenAIChatModel
from llm.service.llm_service import _StreamCollapser
from llm.service.stream_log import StreamLogAccumulator
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.streaming import StreamEvent


def _request():
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        stream=True,
        model="openai/gpt-6-luna",
        context=RunContext.create(),
    )


def _text(text, index, **extra):
    return AIMessageChunk(content=[{"type": "text", "text": text, "index": index, **extra}])


def _stream(chunks):
    client = MagicMock()
    client.stream.return_value = iter(chunks)
    model = OpenAIChatModel("openai/gpt-6-luna", client)
    with patch.object(OpenAIChatModel, "_get_streaming_client", return_value=client):
        return list(model.stream(_request()))


class OpenAITextSegmentTests(SimpleTestCase):
    def test_second_message_item_streams_a_break_and_joins_with_blank_line(self):
        events = _stream([
            _text("", 0, phase="commentary"),
            _text("Checking the sources.", 0),
            _text("", 1, phase="final_answer"),
            _text("The report ", 1),
            _text("is done.", 1),
        ])
        kinds = [e.event_type for e in events]
        self.assertEqual(
            kinds, ["message_start", "token", "text_break", "token", "token", "message_end"],
        )
        self.assertEqual(
            events[-1].data["content"], "Checking the sources.\n\nThe report is done.",
        )

    def test_single_message_item_has_no_break(self):
        events = _stream([_text("Hello ", 0), _text("world.", 0)])
        self.assertNotIn("text_break", [e.event_type for e in events])
        self.assertEqual(events[-1].data["content"], "Hello world.")

    def test_text_then_tool_call_has_no_break(self):
        events = _stream([
            _text("Looking it up.", 0),
            AIMessageChunk(
                content=[],
                tool_call_chunks=[{"name": "web_search", "args": "{}", "id": "c1", "index": 1}],
            ),
        ])
        self.assertNotIn("text_break", [e.event_type for e in events])
        self.assertEqual(events[-1].data["content"], "Looking it up.")

    def test_plain_string_content_has_no_break(self):
        events = _stream([AIMessageChunk(content="Hi "), AIMessageChunk(content="there")])
        self.assertNotIn("text_break", [e.event_type for e in events])
        self.assertEqual(events[-1].data["content"], "Hi there")

    def test_generate_joins_message_items_with_blank_line(self):
        model = OpenAIChatModel("openai/gpt-6-luna", MagicMock())
        joined = model._join_text_blocks([
            {"type": "text", "text": "Note.", "id": "msg_1"},
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": "Answer.", "id": "msg_2"},
        ])
        self.assertEqual(joined, "Note.\n\nAnswer.")


class StreamFoldTests(SimpleTestCase):
    def _events(self):
        return [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id="r"),
            StreamEvent(event_type="token", data={"text": "Note."}, sequence=2, run_id="r"),
            StreamEvent(event_type="text_break", data={}, sequence=3, run_id="r"),
            StreamEvent(event_type="token", data={"text": "Answer."}, sequence=4, run_id="r"),
            StreamEvent(event_type="message_end", data={}, sequence=5, run_id="r"),
        ]

    def test_collapser_keeps_only_the_final_answer(self):
        collapser = _StreamCollapser()
        for event in self._events():
            collapser.feed(event)
        self.assertEqual("".join(collapser._text_parts), "Answer.")

    def test_log_accumulator_keeps_both_parts(self):
        acc = StreamLogAccumulator()
        for event in self._events():
            acc.add(event)
        self.assertEqual("".join(acc._text_parts), "Note.\n\nAnswer.")
