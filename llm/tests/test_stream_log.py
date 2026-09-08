"""Tests for llm.service.stream_log — the fold-as-you-go log accumulator."""

import json
import os
from unittest.mock import patch

from django.test import SimpleTestCase

from llm.service.stream_log import (
    LOG_TEXT_PREVIEW_CHARS,
    StreamLogAccumulator,
    cap_text,
    full_payloads_enabled,
    slim_response_raw_output,
)
from llm.types.messages import Message, ToolCall
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


def _ev(event_type, data, seq=1):
    return StreamEvent(event_type=event_type, data=data, sequence=seq, run_id="r1")


class StreamLogAccumulatorTests(SimpleTestCase):
    def test_token_text_accumulates_across_message_starts(self):
        """The log fold joins ALL turns' text — no reset on message_start
        (unlike the run_via_stream collapse, which keeps only the last turn)."""
        acc = StreamLogAccumulator.from_events([
            _ev("message_start", {}),
            _ev("token", {"text": "first "}),
            _ev("message_start", {}),
            _ev("token", {"text": "second"}),
        ])
        self.assertEqual(acc.streamed_text, "first second")

    def test_first_message_end_wins(self):
        acc = StreamLogAccumulator.from_events([
            _ev("message_end", {"input_tokens": 10}),
            _ev("message_end", {"input_tokens": 99}),
        ])
        self.assertEqual(acc.end_data["input_tokens"], 10)

    def test_empty_data_message_end_still_terminal(self):
        acc = StreamLogAccumulator.from_events([_ev("message_end", {})])
        self.assertTrue(acc.has_end_event)
        self.assertEqual(acc.end_data, {})

    def test_no_message_end_not_terminal(self):
        acc = StreamLogAccumulator.from_events([_ev("token", {"text": "x"})])
        self.assertFalse(acc.has_end_event)

    def test_first_error_wins(self):
        acc = StreamLogAccumulator.from_events([
            _ev("error", {"error_code": "overloaded"}),
            _ev("error", {"error_code": "later"}),
        ])
        self.assertEqual(acc.error_data["error_code"], "overloaded")

    def test_tool_pairing_and_end_without_start(self):
        acc = StreamLogAccumulator.from_events([
            _ev("tool_start", {"tool_call_id": "a", "tool_name": "search",
                               "arguments": {"q": "x"}}),
            _ev("tool_end", {"tool_call_id": "a", "result": "found it"}),
            _ev("tool_end", {"tool_call_id": "b", "tool_name": "orphan",
                             "result": "late"}),
        ])
        parsed = json.loads(acc.build_raw_output())
        by_id = {t["tool_call_id"]: t for t in parsed["tool_calls"]}
        self.assertEqual(by_id["a"]["tool_name"], "search")
        self.assertEqual(by_id["a"]["result_size"], len("found it"))
        self.assertEqual(by_id["a"]["result_preview"], "found it")
        self.assertGreater(by_id["a"]["args_size"], 0)
        self.assertEqual(by_id["b"]["tool_name"], "orphan")
        self.assertEqual(by_id["b"]["result_size"], len("late"))

    def test_large_tool_result_only_preview_retained(self):
        big = "r" * 100_000
        acc = StreamLogAccumulator.from_events([
            _ev("tool_end", {"tool_call_id": "a", "tool_name": "big", "result": big}),
        ])
        raw = acc.build_raw_output()
        self.assertLess(len(raw), 2000)
        parsed = json.loads(raw)
        self.assertEqual(parsed["tool_calls"][0]["result_size"], 100_000)

    def test_counts_and_event_tracking(self):
        acc = StreamLogAccumulator()
        self.assertFalse(acc.has_events)
        acc.add(_ev("token", {"text": "abc"}))
        acc.add(_ev("message_end", {}))
        self.assertTrue(acc.has_events)
        self.assertEqual(acc.event_count, 2)
        parsed = json.loads(acc.build_raw_output())
        self.assertEqual(parsed["counts"], {"events": 2, "text_chars": 3, "tool_calls": 0})

    def test_full_payload_flag_raises_text_cap(self):
        long_text = "z" * (LOG_TEXT_PREVIEW_CHARS * 4)
        acc = StreamLogAccumulator.from_events([_ev("token", {"text": long_text})])
        default_parsed = json.loads(acc.build_raw_output())
        self.assertLess(len(default_parsed["final_text_preview"]), LOG_TEXT_PREVIEW_CHARS + 100)
        with patch.dict(os.environ, {"LLM_LOG_FULL_PAYLOADS": "true"}):
            self.assertTrue(full_payloads_enabled())
            full_parsed = json.loads(acc.build_raw_output())
        self.assertEqual(full_parsed["final_text_preview"], long_text)


class CapTextTests(SimpleTestCase):
    def test_under_cap_untouched(self):
        self.assertEqual(cap_text("short", 10), "short")

    def test_over_cap_truncated_with_marker(self):
        result = cap_text("a" * 2000, 100)
        self.assertTrue(result.startswith("a" * 100))
        self.assertIn("2,000 chars total", result)


class SlimResponseRawOutputTests(SimpleTestCase):
    def test_summarizes_response(self):
        response = ChatResponse(
            message=Message(role="assistant", content="c" * 3000, tool_calls=[
                ToolCall(id="t1", name="search", arguments={"q": "x"}),
            ]),
            model="m",
            usage=None,
            metadata={},
        )
        parsed = json.loads(slim_response_raw_output(response))
        self.assertLess(len(parsed["final_text_preview"]), 600)
        self.assertEqual(parsed["counts"]["text_chars"], 3000)
        self.assertEqual(parsed["tool_calls"][0]["tool_name"], "search")
