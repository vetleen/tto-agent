"""Tests for llm.service.logger — log_call, log_stream, log_error."""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm.models import LLMCallLog
from llm.service.logger import log_call, log_error, log_stream
from llm.service.stream_log import StreamLogAccumulator
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.messages import ToolCall
from llm.types.streaming import StreamEvent

User = get_user_model()

# log_stream takes a fold-as-you-go accumulator instead of the raw event list;
# tests still build event lists for readability and wrap them here.
_acc = StreamLogAccumulator.from_events


def _make_request(model="gpt-4o-mini", user_id=None, stream=False, conversation_id=None):
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        stream=stream,
        model=model,
        context=RunContext.create(user_id=user_id, conversation_id=conversation_id),
    )


def _get_log(request):
    """Fetch the LLMCallLog entry matching the request's run_id."""
    return LLMCallLog.objects.get(run_id=request.context.run_id)


class LogCallTests(TestCase):
    """Tests for log_call (non-streaming success)."""

    def test_creates_success_entry(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi!"),
            model="gpt-4o-mini",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8, cost_usd=0.001),
            metadata={},
        )
        log_call(request, response, duration_ms=150)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertEqual(log.model, "gpt-4o-mini")
        self.assertFalse(log.is_stream)
        # raw_output = slim summary JSON (never the full response)
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["final_text_preview"], "Hi!")
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(log.input_tokens, 5)
        self.assertEqual(log.output_tokens, 3)
        self.assertEqual(log.total_tokens, 8)
        self.assertEqual(log.cost_usd, Decimal("0.001"))
        self.assertEqual(log.duration_ms, 150)
        self.assertEqual(log.run_id, request.context.run_id)

    def test_serializes_prompt_messages(self):
        request = ChatRequest(
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hi"),
            ],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        response = ChatResponse(
            message=Message(role="assistant", content="Ok"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(len(log.prompt), 2)
        self.assertEqual(log.prompt[0]["role"], "system")
        self.assertEqual(log.prompt[1]["content"], "Hi")

    def test_handles_no_usage(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=100)

        log = _get_log(request)
        self.assertIsNone(log.input_tokens)
        self.assertIsNone(log.output_tokens)
        self.assertIsNone(log.total_tokens)
        self.assertIsNone(log.cost_usd)

    def test_resolves_user_fk(self):
        user = User.objects.create_user(email="log@example.com", password="pass")
        request = _make_request(user_id=str(user.pk))
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.user, user)

    def test_no_user_when_user_id_missing(self):
        request = _make_request(user_id=None)
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertIsNone(log.user)

    def test_no_user_when_user_id_invalid(self):
        request = _make_request(user_id="nonexistent-999")
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertIsNone(log.user)

    def test_never_raises_on_db_error(self):
        """Logging failures must be swallowed, not propagated."""
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        with patch.object(LLMCallLog.objects, "create", side_effect=RuntimeError("DB is down")):
            # Should not raise
            log_call(request, response, duration_ms=50)

    def test_handles_no_context(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="gpt-4o-mini",
            context=None,
        )
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = LLMCallLog.objects.filter(run_id="").first()
        self.assertIsNotNone(log)
        self.assertIsNone(log.user)

    def test_stores_tools_from_request(self):
        from unittest.mock import MagicMock

        tool1 = MagicMock()
        tool1.name = "document_search"
        tool1.description = "Search docs"
        tool2 = MagicMock()
        tool2.name = "document_read"
        tool2.description = "Read a doc"

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            tool_schemas=[tool1, tool2],
            context=RunContext.create(),
        )
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(len(log.tools), 2)
        self.assertEqual(log.tools[0]["name"], "document_search")
        self.assertEqual(log.tools[1]["name"], "document_read")

    def test_tools_none_when_no_tool_schemas(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertIsNone(log.tools)

    def test_populates_tracing_fields(self):
        request = _make_request(conversation_id="conv-123")
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.trace_id, request.context.trace_id)
        self.assertEqual(log.conversation_id, "conv-123")

    def test_populates_response_metadata(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={
                "response_metadata": {"stop_reason": "end_turn", "model_id": "claude-sonnet-4-6"},
                "stop_reason": "end_turn",
                "provider_model_id": "claude-sonnet-4-6",
            },
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.stop_reason, "end_turn")
        self.assertEqual(log.provider_model_id, "claude-sonnet-4-6")
        self.assertIsNotNone(log.response_metadata)

    def test_populates_extended_usage_fields(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=Usage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                cached_tokens=30, reasoning_tokens=10, cost_usd=0.001,
            ),
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.cached_tokens, 30)
        self.assertEqual(log.reasoning_tokens, 10)

    def test_serializes_tool_calls_in_prompt(self):
        request = ChatRequest(
            messages=[
                Message(role="assistant", content="", tool_calls=[
                    ToolCall(id="tc1", name="search", arguments={"q": "test"}),
                ]),
                Message(role="tool", content="result", tool_call_id="tc1"),
            ],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        response = ChatResponse(
            message=Message(role="assistant", content="Done"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.prompt[0]["tool_calls"][0]["name"], "search")
        self.assertEqual(log.prompt[1]["tool_call_id"], "tc1")


class LogStreamTests(TestCase):
    """Tests for log_stream (streaming success)."""

    def test_creates_success_entry_with_concatenated_output(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Hel"}, sequence=2, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "lo!"}, sequence=3, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=4, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=300)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertTrue(log.is_stream)
        # raw_output = slim summary JSON built from the folded events
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["final_text_preview"], "Hello!")
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(log.duration_ms, 300)
        # No usage in message_end data → still None
        self.assertIsNone(log.input_tokens)
        self.assertIsNone(log.cost_usd)

    def test_empty_events_produces_empty_output(self):
        request = _make_request(stream=True)
        log_stream(request, _acc([]), duration_ms=10)

        log = _get_log(request)
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["final_text_preview"], "")
        self.assertEqual(parsed["tool_calls"], [])

    def test_stores_tools_from_request_in_stream(self):
        from unittest.mock import MagicMock

        tool = MagicMock()
        tool.name = "document_search"
        tool.description = "Search docs"

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=True,
            model="gpt-4o-mini",
            tool_schemas=[tool],
            context=RunContext.create(),
        )
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=2, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=200)

        log = _get_log(request)
        self.assertEqual(len(log.tools), 1)
        self.assertEqual(log.tools[0]["name"], "document_search")

    def test_tools_none_when_no_tool_schemas_in_stream(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=1, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=100)

        log = _get_log(request)
        self.assertIsNone(log.tools)

    def test_extracts_usage_from_message_end(self):
        """log_stream should populate token/cost fields from message_end data."""
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=2, run_id=run_id),
            StreamEvent(
                event_type="message_end",
                data={
                    "model": "gpt-5-mini",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                    "cost_usd": 0.000125,
                },
                sequence=3,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=200)

        log = _get_log(request)
        self.assertEqual(log.input_tokens, 100)
        self.assertEqual(log.output_tokens, 50)
        self.assertEqual(log.total_tokens, 150)
        self.assertEqual(log.cost_usd, Decimal("0.000125"))

    def test_missing_usage_in_message_end_stays_none(self):
        """Backward compat: empty message_end data → None for all usage fields."""
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_end", data={"model": "gpt-5-mini"}, sequence=1, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=50)

        log = _get_log(request)
        self.assertIsNone(log.input_tokens)
        self.assertIsNone(log.output_tokens)
        self.assertIsNone(log.total_tokens)
        self.assertIsNone(log.cost_usd)

    def test_populates_tracing_fields_for_stream(self):
        request = _make_request(stream=True, conversation_id="conv-456")
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_end", data={}, sequence=1, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.trace_id, request.context.trace_id)
        self.assertEqual(log.conversation_id, "conv-456")

    def test_populates_response_metadata_for_stream(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(
                event_type="message_end",
                data={
                    "model": "claude-sonnet-4-6",
                    "response_metadata": {"stop_reason": "end_turn"},
                    "stop_reason": "end_turn",
                    "provider_model_id": "claude-sonnet-4-6",
                    "cached_tokens": 25,
                    "reasoning_tokens": 15,
                },
                sequence=1,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.stop_reason, "end_turn")
        self.assertEqual(log.provider_model_id, "claude-sonnet-4-6")
        self.assertEqual(log.cached_tokens, 25)
        self.assertEqual(log.reasoning_tokens, 15)

    def test_never_raises_on_db_error(self):
        request = _make_request(stream=True)
        with patch.object(LLMCallLog.objects, "create", side_effect=RuntimeError("DB is down")):
            log_stream(request, _acc([]), duration_ms=10)

    def test_error_event_without_message_end_writes_error_row(self):
        """Providers convert stream exceptions into ``error`` events and return,
        so ``LLMService.stream`` never sees the exception. log_stream must
        recognise this and write an ERROR row rather than a bogus SUCCESS."""
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(
                event_type="error",
                data={
                    "message": "The request could not be processed.",
                    "error_code": "invalid_request",
                    "details": "Malformed url parameter. Must be either an image URL or base64...",
                },
                sequence=2,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=120)

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertTrue(log.is_stream)
        self.assertEqual(log.error_type, "invalid_request")
        self.assertIn("Malformed url parameter", log.error_message)
        self.assertEqual(log.duration_ms, 120)

    def test_error_event_default_error_type_when_code_missing(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(
                event_type="error",
                data={"message": "boom"},
                sequence=1,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=10)

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertEqual(log.error_type, "stream_error")
        self.assertEqual(log.error_message, "boom")

    def test_error_event_with_trailing_message_end_still_logs_error(self):
        """Defensive: if both an error event and a message_end are present,
        the row must be ERROR (not SUCCESS), but usage from the message_end
        is kept so spend stays visible."""
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(
                event_type="error",
                data={"message": "boom", "error_code": "provider_error"},
                sequence=2,
                run_id=run_id,
            ),
            StreamEvent(
                event_type="message_end",
                data={
                    "input_tokens": 100,
                    "output_tokens": 40,
                    "total_tokens": 140,
                    "cost_usd": 0.0005,
                },
                sequence=3,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=80)

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertEqual(log.error_type, "provider_error")
        self.assertEqual(log.input_tokens, 100)
        self.assertEqual(log.output_tokens, 40)
        self.assertEqual(log.cost_usd, Decimal("0.0005"))


class CancelledStreamLoggingTests(TestCase):
    """Interrupted streams (no message_end) estimate output tokens so cost stays
    visible; explicit user cancellations are logged with status CANCELLED."""

    def _request(self, cancel_event=None, cancel_check=None, model="gpt-5.4-mini"):
        params = {}
        if cancel_event is not None:
            params["_cancel_event"] = cancel_event
        if cancel_check is not None:
            params["_cancel_check"] = cancel_check
        return ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=True,
            model=model,
            context=RunContext.create(),
            params=params,
        )

    def _interrupted_events(self, run_id):
        # Tokens streamed but NO terminal message_end — the shape of a stream
        # the consumer stopped mid-flight.
        return [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Lets do this. Lets go. "}, sequence=2, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Let me know what you think. "}, sequence=3, run_id=run_id),
        ]

    def test_cancelled_stream_logged_cancelled_with_estimated_usage(self):
        import threading

        ev = threading.Event()
        ev.set()
        request = self._request(cancel_event=ev)
        log_stream(request, _acc(self._interrupted_events(request.context.run_id)), duration_ms=200)

        log = _get_log(request)
        self.assertEqual(log.status, "cancelled")
        self.assertIsNotNone(log.output_tokens)
        self.assertGreater(log.output_tokens, 0)
        # Priced model → cost computed from the estimated output tokens.
        self.assertIsNotNone(log.cost_usd)

    def test_interrupted_without_cancel_event_stays_success(self):
        request = self._request(cancel_event=None)
        log_stream(request, _acc(self._interrupted_events(request.context.run_id)), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        # Tokens are still estimated so a network-dropped stream isn't free.
        self.assertIsNotNone(log.output_tokens)

    def test_unset_cancel_event_stays_success(self):
        import threading

        ev = threading.Event()  # created but NOT set
        request = self._request(cancel_event=ev)
        log_stream(request, _acc(self._interrupted_events(request.context.run_id)), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.status, "success")

    def test_completed_stream_not_marked_cancelled_even_with_cancel_event(self):
        """A stream that reached message_end is SUCCESS even if a (set)
        cancel_event is present — cancellation requires a missing terminal."""
        import threading

        ev = threading.Event()
        ev.set()
        request = self._request(cancel_event=ev)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=1, run_id=run_id),
            StreamEvent(
                event_type="message_end",
                data={"input_tokens": 5, "output_tokens": 1},
                sequence=2,
                run_id=run_id,
            ),
        ]
        log_stream(request, _acc(events), duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertEqual(log.output_tokens, 1)

    def test_subagent_cancel_check_true_logged_cancelled(self):
        """Sub-agent runs cancel via a _cancel_check callable (DB poll), not a
        threading.Event. An interrupted stream whose _cancel_check() returns True
        must be logged CANCELLED (was previously SUCCESS)."""
        request = self._request(cancel_check=lambda: True)
        log_stream(request, _acc(self._interrupted_events(request.context.run_id)), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.status, "cancelled")

    def test_subagent_cancel_check_false_stays_success(self):
        request = self._request(cancel_check=lambda: False)
        log_stream(request, _acc(self._interrupted_events(request.context.run_id)), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.status, "success")


class LogStreamErrorUsageTests(TestCase):
    """A stream that ends in an error event but also carries a usage-only
    message_end (emitted by the tool loop so mid-loop cost isn't lost) must be
    logged ERROR — the error event takes precedence — WITH the usage."""

    def test_error_with_usage_message_end_logs_error_and_usage(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="error", data={
                "message": "Provider overloaded", "error_code": "overloaded",
            }, sequence=2, run_id=run_id),
            StreamEvent(event_type="message_end", data={
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            }, sequence=3, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=100)

        log = _get_log(request)
        self.assertEqual(log.status, "error")            # error event wins
        self.assertEqual(log.input_tokens, 100)          # usage still logged
        self.assertEqual(log.output_tokens, 20)


class LogErrorTests(TestCase):
    """Tests for log_error."""

    def test_creates_error_entry(self):
        request = _make_request()
        exc = ValueError("Bad input")
        log_error(request, exc, duration_ms=50)

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertEqual(log.error_type, "ValueError")
        self.assertEqual(log.error_message, "Bad input")
        self.assertEqual(log.raw_output, "")
        self.assertEqual(log.duration_ms, 50)

    def test_is_stream_flag_passed_through(self):
        request = _make_request(stream=True)
        exc = TimeoutError("Timed out")
        log_error(request, exc, duration_ms=100, is_stream=True)

        log = _get_log(request)
        self.assertTrue(log.is_stream)

    def test_never_raises_on_db_error(self):
        request = _make_request()
        with patch.object(LLMCallLog.objects, "create", side_effect=RuntimeError("DB is down")):
            log_error(request, ValueError("x"), duration_ms=10)


class SerializeMessagesTests(TestCase):
    """Tests for _serialize_messages including base64 truncation."""

    def test_truncates_base64_in_content(self):
        from llm.service.logger import _serialize_messages

        big_b64 = "A" * 10000
        request = ChatRequest(
            messages=[Message(role="user", content=[
                {"type": "text", "text": "Look at this:"},
                {"type": "image", "base64": big_b64},
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        # The base64 field should be replaced with a placeholder
        img_block = result[0]["content"][1]
        self.assertIn("10000 chars", img_block["base64"])

    def test_truncates_data_uri_in_image_url(self):
        from llm.service.logger import _serialize_messages

        big_uri = "data:image/png;base64," + "A" * 10000
        request = ChatRequest(
            messages=[Message(role="user", content=[
                {"type": "image_url", "image_url": {"url": big_uri}},
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        img_block = result[0]["content"][0]
        self.assertIn("data URI", img_block["image_url"]["url"])

    def test_preserves_normal_image_url(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="user", content=[
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        self.assertEqual(result[0]["content"][0]["image_url"]["url"], "https://example.com/img.png")

    def test_string_content_unchanged(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        self.assertEqual(result[0]["content"], "Hello")

    def test_long_string_content_capped_with_marker(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="user", content="x" * 5000)],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        content = result[0]["content"]
        self.assertLess(len(content), 600)
        self.assertIn("5,000 chars total", content)

    def test_long_text_block_capped(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="user", content=[
                {"type": "text", "text": "y" * 3000},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        text_block = result[0]["content"][0]
        self.assertLess(len(text_block["text"]), 600)
        self.assertIn("3,000 chars total", text_block["text"])
        # Non-text blocks pass through untouched
        self.assertEqual(result[0]["content"][1]["image_url"]["url"], "https://example.com/a.png")

    def test_large_tool_arguments_replaced_with_summary(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tc1", name="canvas_write", arguments={"content": "z" * 5000}),
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        tc = result[0]["tool_calls"][0]
        self.assertEqual(tc["name"], "canvas_write")
        self.assertIsInstance(tc["arguments"], str)
        self.assertIn("tool args truncated", tc["arguments"])

    def test_small_tool_arguments_kept_as_dict(self):
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tc1", name="search", arguments={"q": "test"}),
            ])],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        result = _serialize_messages(request)
        self.assertEqual(result[0]["tool_calls"][0]["arguments"], {"q": "test"})

    def test_full_payloads_flag_raises_caps(self):
        import os
        from llm.service.logger import _serialize_messages

        request = ChatRequest(
            messages=[Message(role="user", content="x" * 5000)],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict(os.environ, {"LLM_LOG_FULL_PAYLOADS": "true"}):
            result = _serialize_messages(request)
        self.assertEqual(result[0]["content"], "x" * 5000)


class SlimRawOutputTests(TestCase):
    """raw_output carries a compact tool summary, never full results."""

    def test_stream_raw_output_summarizes_tool_calls(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        big_result = json.dumps({"data": "r" * 5000})
        events = [
            StreamEvent(event_type="tool_start", data={
                "tool_call_id": "tc1", "tool_name": "document_search",
                "arguments": {"query": "alpha"},
            }, sequence=1, run_id=run_id),
            StreamEvent(event_type="tool_end", data={
                "tool_call_id": "tc1", "tool_name": "document_search",
                "result": big_result,
            }, sequence=2, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "Done"}, sequence=3, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=4, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=100)

        log = _get_log(request)
        parsed = json.loads(log.raw_output)
        self.assertEqual(len(parsed["tool_calls"]), 1)
        entry = parsed["tool_calls"][0]
        self.assertEqual(entry["tool_name"], "document_search")
        self.assertEqual(entry["result_size"], len(big_result))
        self.assertLess(len(entry["result_preview"]), 300)
        self.assertIn("chars total", entry["result_preview"])
        # The full result must NOT be stored anywhere in the row
        self.assertLess(len(log.raw_output), 2000)

    def test_stream_raw_output_caps_final_text(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="token", data={"text": "t" * 4000}, sequence=1, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id=run_id),
        ]
        log_stream(request, _acc(events), duration_ms=100)

        log = _get_log(request)
        parsed = json.loads(log.raw_output)
        self.assertLess(len(parsed["final_text_preview"]), 600)
        self.assertEqual(parsed["counts"]["text_chars"], 4000)

    def test_call_raw_output_is_slim_summary(self):
        request = _make_request()
        response = ChatResponse(
            message=Message(role="assistant", content="w" * 4000),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=50)

        log = _get_log(request)
        parsed = json.loads(log.raw_output)
        self.assertLess(len(parsed["final_text_preview"]), 600)
        self.assertEqual(parsed["counts"]["text_chars"], 4000)


class LogCallIntegrationTests(TestCase):
    """Verify that LLMService.run/stream actually create log entries."""

    def _make_service_with_fake_pipeline(self, pipeline_response=None, stream_events=None):
        from unittest.mock import MagicMock
        from llm.pipelines.registry import PipelineRegistry
        from llm.service.llm_service import LLMService

        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}

        if pipeline_response:
            fake_pipeline.run.return_value = pipeline_response
        if stream_events is not None:
            fake_pipeline.stream.side_effect = lambda req: iter(stream_events)

        registry = PipelineRegistry()
        fake_pipeline.id = "simple_chat"
        registry.register_pipeline(fake_pipeline)

        service = LLMService(
            pipeline_registry=registry,
            resolve_model_fn=lambda m: m or "gpt-4o-mini",
        )
        return service, fake_pipeline

    def test_run_success_creates_log_entry(self):
        response = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            metadata={},
        )
        service, _ = self._make_service_with_fake_pipeline(pipeline_response=response)
        request = _make_request()
        service.run("simple_chat", request)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertEqual(log.model, "gpt-4o-mini")

    def test_run_error_creates_error_log_entry(self):
        from unittest.mock import MagicMock
        from llm.pipelines.registry import PipelineRegistry
        from llm.service.llm_service import LLMService
        from llm.service.errors import LLMProviderError

        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.run.side_effect = RuntimeError("API down")
        fake_pipeline.id = "simple_chat"

        registry = PipelineRegistry()
        registry.register_pipeline(fake_pipeline)
        service = LLMService(
            pipeline_registry=registry,
            resolve_model_fn=lambda m: m or "gpt-4o-mini",
        )

        request = _make_request()
        with self.assertRaises(LLMProviderError):
            service.run("simple_chat", request)

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertEqual(log.error_type, "RuntimeError")

    def test_stream_success_creates_log_entry(self):
        request = _make_request(stream=True)
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=1, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id=run_id),
        ]
        service, _ = self._make_service_with_fake_pipeline(stream_events=events)
        list(service.stream("simple_chat", request))

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertTrue(log.is_stream)
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["final_text_preview"], "Hi")

    def test_stream_error_creates_error_log_entry(self):
        from unittest.mock import MagicMock
        from llm.pipelines.registry import PipelineRegistry
        from llm.service.llm_service import LLMService
        from llm.service.errors import LLMProviderError

        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.stream.side_effect = ConnectionError("Lost connection")
        fake_pipeline.id = "simple_chat"

        registry = PipelineRegistry()
        registry.register_pipeline(fake_pipeline)
        service = LLMService(
            pipeline_registry=registry,
            resolve_model_fn=lambda m: m or "gpt-4o-mini",
        )

        request = _make_request(stream=True)
        with self.assertRaises(LLMProviderError):
            list(service.stream("simple_chat", request))

        log = _get_log(request)
        self.assertEqual(log.status, "error")
        self.assertTrue(log.is_stream)
