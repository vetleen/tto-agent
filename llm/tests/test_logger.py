"""Tests for llm.service.logger — log_call, log_stream, log_error."""

import json
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm.models import LLMCallLog
from llm.service.logger import log_call, log_error, log_stream
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

User = get_user_model()


def _make_request(model="gpt-4o-mini", user_id=None, stream=False):
    return ChatRequest(
        messages=[Message(role="user", content="Hello")],
        stream=stream,
        model=model,
        context=RunContext.create(user_id=user_id),
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
        # raw_output = full response JSON
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["message"]["content"], "Hi!")
        self.assertEqual(parsed["model"], "gpt-4o-mini")
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
        log_stream(request, events, duration_ms=300)

        log = _get_log(request)
        self.assertEqual(log.status, "success")
        self.assertTrue(log.is_stream)
        # raw_output = assembled response JSON (single coherent blob)
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["message"]["role"], "assistant")
        self.assertEqual(parsed["message"]["content"], "Hello!")
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(log.duration_ms, 300)
        # Streaming doesn't log token counts
        self.assertIsNone(log.input_tokens)
        self.assertIsNone(log.cost_usd)

    def test_empty_events_produces_empty_output(self):
        request = _make_request(stream=True)
        log_stream(request, [], duration_ms=10)

        log = _get_log(request)
        parsed = json.loads(log.raw_output)
        self.assertEqual(parsed["message"]["content"], "")
        self.assertEqual(parsed["tool_calls"], [])

    def test_never_raises_on_db_error(self):
        request = _make_request(stream=True)
        with patch.object(LLMCallLog.objects, "create", side_effect=RuntimeError("DB is down")):
            log_stream(request, [], duration_ms=10)


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
        self.assertEqual(parsed["message"]["content"], "Hi")

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
