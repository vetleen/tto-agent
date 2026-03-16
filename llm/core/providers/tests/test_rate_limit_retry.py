"""Tests for error handling in BaseLangChainChatModel.

Retry logic is now handled by LangChain's built-in max_retries (set in
model_factory). These tests verify error wrapping: rate limit errors (429)
are raised as LLMRateLimitError, other provider errors as LLMProviderError,
and stream errors produce error events.
"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock

from llm.core.providers.base import BaseLangChainChatModel
from llm.service.errors import LLMProviderError, LLMRateLimitError
from llm.types.messages import Message
from llm.types.requests import ChatRequest


def _make_model():
    """Create a minimal BaseLangChainChatModel instance for testing."""
    client = MagicMock()
    model = BaseLangChainChatModel(model_name="test-model", client=client)
    model._provider_label = "Test"
    return model


def _make_request():
    return ChatRequest(
        messages=[Message(role="user", content="hi")],
        model="test-model",
    )


def _rate_limit_error():
    exc = Exception("rate limited")
    exc.status_code = 429
    return exc


def _auth_error():
    exc = Exception("unauthorized")
    exc.status_code = 401
    return exc


def _ai_message(text="hello"):
    msg = MagicMock()
    msg.content = text
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    msg.response_metadata = {}
    msg.tool_calls = []
    msg.additional_kwargs = {}
    return msg


class TestGenerateErrorHandling(TestCase):

    def test_generate_rate_limit_raises_llm_rate_limit_error(self):
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_rate_limit_error())
        with self.assertRaises(LLMRateLimitError):
            model.generate(_make_request())
        self.assertEqual(model._client.invoke.call_count, 1)

    def test_generate_non_rate_limit_error_raises_provider_error(self):
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_auth_error())
        with self.assertRaises(LLMProviderError) as ctx:
            model.generate(_make_request())
        self.assertNotIsInstance(ctx.exception, LLMRateLimitError)
        self.assertEqual(model._client.invoke.call_count, 1)

    def test_generate_success(self):
        model = _make_model()
        model._client.invoke = MagicMock(return_value=_ai_message())
        result = model.generate(_make_request())
        self.assertEqual(result.message.content, "hello")
        self.assertEqual(model._client.invoke.call_count, 1)


class TestStreamErrorHandling(TestCase):

    def test_stream_error_produces_error_event(self):
        model = _make_model()

        def stream_fail(*args, **kwargs):
            raise _rate_limit_error()

        model._client.stream = MagicMock(side_effect=stream_fail)
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("error", event_types)
        self.assertIn("message_start", event_types)
        # Should not have message_end (error terminates early)
        self.assertNotIn("message_end", event_types)

    def test_stream_error_after_content_produces_error_event(self):
        """Error after partial content is yielded still produces an error event."""
        model = _make_model()

        chunk = MagicMock()
        chunk.content = "partial"
        chunk.usage_metadata = {}
        chunk.response_metadata = {}

        def stream_then_fail(*args, **kwargs):
            yield chunk
            raise _rate_limit_error()

        model._client.stream = MagicMock(side_effect=stream_then_fail)
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("error", event_types)

    def test_stream_success(self):
        model = _make_model()

        chunk = MagicMock()
        chunk.content = "hello"
        chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        chunk.response_metadata = {}

        model._client.stream = MagicMock(return_value=[chunk])
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("message_start", event_types)
        self.assertIn("token", event_types)
        self.assertIn("message_end", event_types)
        self.assertNotIn("error", event_types)
