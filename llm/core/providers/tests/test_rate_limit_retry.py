"""Tests for rate-limit retry/backoff in BaseLangChainChatModel."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch, PropertyMock

from llm.core.providers.base import BaseLangChainChatModel
from llm.service.errors import LLMProviderError, LLMRateLimitError
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.streaming import StreamEvent


def _make_model():
    """Create a minimal BaseLangChainChatModel instance for testing."""
    model = BaseLangChainChatModel.__new__(BaseLangChainChatModel)
    model.name = "test-model"
    model._provider_label = "Test"
    model._client = MagicMock()
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


class TestGenerateRetry(TestCase):

    @patch("llm.core.providers.base.time.sleep")
    def test_generate_retries_on_rate_limit(self, mock_sleep):
        model = _make_model()
        model._client.invoke = MagicMock(
            side_effect=[_rate_limit_error(), _rate_limit_error(), _ai_message()]
        )
        result = model.generate(_make_request())
        self.assertEqual(result.message.content, "hello")
        self.assertEqual(model._client.invoke.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("llm.core.providers.base.time.sleep")
    def test_generate_no_retry_on_auth_error(self, mock_sleep):
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_auth_error())
        with self.assertRaises(LLMProviderError) as ctx:
            model.generate(_make_request())
        self.assertNotIsInstance(ctx.exception, LLMRateLimitError)
        self.assertEqual(model._client.invoke.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("llm.core.providers.base.time.sleep")
    def test_generate_retries_exhausted(self, mock_sleep):
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_rate_limit_error())
        with self.assertRaises(LLMRateLimitError):
            model.generate(_make_request())
        self.assertEqual(model._client.invoke.call_count, 3)


class TestStreamRetry(TestCase):

    @patch("llm.core.providers.base.time.sleep")
    def test_stream_retries_before_content(self, mock_sleep):
        model = _make_model()

        chunk = MagicMock()
        chunk.content = "hi"
        chunk.usage_metadata = {}
        chunk.response_metadata = {}

        # First call raises 429, second succeeds
        model._client.stream = MagicMock(
            side_effect=[_rate_limit_error(), [chunk]]
        )
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("token", event_types)
        self.assertNotIn("error", event_types)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("llm.core.providers.base.time.sleep")
    def test_stream_no_retry_after_content(self, mock_sleep):
        """Once content has been yielded (raw_prompt_yielded=True), don't retry."""
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
        mock_sleep.assert_not_called()
