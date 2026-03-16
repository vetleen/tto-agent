"""Tests for error handling and rate-limit retry in BaseLangChainChatModel.

Rate limit errors (429) trigger exponential backoff retries (up to 3).
Other provider errors raise LLMProviderError immediately.
Stream errors produce error events.
"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

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


@patch("llm.core.providers.base.time.sleep", return_value=None)
class TestGenerateErrorHandling(TestCase):

    def test_generate_rate_limit_retries_then_raises(self, mock_sleep):
        """429 errors retry with backoff, then raise LLMRateLimitError."""
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_rate_limit_error())
        with self.assertRaises(LLMRateLimitError):
            model.generate(_make_request())
        # 1 initial + 3 retries = 4 total calls
        self.assertEqual(model._client.invoke.call_count, 4)
        # 3 sleeps with exponential backoff: 30, 60, 120
        self.assertEqual(mock_sleep.call_count, 3)
        self.assertEqual(mock_sleep.call_args_list[0].args[0], 30)
        self.assertEqual(mock_sleep.call_args_list[1].args[0], 60)
        self.assertEqual(mock_sleep.call_args_list[2].args[0], 120)

    def test_generate_rate_limit_succeeds_on_retry(self, mock_sleep):
        """429 on first attempt, success on second."""
        model = _make_model()
        model._client.invoke = MagicMock(
            side_effect=[_rate_limit_error(), _ai_message("ok")]
        )
        result = model.generate(_make_request())
        self.assertEqual(result.message.content, "ok")
        self.assertEqual(model._client.invoke.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_generate_non_rate_limit_error_raises_provider_error(self, mock_sleep):
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_auth_error())
        with self.assertRaises(LLMProviderError) as ctx:
            model.generate(_make_request())
        self.assertNotIsInstance(ctx.exception, LLMRateLimitError)
        # No retries for non-429 errors
        self.assertEqual(model._client.invoke.call_count, 1)
        mock_sleep.assert_not_called()

    def test_generate_success(self, mock_sleep):
        model = _make_model()
        model._client.invoke = MagicMock(return_value=_ai_message())
        result = model.generate(_make_request())
        self.assertEqual(result.message.content, "hello")
        self.assertEqual(model._client.invoke.call_count, 1)
        mock_sleep.assert_not_called()


@patch("llm.core.providers.base.time.sleep", return_value=None)
class TestStreamErrorHandling(TestCase):

    def test_stream_rate_limit_retries_then_errors(self, mock_sleep):
        """429 on stream retries with backoff, then yields error event."""
        model = _make_model()
        model._client.stream = MagicMock(side_effect=_rate_limit_error())
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("error", event_types)
        self.assertIn("message_start", event_types)
        self.assertNotIn("message_end", event_types)
        # 1 initial + 3 retries = 4 total calls
        self.assertEqual(model._client.stream.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    def test_stream_rate_limit_succeeds_on_retry(self, mock_sleep):
        """429 on first stream attempt, success on second."""
        model = _make_model()

        chunk = MagicMock()
        chunk.content = "hello"
        chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        chunk.response_metadata = {}

        model._client.stream = MagicMock(
            side_effect=[_rate_limit_error(), [chunk]]
        )
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("token", event_types)
        self.assertIn("message_end", event_types)
        self.assertNotIn("error", event_types)
        self.assertEqual(model._client.stream.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    def test_stream_mid_stream_rate_limit_no_retry(self, mock_sleep):
        """Rate limit after tokens have been yielded should NOT retry."""
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
        # No retry because tokens were already streamed
        self.assertEqual(model._client.stream.call_count, 1)
        mock_sleep.assert_not_called()

    def test_stream_non_rate_limit_error_no_retry(self, mock_sleep):
        """Non-429 errors produce error event immediately, no retries."""
        model = _make_model()
        model._client.stream = MagicMock(side_effect=_auth_error())
        model._client.bind_tools = MagicMock(return_value=model._client)

        events = list(model.stream(_make_request()))
        event_types = [e.event_type for e in events]
        self.assertIn("error", event_types)
        self.assertEqual(model._client.stream.call_count, 1)
        mock_sleep.assert_not_called()

    def test_stream_success(self, mock_sleep):
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
        mock_sleep.assert_not_called()
