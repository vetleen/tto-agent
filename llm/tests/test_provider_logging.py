"""Tests for Python logging around provider API calls (base.py generate/stream)."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from llm.core.providers.base import BaseLangChainChatModel
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest


def _make_request(model="test-model", messages=None):
    return ChatRequest(
        messages=messages or [Message(role="user", content="Hello")],
        stream=False,
        model=model,
        context=RunContext.create(),
    )


class _FakeAIMessage:
    """Minimal stand-in for a LangChain AIMessage."""

    def __init__(self, content="Hi", usage_metadata=None, response_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        self.response_metadata = response_metadata or {}
        self.tool_calls = []
        self.additional_kwargs = {}


class _FakeChunk:
    """Minimal stand-in for a LangChain AIMessageChunk."""

    def __init__(self, content="Hi", usage_metadata=None, response_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}
        self.additional_kwargs = {}


class _TestModel(BaseLangChainChatModel):
    """Concrete subclass for testing."""

    def __init__(self):
        self.name = "test-model"
        self._provider_label = "TestProvider"
        self._client = MagicMock()


class GenerateLoggingTests(SimpleTestCase):
    """Verify INFO/ERROR logs emitted by generate()."""

    def test_generate_logs_start_and_complete_on_success(self):
        model = _TestModel()
        model._client.invoke.return_value = _FakeAIMessage()
        request = _make_request()

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            model.generate(request)

        log_output = "\n".join(cm.output)
        self.assertIn("LLM generate start", log_output)
        self.assertIn("model=test-model", log_output)
        self.assertIn("provider=TestProvider", log_output)
        self.assertIn("messages=1", log_output)
        self.assertIn("LLM generate complete", log_output)
        self.assertIn("input_tokens=10", log_output)
        self.assertIn("output_tokens=5", log_output)

    def test_generate_does_not_log_message_content(self):
        model = _TestModel()
        model._client.invoke.return_value = _FakeAIMessage(content="secret response")
        request = _make_request(messages=[Message(role="user", content="secret prompt")])

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            model.generate(request)

        log_output = "\n".join(cm.output)
        self.assertNotIn("secret prompt", log_output)
        self.assertNotIn("secret response", log_output)

    def test_generate_logs_error_on_provider_failure(self):
        model = _TestModel()
        model._client.invoke.side_effect = ValueError("API error")
        request = _make_request()

        from llm.service.errors import LLMProviderError

        with self.assertLogs("llm.core.providers.base", level="ERROR") as cm:
            with self.assertRaises(LLMProviderError):
                model.generate(request)

        log_output = "\n".join(cm.output)
        self.assertIn("LLM generate failed", log_output)
        self.assertIn("model=test-model", log_output)

    @patch("llm.core.providers.base.time.sleep", return_value=None)
    def test_generate_logs_rate_limit_exhaustion(self, mock_sleep):
        model = _TestModel()
        exc = Exception("rate limited")
        exc.status_code = 429
        model._client.invoke.side_effect = exc
        request = _make_request()

        from llm.service.errors import LLMRateLimitError

        with self.assertLogs("llm.core.providers.base", level="WARNING") as cm:
            with self.assertRaises(LLMRateLimitError):
                model.generate(request)

        log_output = "\n".join(cm.output)
        self.assertIn("rate limit", log_output.lower())

    def test_generate_includes_run_id_in_logs(self):
        model = _TestModel()
        model._client.invoke.return_value = _FakeAIMessage()
        request = _make_request()
        run_id = request.context.run_id

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            model.generate(request)

        log_output = "\n".join(cm.output)
        self.assertIn(run_id, log_output)


class StreamLoggingTests(SimpleTestCase):
    """Verify INFO/ERROR logs emitted by stream()."""

    def test_stream_logs_start_and_complete_on_success(self):
        model = _TestModel()
        chunk = _FakeChunk(
            content="Hi",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        model._client.stream.return_value = iter([chunk])
        request = _make_request()

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            list(model.stream(request))

        log_output = "\n".join(cm.output)
        self.assertIn("LLM stream start", log_output)
        self.assertIn("model=test-model", log_output)
        self.assertIn("provider=TestProvider", log_output)
        self.assertIn("LLM stream complete", log_output)

    def test_stream_does_not_log_message_content(self):
        model = _TestModel()
        chunk = _FakeChunk(content="secret streamed text")
        model._client.stream.return_value = iter([chunk])
        request = _make_request(messages=[Message(role="user", content="secret input")])

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            list(model.stream(request))

        log_output = "\n".join(cm.output)
        self.assertNotIn("secret input", log_output)
        self.assertNotIn("secret streamed text", log_output)

    def test_stream_logs_error_on_failure(self):
        model = _TestModel()
        model._client.stream.side_effect = ConnectionError("Network error")
        request = _make_request()

        with self.assertLogs("llm.core.providers.base", level="ERROR") as cm:
            events = list(model.stream(request))

        log_output = "\n".join(cm.output)
        self.assertIn("LLM stream error", log_output)
        self.assertIn("model=test-model", log_output)
        # Should also yield an error event
        error_events = [e for e in events if e.event_type == "error"]
        self.assertEqual(len(error_events), 1)

    def test_stream_includes_run_id_in_logs(self):
        model = _TestModel()
        chunk = _FakeChunk(content="Hi")
        model._client.stream.return_value = iter([chunk])
        request = _make_request()
        run_id = request.context.run_id

        with self.assertLogs("llm.core.providers.base", level="INFO") as cm:
            list(model.stream(request))

        log_output = "\n".join(cm.output)
        self.assertIn(run_id, log_output)
