"""Tests for LLMService (routing, context injection, model resolution, error wrapping)."""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from llm.service.llm_service import LLMService, get_llm_service
from llm.service.errors import LLMConfigurationError, LLMPolicyDenied, LLMProviderError
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class LLMServiceTests(TestCase):
    """Test LLMService.run and .stream behavior."""

    def setUp(self):
        super().setUp()
        self.service = LLMService()
        # Ensure policies allow a model so pipeline lookup and run can proceed
        self.env = {
            "LLM_ALLOWED_MODELS": "gpt-4o-mini,claude-3-5-sonnet",
            "DEFAULT_LLM_MODEL": "gpt-4o-mini",
        }

    def test_run_fills_context_when_missing(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=None,
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": True, "tools": True}
                fake_pipeline.run.return_value = ChatResponse(
                    message=Message(role="assistant", content="Hi"),
                    model="gpt-4o-mini",
                    usage=None,
                    metadata={},
                )
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                self.service.run("simple_chat", request)
        self.assertIsNotNone(request.context)
        self.assertIsInstance(request.context, RunContext)

    def test_run_resolves_model_and_calls_pipeline(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model=None,
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": True, "tools": True}
                fake_pipeline.run.return_value = ChatResponse(
                    message=Message(role="assistant", content="Hi"),
                    model="gpt-4o-mini",
                    usage=None,
                    metadata={},
                )
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                response = self.service.run("simple_chat", request)
                self.assertEqual(request.model, "gpt-4o-mini")
                fake_pipeline.run.assert_called_once_with(request)
                self.assertEqual(response.message.content, "Hi")

    def test_stream_yields_events_with_run_id_and_monotonic_sequence(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        run_id = request.context.run_id

        def fake_stream(req):
            yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id)
            yield StreamEvent(event_type="token", data={"text": "Hello"}, sequence=2, run_id=run_id)
            yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id)

        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": True, "tools": True}
                fake_pipeline.stream.side_effect = fake_stream
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                events = list(self.service.stream("simple_chat", request))
        self.assertEqual(len(events), 3)
        for i, ev in enumerate(events):
            self.assertEqual(ev.run_id, run_id)
            self.assertEqual(ev.sequence, i + 1)

    def test_run_unknown_pipeline_raises(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                mock_reg.return_value.get_pipeline.side_effect = LLMConfigurationError("Unknown pipeline")
                with self.assertRaises(LLMConfigurationError):
                    self.service.run("unknown_pipeline", request)

    def test_get_llm_service_returns_singleton(self):
        a = get_llm_service()
        b = get_llm_service()
        self.assertIs(a, b)

    def test_run_when_pipeline_raises_generic_exception_wraps_as_llm_provider_error(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": True, "tools": True}
                fake_pipeline.run.side_effect = RuntimeError("Provider API down")
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                with self.assertRaises(LLMProviderError) as ctx:
                    self.service.run("simple_chat", request)
                self.assertIn("run failed", str(ctx.exception))
                self.assertIs(ctx.exception.__cause__, fake_pipeline.run.side_effect)

    def test_stream_when_pipeline_raises_generic_exception_wraps_as_llm_provider_error(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": True, "tools": True}
                fake_pipeline.stream.side_effect = ConnectionError("Network error")
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                with self.assertRaises(LLMProviderError) as ctx:
                    list(self.service.stream("simple_chat", request))
                self.assertIn("stream failed", str(ctx.exception))
                self.assertIs(ctx.exception.__cause__, fake_pipeline.stream.side_effect)

    def test_stream_when_pipeline_does_not_support_streaming_raises_policy_denied(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": False, "tools": True}
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                with self.assertRaises(LLMPolicyDenied) as ctx:
                    list(self.service.stream("simple_chat", request))
                self.assertIn("does not support streaming", str(ctx.exception))
                fake_pipeline.stream.assert_not_called()

    def test_run_when_request_stream_true_but_pipeline_no_streaming_raises_policy_denied(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        with patch.dict("os.environ", self.env, clear=False):
            with patch("llm.service.llm_service.get_pipeline_registry") as mock_reg:
                fake_pipeline = MagicMock()
                fake_pipeline.capabilities = {"streaming": False, "tools": True}
                mock_reg.return_value.get_pipeline.return_value = fake_pipeline
                with self.assertRaises(LLMPolicyDenied) as ctx:
                    self.service.run("simple_chat", request)
                self.assertIn("does not support streaming", str(ctx.exception))
                fake_pipeline.run.assert_not_called()
