"""Tests for LLMService (routing, context injection, model resolution, error wrapping)."""

from unittest.mock import MagicMock

from django.test import TestCase

from llm.pipelines.registry import PipelineRegistry
from llm.service.llm_service import LLMService, get_llm_service
from llm.service.errors import LLMConfigurationError, LLMPolicyDenied, LLMProviderError
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


def _stub_resolve(model=None):
    """Resolve model stub: returns the requested model or a default."""
    return model or "gpt-4o-mini"


def _make_service(pipeline, pipeline_id="simple_chat"):
    """Create an LLMService with a single pipeline injected."""
    registry = PipelineRegistry()
    pipeline.id = pipeline_id
    registry.register_pipeline(pipeline)
    return LLMService(pipeline_registry=registry, resolve_model_fn=_stub_resolve)


class LLMServiceTests(TestCase):
    """Test LLMService.run and .stream behavior."""

    def test_run_fills_context_when_missing(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=None,
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.run.return_value = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        service = _make_service(fake_pipeline)
        service.run("simple_chat", request)
        self.assertIsNotNone(request.context)
        self.assertIsInstance(request.context, RunContext)

    def test_run_resolves_model_and_calls_pipeline(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model=None,
            context=RunContext.create(),
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.run.return_value = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        service = _make_service(fake_pipeline)
        response = service.run("simple_chat", request)
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

        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.stream.side_effect = fake_stream
        service = _make_service(fake_pipeline)
        events = list(service.stream("simple_chat", request))
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
        service = LLMService(
            pipeline_registry=PipelineRegistry(),
            resolve_model_fn=_stub_resolve,
        )
        with self.assertRaises(LLMConfigurationError):
            service.run("unknown_pipeline", request)

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
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.run.side_effect = RuntimeError("Provider API down")
        service = _make_service(fake_pipeline)
        with self.assertRaises(LLMProviderError) as ctx:
            service.run("simple_chat", request)
        self.assertIn("run failed", str(ctx.exception))
        self.assertIs(ctx.exception.__cause__, fake_pipeline.run.side_effect)

    def test_stream_when_pipeline_raises_generic_exception_wraps_as_llm_provider_error(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.stream.side_effect = ConnectionError("Network error")
        service = _make_service(fake_pipeline)
        with self.assertRaises(LLMProviderError) as ctx:
            list(service.stream("simple_chat", request))
        self.assertIn("stream failed", str(ctx.exception))
        self.assertIs(ctx.exception.__cause__, fake_pipeline.stream.side_effect)

    def test_stream_when_pipeline_does_not_support_streaming_raises_policy_denied(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": False, "tools": True}
        service = _make_service(fake_pipeline)
        with self.assertRaises(LLMPolicyDenied) as ctx:
            list(service.stream("simple_chat", request))
        self.assertIn("does not support streaming", str(ctx.exception))
        fake_pipeline.stream.assert_not_called()

    def test_run_when_request_stream_true_but_pipeline_no_streaming_raises_policy_denied(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": False, "tools": True}
        service = _make_service(fake_pipeline)
        with self.assertRaises(LLMPolicyDenied) as ctx:
            service.run("simple_chat", request)
        self.assertIn("does not support streaming", str(ctx.exception))
        fake_pipeline.run.assert_not_called()

    async def test_arun_delegates_to_run(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        expected = ChatResponse(
            message=Message(role="assistant", content="Hi"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.run.return_value = expected
        service = _make_service(fake_pipeline)
        result = await service.arun("simple_chat", request)
        self.assertEqual(result.message.content, "Hi")
        fake_pipeline.run.assert_called_once_with(request)

    async def test_astream_yields_events(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        run_id = request.context.run_id

        def fake_stream(req):
            yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id)
            yield StreamEvent(event_type="token", data={"text": "Hi"}, sequence=2, run_id=run_id)
            yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id)

        fake_pipeline = MagicMock()
        fake_pipeline.capabilities = {"streaming": True, "tools": True}
        fake_pipeline.stream.side_effect = fake_stream
        service = _make_service(fake_pipeline)
        events = []
        async for event in service.astream("simple_chat", request):
            events.append(event)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].data, {"text": "Hi"})
        self.assertEqual(events[2].event_type, "message_end")
