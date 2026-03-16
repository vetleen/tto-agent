"""Tests for the tools field — verifying that tool schemas are stored in LLMCallLog.tools.

Tests cover:
- generate() no longer emits raw_prompt in metadata
- stream() no longer emits raw_prompt events
- Logger stores tools from request.tool_schemas
- Error logging stores tools
"""

from unittest.mock import MagicMock

from django.test import TestCase

from llm.core.providers.base import BaseLangChainChatModel
from llm.models import LLMCallLog
from llm.service.logger import log_call, log_error, log_stream
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_tool(name="search_documents", description="Search docs"):
    """Return a ContextAwareTool instance for schema tests."""
    from pydantic import BaseModel, Field
    from llm.tools.interfaces import ContextAwareTool

    class _Input(BaseModel):
        query: str = Field(description="Search query")

    class _Tool(ContextAwareTool):
        name: str = "placeholder"
        description: str = "placeholder"
        args_schema: type[BaseModel] = _Input
        def _run(self, query: str) -> str:
            return "{}"

    return _Tool(name=name, description=description)


def _make_lc_ai_message(content="Hello"):
    """Create a mock that behaves like a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    msg.usage_metadata = None
    return msg


def _invoke_side_effect(return_value):
    """Side-effect for mock invoke."""
    def side_effect(messages, config=None, **kwargs):
        return return_value
    return side_effect


def _stream_side_effect(chunks):
    """Side-effect for mock stream."""
    def side_effect(messages, config=None, **kwargs):
        yield from chunks
    return side_effect


def _make_model_with_mock_client():
    """Create a BaseLangChainChatModel subclass with a mock LangChain client."""
    model = BaseLangChainChatModel.__new__(BaseLangChainChatModel)
    model.name = "test-model"
    model._provider_label = "Test"

    client = MagicMock()
    bound_client = MagicMock()
    client.bind_tools.return_value = bound_client

    model._client = client
    return model, client, bound_client


# ---------------------------------------------------------------------------
# generate() — no raw_prompt in metadata
# ---------------------------------------------------------------------------

class GenerateNoRawPromptTests(TestCase):
    """Verify generate() no longer includes raw_prompt in response metadata."""

    def test_metadata_has_no_raw_prompt_key(self):
        model, client, bound_client = _make_model_with_mock_client()
        bound_client.invoke.side_effect = _invoke_side_effect(_make_lc_ai_message("OK"))

        schemas = [_fake_tool("search_documents")]
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        response = model.generate(request)

        self.assertNotIn("raw_prompt", response.metadata)

    def test_metadata_has_no_raw_prompt_without_tools(self):
        model, client, bound_client = _make_model_with_mock_client()
        client.invoke.side_effect = _invoke_side_effect(_make_lc_ai_message("Hi"))

        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        response = model.generate(request)

        self.assertNotIn("raw_prompt", response.metadata)


# ---------------------------------------------------------------------------
# stream() — no raw_prompt events
# ---------------------------------------------------------------------------

class StreamNoRawPromptTests(TestCase):
    """Verify stream() no longer emits raw_prompt events."""

    def test_no_raw_prompt_event_with_tools(self):
        model, client, bound_client = _make_model_with_mock_client()

        chunk = MagicMock()
        chunk.content = "Hello"
        bound_client.stream.side_effect = _stream_side_effect([chunk])

        schemas = [_fake_tool("search_documents")]
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        events = list(model.stream(request))

        raw_prompt_events = [e for e in events if e.event_type == "raw_prompt"]
        self.assertEqual(len(raw_prompt_events), 0)

    def test_no_raw_prompt_event_without_tools(self):
        model, client, bound_client = _make_model_with_mock_client()

        chunk = MagicMock()
        chunk.content = "Hi"
        client.stream.side_effect = _stream_side_effect([chunk])

        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            context=RunContext.create(),
        )
        events = list(model.stream(request))

        raw_prompt_events = [e for e in events if e.event_type == "raw_prompt"]
        self.assertEqual(len(raw_prompt_events), 0)


# ---------------------------------------------------------------------------
# Logger — tools field populated from request.tool_schemas
# ---------------------------------------------------------------------------

class LogCallToolsTests(TestCase):
    """Verify log_call stores tools from request.tool_schemas."""

    def test_tools_populated_with_tool_schemas(self):
        tool1 = _fake_tool("search_documents", "Search docs")
        tool2 = _fake_tool("read_document", "Read a doc")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            tool_schemas=[tool1, tool2],
            context=RunContext.create(),
        )
        response = ChatResponse(
            message=Message(role="assistant", content="OK"),
            model="test-model",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=100)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertEqual(len(log.tools), 2)
        self.assertEqual(log.tools[0]["name"], "search_documents")
        self.assertEqual(log.tools[1]["name"], "read_document")

    def test_tools_none_without_tool_schemas(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        response = ChatResponse(
            message=Message(role="assistant", content="OK"),
            model="test-model",
            usage=None,
            metadata={},
        )
        log_call(request, response, duration_ms=100)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertIsNone(log.tools)


class LogStreamToolsTests(TestCase):
    """Verify log_stream stores tools from request.tool_schemas."""

    def test_tools_populated_in_stream_log(self):
        tool = _fake_tool("search_documents", "Search docs")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            tool_schemas=[tool],
            context=RunContext.create(),
        )
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "OK"}, sequence=2, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id),
        ]
        log_stream(request, events, duration_ms=200)

        log = LLMCallLog.objects.get(run_id=run_id)
        self.assertEqual(len(log.tools), 1)
        self.assertEqual(log.tools[0]["name"], "search_documents")

    def test_tools_none_in_stream_log_without_schemas(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            context=RunContext.create(),
        )
        run_id = request.context.run_id
        events = [
            StreamEvent(event_type="token", data={"text": "OK"}, sequence=1, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id=run_id),
        ]
        log_stream(request, events, duration_ms=100)

        log = LLMCallLog.objects.get(run_id=run_id)
        self.assertIsNone(log.tools)


class LogErrorToolsTests(TestCase):
    """Verify log_error stores tools from request.tool_schemas."""

    def test_tools_populated_in_error_log(self):
        tool = _fake_tool("search_documents", "Search docs")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            tool_schemas=[tool],
            context=RunContext.create(),
        )
        log_error(request, ValueError("Bad"), duration_ms=50)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertEqual(len(log.tools), 1)
        self.assertEqual(log.tools[0]["name"], "search_documents")

    def test_tools_none_in_error_log_without_schemas(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        log_error(request, ValueError("Bad"), duration_ms=50)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertIsNone(log.tools)
