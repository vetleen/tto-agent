"""Tests for raw_prompt capture — verifying that tool schemas appear in LLMCallLog.raw_prompt.

All providers inherit BaseLangChainChatModel, so one test suite covers the
tool-attachment logic for all of them.
"""

import uuid
from unittest.mock import MagicMock

from django.test import TestCase

from llm.core.providers.base import BaseLangChainChatModel
from llm.models import LLMCallLog
from llm.service.logger import log_call, log_stream
from llm.tools.schema import tools_to_langchain_schemas
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_schema(name, description="A tool"):
    """Build an OpenAI-format tool schema dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }


def _fake_tool(name="search_documents", description="Search docs"):
    """Return a mock Tool with .name, .description, .parameters."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    return tool


def _make_lc_ai_message(content="Hello"):
    """Create a mock that behaves like a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    msg.usage_metadata = None
    return msg


def _fire_callback(messages, config):
    """Simulate what a real LangChain client does: fire on_chat_model_start."""
    if config and "callbacks" in config:
        for cb in config["callbacks"]:
            if hasattr(cb, "on_chat_model_start"):
                cb.on_chat_model_start({}, [messages], run_id=uuid.uuid4())


def _invoke_with_callback(return_value):
    """Side-effect for mock invoke: fires callback then returns result."""
    def side_effect(messages, config=None, **kwargs):
        _fire_callback(messages, config)
        return return_value
    return side_effect


def _stream_with_callback(chunks):
    """Side-effect for mock stream: fires callback then yields chunks."""
    def side_effect(messages, config=None, **kwargs):
        _fire_callback(messages, config)
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
# BaseLangChainChatModel.generate — raw_prompt in response metadata
# ---------------------------------------------------------------------------

class GenerateRawPromptTests(TestCase):
    """Verify generate() attaches tool schemas to raw_prompt in metadata."""

    def test_raw_prompt_includes_tools_when_tool_schemas_present(self):
        model, client, bound_client = _make_model_with_mock_client()
        bound_client.invoke.side_effect = _invoke_with_callback(_make_lc_ai_message("OK"))

        schemas = [_make_tool_schema("search_documents"), _make_tool_schema("add_number")]
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        response = model.generate(request)

        raw_prompt = response.metadata["raw_prompt"]
        self.assertIsNotNone(raw_prompt)
        self.assertEqual(len(raw_prompt["tools"]), 2)
        tool_names = [t["function"]["name"] for t in raw_prompt["tools"]]
        self.assertIn("search_documents", tool_names)
        self.assertIn("add_number", tool_names)

    def test_raw_prompt_tools_empty_when_no_tool_schemas(self):
        model, client, bound_client = _make_model_with_mock_client()
        client.invoke.side_effect = _invoke_with_callback(_make_lc_ai_message("Hi"))

        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        response = model.generate(request)

        raw_prompt = response.metadata["raw_prompt"]
        self.assertIsNotNone(raw_prompt)
        self.assertEqual(raw_prompt["tools"], [])

    def test_raw_prompt_messages_populated(self):
        model, client, bound_client = _make_model_with_mock_client()
        bound_client.invoke.side_effect = _invoke_with_callback(_make_lc_ai_message("Done"))

        schemas = [_make_tool_schema("search_documents")]
        request = ChatRequest(
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Search for cats"),
            ],
            stream=False,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        response = model.generate(request)

        raw_prompt = response.metadata["raw_prompt"]
        messages = raw_prompt["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Search for cats")

    def test_tools_to_langchain_schemas_integration(self):
        """Verify tools_to_langchain_schemas output flows through to raw_prompt."""
        model, client, bound_client = _make_model_with_mock_client()
        bound_client.invoke.side_effect = _invoke_with_callback(_make_lc_ai_message("5"))

        tools = [_fake_tool("add_number", "Add two numbers")]
        schemas = tools_to_langchain_schemas(tools)

        request = ChatRequest(
            messages=[Message(role="user", content="2+3?")],
            stream=False,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        response = model.generate(request)

        raw_prompt = response.metadata["raw_prompt"]
        self.assertEqual(len(raw_prompt["tools"]), 1)
        fn = raw_prompt["tools"][0]["function"]
        self.assertEqual(fn["name"], "add_number")
        self.assertEqual(fn["description"], "Add two numbers")
        self.assertIn("query", fn["parameters"]["properties"])


# ---------------------------------------------------------------------------
# BaseLangChainChatModel.stream — raw_prompt event
# ---------------------------------------------------------------------------

class StreamRawPromptTests(TestCase):
    """Verify stream() emits a raw_prompt event containing tool schemas."""

    def test_raw_prompt_event_includes_tools(self):
        model, client, bound_client = _make_model_with_mock_client()

        chunk = MagicMock()
        chunk.content = "Hello"
        bound_client.stream.side_effect = _stream_with_callback([chunk])

        schemas = [_make_tool_schema("search_documents")]
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            tool_schemas=schemas,
            context=RunContext.create(),
        )
        events = list(model.stream(request))

        raw_prompt_events = [e for e in events if e.event_type == "raw_prompt"]
        self.assertEqual(len(raw_prompt_events), 1)

        raw_prompt = raw_prompt_events[0].data["raw_prompt"]
        self.assertEqual(len(raw_prompt["tools"]), 1)
        self.assertEqual(raw_prompt["tools"][0]["function"]["name"], "search_documents")

    def test_raw_prompt_event_tools_empty_without_schemas(self):
        model, client, bound_client = _make_model_with_mock_client()

        chunk = MagicMock()
        chunk.content = "Hi"
        client.stream.side_effect = _stream_with_callback([chunk])

        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            context=RunContext.create(),
        )
        events = list(model.stream(request))

        raw_prompt_events = [e for e in events if e.event_type == "raw_prompt"]
        self.assertEqual(len(raw_prompt_events), 1)

        raw_prompt = raw_prompt_events[0].data["raw_prompt"]
        self.assertEqual(raw_prompt["tools"], [])


# ---------------------------------------------------------------------------
# Logger integration — LLMCallLog.raw_prompt stores tools
# ---------------------------------------------------------------------------

class LogCallRawPromptToolsTests(TestCase):
    """Verify log_call stores raw_prompt with tools into the database."""

    def test_log_call_stores_tools_in_raw_prompt(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        raw = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                _make_tool_schema("search_documents"),
                _make_tool_schema("add_number"),
            ],
        }
        response = ChatResponse(
            message=Message(role="assistant", content="OK"),
            model="test-model",
            usage=None,
            metadata={"raw_prompt": raw},
        )
        log_call(request, response, duration_ms=100)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertIsNotNone(log.raw_prompt)
        self.assertEqual(len(log.raw_prompt["tools"]), 2)
        tool_names = [t["function"]["name"] for t in log.raw_prompt["tools"]]
        self.assertIn("search_documents", tool_names)
        self.assertIn("add_number", tool_names)

    def test_log_call_stores_empty_tools_when_no_tools(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="test-model",
            context=RunContext.create(),
        )
        raw = {"messages": [{"role": "user", "content": "Hi"}], "tools": []}
        response = ChatResponse(
            message=Message(role="assistant", content="OK"),
            model="test-model",
            usage=None,
            metadata={"raw_prompt": raw},
        )
        log_call(request, response, duration_ms=100)

        log = LLMCallLog.objects.get(run_id=request.context.run_id)
        self.assertEqual(log.raw_prompt["tools"], [])


class LogStreamRawPromptToolsTests(TestCase):
    """Verify log_stream stores raw_prompt with tools from the stream events."""

    def test_log_stream_stores_tools_in_raw_prompt(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="test-model",
            context=RunContext.create(),
        )
        run_id = request.context.run_id
        raw = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [_make_tool_schema("search_documents")],
        }
        events = [
            StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id),
            StreamEvent(event_type="raw_prompt", data={"raw_prompt": raw}, sequence=2, run_id=run_id),
            StreamEvent(event_type="token", data={"text": "OK"}, sequence=3, run_id=run_id),
            StreamEvent(event_type="message_end", data={}, sequence=4, run_id=run_id),
        ]
        log_stream(request, events, duration_ms=200)

        log = LLMCallLog.objects.get(run_id=run_id)
        self.assertIsNotNone(log.raw_prompt)
        self.assertEqual(len(log.raw_prompt["tools"]), 1)
        self.assertEqual(log.raw_prompt["tools"][0]["function"]["name"], "search_documents")
