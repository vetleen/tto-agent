"""Tests for the simple_chat pipeline (tool loop and model delegation)."""

from unittest.mock import MagicMock, patch

from django.test import TestCase
from pydantic import BaseModel, Field

from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.tools.interfaces import ContextAwareTool
from llm.types.context import RunContext
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class _MockInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class _MockToolImpl(ContextAwareTool):
    name: str = "search_documents"
    description: str = "A mock tool"
    args_schema: type[BaseModel] = _MockInput
    _return_value: str = '{"result": 5}'

    def _run(self, a: float, b: float) -> str:
        return self._return_value


class SimpleChatPipelineTests(TestCase):
    """Test simple_chat pipeline: tool loop and model delegation."""

    def test_run_no_tools_delegates_directly_to_model(self):
        """When request.tools is None/empty, pipeline delegates to model without tool loop."""
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_response = ChatResponse(
            message=Message(role="assistant", content="Hi there"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.return_value = fake_response

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create:
            mock_create.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_create.assert_called_once_with("gpt-4o-mini")
            fake_model.generate.assert_called_once()
            # Request should not have tool_schemas
            call_args = fake_model.generate.call_args[0][0]
            self.assertIsNone(call_args.tool_schemas)
        self.assertEqual(response.message.content, "Hi there")

    def _make_mock_tool(self, name="mock_tool"):
        """Create a mock ContextAwareTool for testing."""
        tool = _MockToolImpl(name=name)
        return tool

    def _patch_tool_registry(self, tool):
        """Return a patch context for get_tool_registry that returns the given tool."""
        mock_registry = MagicMock()
        mock_registry.return_value.get_tool.side_effect = lambda n: tool if n == tool.name else None
        return patch("llm.pipelines.simple_chat.get_tool_registry", mock_registry)

    def test_run_tool_loop_executes_tool_and_returns_final_response(self):
        """Model returns tool_calls on first call, then text on second; tool is executed."""
        mock_tool = self._make_mock_tool("search_documents")
        request = ChatRequest(
            messages=[Message(role="user", content="What is 2 + 3?")],
            stream=False,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="call_1", name="search_documents", arguments={"a": 2, "b": 3})
        response_with_tool = ChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[tool_call],
            ),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        final_response = ChatResponse(
            message=Message(role="assistant", content="The sum is 5."),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.side_effect = [response_with_tool, final_response]

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            response = SimpleChatPipeline().run(request)

        self.assertEqual(fake_model.generate.call_count, 2)
        self.assertEqual(response.message.content, "The sum is 5.")
        # First call had tool_schemas, second had messages with assistant + tool result
        first_call = fake_model.generate.call_args_list[0][0][0]
        self.assertIsNotNone(first_call.tool_schemas)
        self.assertEqual(len(first_call.messages), 1)
        second_call = fake_model.generate.call_args_list[1][0][0]
        self.assertEqual(len(second_call.messages), 3)  # user, assistant with tool_calls, tool result
        self.assertEqual(second_call.messages[2].role, "tool")
        self.assertIn("5", second_call.messages[2].content)

    def test_model_branch_calls_factory_and_returns_response(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_response = ChatResponse(
            message=Message(role="assistant", content="Hi there"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.return_value = fake_response

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create:
            mock_create.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_create.assert_called_once_with("gpt-4o-mini")
            fake_model.generate.assert_called_once()
        self.assertEqual(response.message.content, "Hi there")

    def test_model_branch_stream_yields_from_model(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        def fake_stream(req):
            yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id=req.context.run_id)
            yield StreamEvent(event_type="token", data={"text": "Hello"}, sequence=2, run_id=req.context.run_id)
            yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=req.context.run_id)

        fake_model = MagicMock()
        fake_model.stream.side_effect = fake_stream

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create:
            mock_create.return_value = fake_model
            events = list(SimpleChatPipeline().stream(request))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1].data.get("text"), "Hello")

    def test_run_no_user_message_delegates_to_model(self):
        """When there is no user message, pipeline delegates to model (model branch)."""
        request = ChatRequest(
            messages=[Message(role="assistant", content="Only assistant")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_response = ChatResponse(
            message=Message(role="assistant", content="Echo"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.return_value = fake_response
        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create:
            mock_create.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_create.assert_called_once_with("gpt-4o-mini")
        self.assertEqual(response.message.content, "Echo")

    def test_run_missing_model_raises_value_error(self):
        """When request.model is empty, pipeline raises ValueError."""
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=False,
            model="",
            context=RunContext.create(),
        )
        with self.assertRaises(ValueError) as ctx:
            SimpleChatPipeline().run(request)
        self.assertIn("model", str(ctx.exception))

    def test_stream_missing_model_raises_value_error(self):
        """When request.model is empty, stream raises ValueError."""
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            stream=True,
            model="",
            context=RunContext.create(),
        )
        with self.assertRaises(ValueError) as ctx:
            list(SimpleChatPipeline().stream(request))
        self.assertIn("model", str(ctx.exception))

    def test_run_max_iterations_strips_tools_and_returns(self):
        """When model keeps returning tool_calls, cap at max_tool_iterations then final generate."""
        mock_tool = self._make_mock_tool("search_documents")
        request = ChatRequest(
            messages=[Message(role="user", content="Loop")],
            stream=False,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="c1", name="search_documents", arguments={"a": 1, "b": 2})
        tool_response = ChatResponse(
            message=Message(role="assistant", content="", tool_calls=[tool_call]),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        final_response = ChatResponse(
            message=Message(role="assistant", content="Done."),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        # Return tool_calls for 3 rounds, then final response when tool_schemas stripped
        fake_model.generate.side_effect = [tool_response, tool_response, tool_response, final_response]

        pipeline = SimpleChatPipeline(max_tool_iterations=3)
        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            response = pipeline.run(request)

        # 3 tool rounds + 1 final (tool_schemas stripped to force text response)
        self.assertEqual(fake_model.generate.call_count, 4)
        last_call = fake_model.generate.call_args_list[3][0][0]
        self.assertIsNone(last_call.tool_schemas)
        self.assertEqual(response.message.content, "Done.")

    def test_run_unknown_tool_name_raises_value_error(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=False,
            model="gpt-4o-mini",
            tools=["nonexistent_tool"],
            context=RunContext.create(),
        )
        with self.assertRaises(ValueError) as ctx:
            SimpleChatPipeline().run(request)
        self.assertIn("nonexistent_tool", str(ctx.exception))

    def test_run_tool_error_sent_back_to_llm(self):
        """When tool._run() raises, error JSON is appended as tool message and LLM gets another turn."""
        # Create a tool that raises
        class _ErrorTool(ContextAwareTool):
            name: str = "search_documents"
            description: str = "A tool that errors"
            args_schema: type[BaseModel] = _MockInput
            def _run(self, a: float, b: float) -> str:
                raise ValueError("Invalid arguments")

        mock_tool = _ErrorTool()
        request = ChatRequest(
            messages=[Message(role="user", content="Use bad tool")],
            stream=False,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="c1", name="search_documents", arguments={"a": 0, "b": 0})
        tool_response = ChatResponse(
            message=Message(role="assistant", content="", tool_calls=[tool_call]),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        final_response = ChatResponse(
            message=Message(role="assistant", content="I apologize for the error."),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.side_effect = [tool_response, final_response]

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            response = SimpleChatPipeline().run(request)

        self.assertEqual(fake_model.generate.call_count, 2)
        second_req = fake_model.generate.call_args_list[1][0][0]
        tool_msg = second_req.messages[2]
        self.assertEqual(tool_msg.role, "tool")
        self.assertIn("error", tool_msg.content.lower())
        self.assertEqual(response.message.content, "I apologize for the error.")

    def test_stream_with_tools_emits_tool_start_and_tool_end(self):
        """When model returns tool_calls in stream path, tool_start/tool_end events are
        emitted and the final response streams tokens in real-time."""
        mock_tool = self._make_mock_tool("search_documents")
        mock_tool._return_value = '{"result": 3}'
        request = ChatRequest(
            messages=[Message(role="user", content="Add 1 and 2")],
            stream=True,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )

        def fake_tool_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="message_end", data={
                "content": "",
                "tool_calls": [{"id": "s1", "name": "search_documents", "arguments": {"a": 1, "b": 2}}],
            }, sequence=2, run_id="")

        def fake_final_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="token", data={"text": "The result is 3."}, sequence=2, run_id="")
            yield StreamEvent(event_type="message_end", data={"content": "The result is 3."}, sequence=3, run_id="")

        fake_model = MagicMock()
        fake_model.stream.side_effect = [fake_tool_stream(None), fake_final_stream(None)]

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            events = list(SimpleChatPipeline().stream(request))

        # Verify tool events
        tool_starts = [e for e in events if e.event_type == "tool_start"]
        tool_ends = [e for e in events if e.event_type == "tool_end"]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(len(tool_ends), 1)
        self.assertEqual(tool_starts[0].data["tool_name"], "search_documents")
        self.assertEqual(tool_starts[0].data["tool_call_id"], "s1")
        self.assertEqual(tool_starts[0].data["arguments"], {"a": 1, "b": 2})
        self.assertEqual(tool_ends[0].data["tool_name"], "search_documents")
        self.assertIn("3", tool_ends[0].data["result"])

        # Verify stream() was called (not generate()) for all iterations
        self.assertEqual(fake_model.stream.call_count, 2)
        fake_model.generate.assert_not_called()

        # Verify streamed tokens for final response
        tokens = [e for e in events if e.event_type == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].data["text"], "The result is 3.")

    def test_stream_tool_start_emitted_before_execution(self):
        """tool_start must be yielded BEFORE the tool runs, not after."""
        execution_log = []

        class _TimingTool(ContextAwareTool):
            name: str = "search_documents"
            description: str = "Records execution order"
            args_schema: type[BaseModel] = _MockInput
            def _run(self, a: float, b: float) -> str:
                execution_log.append("tool_executed")
                return '{"ok": true}'

        mock_tool = _TimingTool()
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            stream=True,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )

        def fake_tool_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="message_end", data={
                "content": "",
                "tool_calls": [{"id": "t1", "name": "search_documents", "arguments": {"a": 1, "b": 1}}],
            }, sequence=2, run_id="")

        def fake_final_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="token", data={"text": "Done"}, sequence=2, run_id="")
            yield StreamEvent(event_type="message_end", data={"content": "Done"}, sequence=3, run_id="")

        fake_model = MagicMock()
        fake_model.stream.side_effect = [fake_tool_stream(None), fake_final_stream(None)]

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            for event in SimpleChatPipeline().stream(request):
                if event.event_type == "tool_start":
                    execution_log.append("tool_start_yielded")
                elif event.event_type == "tool_end":
                    execution_log.append("tool_end_yielded")

        # tool_start must appear before tool_executed, and tool_end after
        self.assertEqual(execution_log, [
            "tool_start_yielded",
            "tool_executed",
            "tool_end_yielded",
        ])

    def test_stream_uses_real_streaming_for_all_iterations(self):
        """All iterations (tool-calling and final) should use stream(), not generate()."""
        request = ChatRequest(
            messages=[Message(role="user", content="test")],
            stream=True,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
        )
        mock_tool = self._make_mock_tool("search_documents")

        def fake_tool_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="message_end", data={
                "content": "",
                "tool_calls": [{"id": "t1", "name": "search_documents", "arguments": {"a": 1, "b": 1}}],
            }, sequence=2, run_id="")

        def fake_final_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="token", data={"text": "Answer"}, sequence=2, run_id="")
            yield StreamEvent(event_type="message_end", data={"content": "Answer"}, sequence=3, run_id="")

        fake_model = MagicMock()
        fake_model.stream.side_effect = [fake_tool_stream(None), fake_final_stream(None)]

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            list(SimpleChatPipeline().stream(request))

        # stream() used for all iterations, generate() never called
        self.assertEqual(fake_model.stream.call_count, 2)
        fake_model.generate.assert_not_called()

    def test_stream_respects_per_request_max_iterations(self):
        """Per-request max_tool_iterations override should work in streaming mode."""
        mock_tool = self._make_mock_tool("search_documents")
        request = ChatRequest(
            messages=[Message(role="user", content="Loop")],
            stream=True,
            model="gpt-4o-mini",
            tools=["search_documents"],
            context=RunContext.create(),
            params={"max_tool_iterations": 2},
        )

        def fake_tool_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="message_end", data={
                "content": "",
                "tool_calls": [{"id": "c1", "name": "search_documents", "arguments": {"a": 1, "b": 2}}],
            }, sequence=2, run_id="")

        def fake_final_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id="")
            yield StreamEvent(event_type="token", data={"text": "Capped."}, sequence=2, run_id="")
            yield StreamEvent(event_type="message_end", data={"content": "Capped."}, sequence=3, run_id="")

        fake_model = MagicMock()
        # 2 tool rounds + 1 final stream (after max iterations)
        fake_model.stream.side_effect = [
            fake_tool_stream(None), fake_tool_stream(None), fake_final_stream(None),
        ]

        pipeline = SimpleChatPipeline(max_tool_iterations=10)  # default=10 but request says 2
        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create, \
             self._patch_tool_registry(mock_tool):
            mock_create.return_value = fake_model
            events = list(pipeline.stream(request))

        # Should be 2 tool rounds + 1 final = 3 stream calls
        self.assertEqual(fake_model.stream.call_count, 3)
        tokens = [e for e in events if e.event_type == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].data["text"], "Capped.")

    def test_stream_final_response_is_real_streaming(self):
        """Final response (no tool calls) is emitted as message_start, token, message_end."""
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            stream=True,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        def fake_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id=req.context.run_id)
            yield StreamEvent(event_type="token", data={"text": "Hello"}, sequence=2, run_id=req.context.run_id)
            yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=req.context.run_id)

        fake_model = MagicMock()
        fake_model.stream.side_effect = fake_stream

        with patch("llm.pipelines.simple_chat.create_chat_model") as mock_create:
            mock_create.return_value = fake_model
            events = list(SimpleChatPipeline().stream(request))

        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].event_type, "token")
        self.assertEqual(events[1].data.get("text"), "Hello")
        self.assertEqual(events[2].event_type, "message_end")
