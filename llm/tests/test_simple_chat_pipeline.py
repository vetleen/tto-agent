"""Tests for the simple_chat pipeline (tool loop and model delegation)."""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.types.context import RunContext
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_registry.return_value.get_model.assert_called_once_with("gpt-4o-mini")
            fake_model.generate.assert_called_once()
            # Request should not have tool_schemas
            call_args = fake_model.generate.call_args[0][0]
            self.assertIsNone(call_args.tool_schemas)
        self.assertEqual(response.message.content, "Hi there")

    def test_run_tool_loop_executes_tool_and_returns_final_response(self):
        """Model returns tool_calls on first call, then text on second; tool is executed."""
        request = ChatRequest(
            messages=[Message(role="user", content="What is 2 + 3?")],
            stream=False,
            model="gpt-4o-mini",
            tools=["add_number"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="call_1", name="add_number", arguments={"a": 2, "b": 3})
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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
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

    def test_model_branch_calls_registry_and_returns_response(self):
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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_registry.return_value.get_model.assert_called_once_with("gpt-4o-mini")
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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
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
        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_registry.return_value.get_model.assert_called_once_with("gpt-4o-mini")
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
        request = ChatRequest(
            messages=[Message(role="user", content="Loop")],
            stream=False,
            model="gpt-4o-mini",
            tools=["add_number"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="c1", name="add_number", arguments={"a": 1, "b": 2})
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
        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = pipeline.run(request)

        # 3 tool rounds + 1 final (tool_schemas stripped)
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
        """When tool.run() raises, error JSON is appended as tool message and LLM gets another turn."""
        request = ChatRequest(
            messages=[Message(role="user", content="Use bad tool")],
            stream=False,
            model="gpt-4o-mini",
            tools=["add_number"],
            context=RunContext.create(),
        )
        tool_call = ToolCall(id="c1", name="add_number", arguments={"a": "not", "b": "numbers"})
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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = SimpleChatPipeline().run(request)

        self.assertEqual(fake_model.generate.call_count, 2)
        second_req = fake_model.generate.call_args_list[1][0][0]
        tool_msg = second_req.messages[2]
        self.assertEqual(tool_msg.role, "tool")
        self.assertIn("error", tool_msg.content.lower())
        self.assertEqual(response.message.content, "I apologize for the error.")

    def test_stream_with_tools_emits_tool_start_and_tool_end(self):
        """When model returns tool_calls in stream path, tool_start/tool_end events are
        emitted and the final response is truly streamed via chat_model.stream()."""
        request = ChatRequest(
            messages=[Message(role="user", content="Add 1 and 2")],
            stream=True,
            model="gpt-4o-mini",
            tools=["add_number"],
            context=RunContext.create(),
        )
        run_id = request.context.run_id
        tool_call = ToolCall(id="s1", name="add_number", arguments={"a": 1, "b": 2})
        tool_response = ChatResponse(
            message=Message(role="assistant", content="", tool_calls=[tool_call]),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        # Second generate() returns no tool_calls → triggers stream() for final output
        no_tool_response = ChatResponse(
            message=Message(role="assistant", content="The result is 3."),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )

        def fake_stream(req):
            yield StreamEvent(event_type="message_start", data={"model": "gpt-4o-mini"}, sequence=1, run_id=run_id)
            yield StreamEvent(event_type="token", data={"text": "The result is 3."}, sequence=2, run_id=run_id)
            yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id)

        fake_model = MagicMock()
        fake_model.generate.side_effect = [tool_response, no_tool_response]
        fake_model.stream.side_effect = fake_stream

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            events = list(SimpleChatPipeline().stream(request))

        # Verify tool events
        tool_starts = [e for e in events if e.event_type == "tool_start"]
        tool_ends = [e for e in events if e.event_type == "tool_end"]
        self.assertEqual(len(tool_starts), 1)
        self.assertEqual(len(tool_ends), 1)
        self.assertEqual(tool_starts[0].data["tool_name"], "add_number")
        self.assertEqual(tool_starts[0].data["tool_call_id"], "s1")
        self.assertEqual(tool_starts[0].data["arguments"], {"a": 1, "b": 2})
        self.assertEqual(tool_ends[0].data["tool_name"], "add_number")
        self.assertIn("3", tool_ends[0].data["result"])

        # Verify final response came from stream(), not generate()
        fake_model.stream.assert_called_once()
        # The stream call should have tool_schemas stripped
        stream_req = fake_model.stream.call_args[0][0]
        self.assertIsNone(stream_req.tool_schemas)

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

        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            events = list(SimpleChatPipeline().stream(request))

        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].event_type, "token")
        self.assertEqual(events[1].data.get("text"), "Hello")
        self.assertEqual(events[2].event_type, "message_end")
