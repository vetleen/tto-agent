"""Tests for the simple_chat pipeline (tool shortcut and model delegation)."""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from llm.pipelines.simple_chat import SimpleChatPipeline
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class SimpleChatPipelineTests(TestCase):
    """Test simple_chat pipeline: tool branch and model branch."""

    def test_tool_branch_add_number_returns_result_5(self):
        request = ChatRequest(
            messages=[Message(role="user", content="tool:add_number a=2 b=3")],
            stream=False,
            model="gpt-4o",  # ignored for tool path
            context=RunContext.create(),
        )
        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            response = SimpleChatPipeline().run(request)
            mock_registry.return_value.get_model.assert_not_called()
        self.assertEqual(response.message.role, "assistant")
        self.assertEqual(response.message.content, "Result: 5.0")

    def test_tool_branch_stream_yields_expected_events(self):
        request = ChatRequest(
            messages=[Message(role="user", content="tool:add_number a=1 b=2")],
            stream=True,
            model="gpt-4o",
            context=RunContext.create(),
        )
        with patch("llm.pipelines.simple_chat.get_model_registry"):
            events = list(SimpleChatPipeline().stream(request))
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[0].event_type, "message_start")
        self.assertEqual(events[1].event_type, "token")
        self.assertIn("3", events[1].data.get("text", ""))
        self.assertEqual(events[2].event_type, "message_end")

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

    def test_run_malformed_tool_string_delegates_to_model(self):
        """Malformed tool string (e.g. missing b=) does not match; model branch is used."""
        request = ChatRequest(
            messages=[Message(role="user", content="tool:add_number a=1")],
            stream=False,
            model="gpt-4o-mini",
            context=RunContext.create(),
        )
        fake_response = ChatResponse(
            message=Message(role="assistant", content="Sure"),
            model="gpt-4o-mini",
            usage=None,
            metadata={},
        )
        fake_model = MagicMock()
        fake_model.generate.return_value = fake_response
        with patch("llm.pipelines.simple_chat.get_model_registry") as mock_registry:
            mock_registry.return_value.get_model.return_value = fake_model
            response = SimpleChatPipeline().run(request)
            mock_registry.return_value.get_model.assert_called_once()
        self.assertEqual(response.message.content, "Sure")

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
