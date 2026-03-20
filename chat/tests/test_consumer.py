"""Tests for ChatConsumer WebSocket consumer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.test import override_settings
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase

from chat.consumers import ChatConsumer
from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from channels.routing import URLRouter

User = get_user_model()


def make_application():
    return URLRouter(websocket_urlpatterns)


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerConnectTests(TransactionTestCase):
    """Test WebSocket connection handling."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com", password="pass123"
        )

    async def _communicator(self, user=None):
        app = make_application()
        communicator = WebsocketCommunicator(
            app, "/ws/chat/"
        )
        if user:
            communicator.scope["user"] = user
        return communicator

    async def test_unauthenticated_rejected(self):
        from django.contrib.auth.models import AnonymousUser

        communicator = await self._communicator(AnonymousUser())
        connected, code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4401)
        await communicator.disconnect()

    async def test_connect_succeeds(self):
        communicator = await self._communicator(self.user)
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumerMessageTests(TransactionTestCase):
    """Test message handling in the consumer."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com", password="pass123"
        )

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(
            app, "/ws/chat/"
        )
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    @patch("llm.get_llm_service")
    async def test_message_creates_thread(self, mock_get_service):
        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            return
            yield  # make it an async generator

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
        })

        # Should receive thread.created event
        response = await communicator.receive_json_from(timeout=5)
        self.assertEqual(response["event_type"], "thread.created")
        self.assertIn("thread_id", response)

        # Verify thread and message were persisted
        thread_count = await database_sync_to_async(ChatThread.objects.count)()
        self.assertEqual(thread_count, 1)

        msg_count = await database_sync_to_async(ChatMessage.objects.count)()
        self.assertEqual(msg_count, 1)

        await communicator.disconnect()

    async def test_empty_message_rejected(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "   ",
        })

        response = await communicator.receive_json_from(timeout=5)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Empty message")
        await communicator.disconnect()

    @patch("llm.get_llm_service")
    async def test_stream_events_forwarded(self, mock_get_service):
        from llm.types.streaming import StreamEvent

        events = [
            StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=1, run_id="r1"),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id="r1"),
        ]

        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            for e in events:
                yield e

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
        })

        # First: thread.created
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "thread.created")

        # Then: the 3 stream events
        resp1 = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp1["event_type"], "message_start")

        resp2 = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp2["event_type"], "token")
        self.assertEqual(resp2["data"]["text"], "Hi")

        resp3 = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp3["event_type"], "message_end")

        # Assistant message should be persisted from streamed tokens
        assistant = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="assistant").first()
        )()
        self.assertIsNotNone(assistant)
        self.assertEqual(assistant.content, "Hi")

        await communicator.disconnect()

    @patch("llm.get_llm_service")
    async def test_tool_messages_persisted(self, mock_get_service):
        """Tool call and tool result messages are persisted during streaming."""
        from llm.types.streaming import StreamEvent

        events = [
            StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
            StreamEvent(event_type="tool_start", data={
                "tool_call_id": "tc1", "tool_name": "search", "arguments": {"q": "test"},
            }, sequence=1, run_id="r1"),
            StreamEvent(event_type="tool_end", data={
                "tool_call_id": "tc1", "tool_name": "search", "result": "found it",
            }, sequence=2, run_id="r1"),
            # New LLM turn after tool loop
            StreamEvent(event_type="message_start", data={}, sequence=3, run_id="r1"),
            StreamEvent(event_type="token", data={"text": "Here"}, sequence=4, run_id="r1"),
            StreamEvent(event_type="message_end", data={}, sequence=5, run_id="r1"),
        ]

        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            for e in events:
                yield e

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
        })

        # Consume all events
        for _ in range(10):
            try:
                await communicator.receive_json_from(timeout=2)
            except Exception:
                break

        # Verify tool messages persisted
        assistant_with_tools = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(
                role="assistant", metadata__tool_calls__isnull=False
            ).exclude(metadata={}).first()
        )()
        self.assertIsNotNone(assistant_with_tools)
        self.assertEqual(assistant_with_tools.metadata["tool_calls"][0]["name"], "search")

        tool_msg = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="tool").first()
        )()
        self.assertIsNotNone(tool_msg)
        self.assertEqual(tool_msg.tool_call_id, "tc1")
        self.assertEqual(tool_msg.content, "found it")

        # Final assistant message
        final_assistant = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="assistant", content="Here").first()
        )()
        self.assertIsNotNone(final_assistant)

        try:
            await communicator.disconnect()
        except BaseException:
            pass  # CancelledError on disconnect is expected in tests

    @patch("llm.get_llm_service")
    async def test_thinking_persisted_in_metadata(self, mock_get_service):
        """Thinking content is stored in assistant message metadata."""
        from llm.types.streaming import StreamEvent

        events = [
            StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
            StreamEvent(event_type="thinking", data={"text": "Let me think..."}, sequence=1, run_id="r1"),
            StreamEvent(event_type="thinking", data={"text": " about this."}, sequence=2, run_id="r1"),
            StreamEvent(event_type="token", data={"text": "Answer"}, sequence=3, run_id="r1"),
            StreamEvent(event_type="message_end", data={}, sequence=4, run_id="r1"),
        ]

        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            for e in events:
                yield e

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
        })

        # Consume all events
        for _ in range(10):
            try:
                await communicator.receive_json_from(timeout=2)
            except Exception:
                break

        assistant = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="assistant", content="Answer").first()
        )()
        self.assertIsNotNone(assistant)
        self.assertEqual(assistant.metadata.get("thinking"), "Let me think... about this.")

        try:
            await communicator.disconnect()
        except BaseException:
            pass

    @patch("llm.get_llm_service")
    async def test_no_thinking_means_no_metadata_key(self, mock_get_service):
        """When there's no thinking, metadata should not contain 'thinking' key."""
        from llm.types.streaming import StreamEvent

        events = [
            StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
            StreamEvent(event_type="token", data={"text": "Hi"}, sequence=1, run_id="r1"),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id="r1"),
        ]

        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            for e in events:
                yield e

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
        })

        # Consume all events
        for _ in range(10):
            try:
                await communicator.receive_json_from(timeout=2)
            except Exception:
                break

        assistant = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="assistant", content="Hi").first()
        )()
        self.assertIsNotNone(assistant)
        self.assertNotIn("thinking", assistant.metadata)

        try:
            await communicator.disconnect()
        except BaseException:
            pass

    @patch("llm.get_llm_service")
    async def test_user_message_persisted(self, mock_get_service):
        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            return
            yield

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Test message",
        })

        # Consume thread.created
        await communicator.receive_json_from(timeout=5)

        # Verify user message persisted
        msg = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(role="user").first()
        )()
        self.assertIsNotNone(msg)
        self.assertEqual(msg.content, "Test message")

        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class SlashCommandTests(TransactionTestCase):
    """Test slash command handling in the consumer."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="slash@example.com", password="pass123"
        )

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(app, "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    async def _create_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user, title="Test Thread"
        )
        return str(thread.id)

    async def test_unknown_command_returns_error(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/foo",
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "error")
        self.assertIn("/foo", resp["message"])
        self.assertIn("/clear", resp["message"])
        await communicator.disconnect()

    async def test_clear_command(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/clear",
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["action"], "navigate")
        await communicator.disconnect()

    async def test_cost_no_thread(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/cost",
            "thread_id": "",
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "ok")
        self.assertIn("$0.00", resp["message"])
        await communicator.disconnect()

    async def test_cost_with_thread(self):
        thread_id = await self._create_thread()
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/cost",
            "thread_id": thread_id,
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "ok")
        self.assertIn("$", resp["message"])
        await communicator.disconnect()

    async def test_tag_with_emoji(self):
        thread_id = await self._create_thread()
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/tag 🚀",
            "thread_id": thread_id,
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["emoji"], "🚀")
        self.assertEqual(resp["thread_id"], thread_id)
        # Verify DB
        thread = await database_sync_to_async(
            ChatThread.objects.get
        )(pk=thread_id)
        self.assertEqual(thread.emoji, "🚀")
        await communicator.disconnect()

    async def test_tag_no_thread(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/tag",
            "thread_id": "",
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "error")
        self.assertIn("No active thread", resp["message"])
        await communicator.disconnect()

    async def test_compact_no_thread(self):
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/compact",
            "thread_id": "",
        })
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "command.result")
        self.assertEqual(resp["status"], "error")
        self.assertIn("No active thread", resp["message"])
        await communicator.disconnect()

    async def test_slash_command_not_saved_as_message(self):
        thread_id = await self._create_thread()
        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "/cost",
            "thread_id": thread_id,
        })
        await communicator.receive_json_from(timeout=5)
        msg_count = await database_sync_to_async(ChatMessage.objects.count)()
        self.assertEqual(msg_count, 0)
        await communicator.disconnect()
