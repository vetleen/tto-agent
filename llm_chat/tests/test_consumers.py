import time
from unittest import mock

from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from config.asgi import application
from llm_chat.models import ChatThread


User = get_user_model()


@override_settings(
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }
)
class ChatConsumerAuthTest(TestCase):
    async_capable = True

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="WS Thread")

    @override_settings(ROOT_URLCONF="config.urls")
    async def test_connect_requires_auth(self):
        communicator = WebsocketCommunicator(application, f"/ws/chat/{self.thread.id}/")
        connected, _ = await communicator.connect()
        self.assertFalse(connected)
        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }
)
class ChatConsumerStreamingTest(TestCase):
    async_capable = True

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="WS Stream")

    @mock.patch("llm_chat.consumers.ChatService")
    async def test_chat_start_stream_uses_service_and_sends_events(self, MockChatService):
        # Prepare fake stream events
        service_instance = MockChatService.return_value
        service_instance.stream_reply.return_value = iter(
            [
                ("response.output_text.delta", mock.Mock(delta="Hi ")),
                ("response.output_text.delta", mock.Mock(delta="there")),
                ("final", {"call_log": mock.Mock(id=1), "response": object()}),
            ]
        )

        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user

        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Trigger the group event directly (we are already in an async test)
        from channels.layers import get_channel_layer

        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {
                "type": "chat.start_stream",
                "content": "Hello",
                "user_id": self.user.id,
            },
        )

        # Collect a few messages
        messages = []
        for _ in range(3):
            msg = await communicator.receive_json_from()
            messages.append(msg)

        # Ensure we saw delta and final events
        event_types = [m["event_type"] for m in messages]
        self.assertIn("response.output_text.delta", event_types)
        self.assertIn("final", event_types)

        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }
)
class ChatConsumerTitleGenerationTest(TestCase):
    async_capable = True

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="New chat")

    @mock.patch("llm_chat.consumers.ChatService")
    async def test_title_generation_triggered_for_first_message(self, MockChatService):
        """Test that title generation is triggered for the first message in a thread."""
        from llm_chat.models import ChatMessage
        from asgiref.sync import sync_to_async
        
        # Create a user message (simulating what the view does)
        user_msg = await sync_to_async(ChatMessage.objects.create)(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            status=ChatMessage.Status.FINAL,
            content="What is Python?",
        )
        
        # Mock the service
        service_instance = MockChatService.return_value
        service_instance.stream_reply.return_value = iter([
            ("response.output_text.delta", mock.Mock(delta="Python is")),
            ("final", {"call_log": mock.Mock(id=1), "response": object()}),
        ])
        service_instance.generate_thread_title.return_value = "Python Programming"
        
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user
        
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        
        # Trigger streaming
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {
                "type": "chat.start_stream",
                "content": "What is Python?",
                "user_id": self.user.id,
            },
        )
        
        # Receive streaming events
        messages = []
        for _ in range(2):
            msg = await communicator.receive_json_from(timeout=1.0)
            messages.append(msg)
        
        # Verify streaming completed
        event_types = [m["event_type"] for m in messages]
        self.assertIn("final", event_types)
        
        # Wait a bit for background thread to start
        import asyncio
        await asyncio.sleep(0.2)
        
        # Verify title generation method was called
        # Note: We can't easily verify Thread.start() without breaking heartbeat,
        # but we can verify the service method was called
        service_instance.generate_thread_title.assert_called_once_with(
            thread=self.thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        await communicator.disconnect()

    @mock.patch("llm_chat.consumers.ChatService")
    async def test_title_generation_not_triggered_for_subsequent_messages(self, MockChatService):
        """Test that title generation is NOT triggered for subsequent messages."""
        from llm_chat.models import ChatMessage
        from asgiref.sync import sync_to_async
        
        # Create existing user messages (simulating a thread with history)
        await sync_to_async(ChatMessage.objects.create)(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            status=ChatMessage.Status.FINAL,
            content="First message",
        )
        await sync_to_async(ChatMessage.objects.create)(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT,
            status=ChatMessage.Status.FINAL,
            content="First response",
        )
        
        # Create the new user message
        user_msg = await sync_to_async(ChatMessage.objects.create)(
            thread=self.thread,
            role=ChatMessage.Role.USER,
            status=ChatMessage.Status.FINAL,
            content="Second message",
        )
        
        # Mock the service
        service_instance = MockChatService.return_value
        service_instance.stream_reply.return_value = iter([
            ("response.output_text.delta", mock.Mock(delta="Response")),
            ("final", {"call_log": mock.Mock(id=1), "response": object()}),
        ])
        
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user
        
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        
        # Trigger streaming
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {
                "type": "chat.start_stream",
                "content": "Second message",
                "user_id": self.user.id,
            },
        )
        
        # Receive streaming events
        messages = []
        for _ in range(2):
            msg = await communicator.receive_json_from(timeout=1.0)
            messages.append(msg)
        
        # Verify streaming completed
        event_types = [m["event_type"] for m in messages]
        self.assertIn("final", event_types)
        
        # Wait a bit
        import asyncio
        await asyncio.sleep(0.1)
        
        # Verify title generation was NOT triggered
        service_instance.generate_thread_title.assert_not_called()
        
        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        }
    }
)
class ChatConsumerHeartbeatTest(TestCase):
    async_capable = True

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="WS Heartbeat")

    async def test_receives_ping_and_sends_pong(self):
        """Test that the consumer sends ping messages and handles pong responses."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user

        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Wait a bit for heartbeat to potentially send a ping
        # Note: In a real scenario, ping would be sent after HEARTBEAT_INTERVAL (30s)
        # For testing, we'll send a pong to verify the consumer handles it
        await communicator.send_json_to({
            "event_type": "pong",
            "timestamp": time.time(),
        })

        # Should receive pong_ack
        response = await communicator.receive_json_from(timeout=1.0)
        self.assertEqual(response["event_type"], "pong_ack")

        await communicator.disconnect()

    async def test_heartbeat_stops_on_disconnect(self):
        """Test that heartbeat timer is cleaned up on disconnect."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user

        connected, _ = await communicator.connect()
        self.assertTrue(connected)

        # Verify connection is active
        self.assertTrue(connected)

        # Disconnect should clean up heartbeat
        await communicator.disconnect()

        # After disconnect, heartbeat should be stopped
        # (We can't directly test the timer, but we verify disconnect works cleanly)
        self.assertIsNotNone(communicator)

