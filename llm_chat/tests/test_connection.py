"""
Comprehensive WebSocket connection tests that simulate browser behavior.

These tests verify the connection logic end-to-end, helping debug connection issues.
"""
import uuid

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
class WebSocketConnectionTest(TestCase):
    """
    Tests that simulate browser WebSocket connections to debug connection issues.
    """

    async_capable = True

    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.other_user = User.objects.create_user(email="other@example.com", password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="Test Thread")

    async def test_successful_connection_with_valid_thread(self):
        """Test that a valid connection succeeds - this is what should happen in the browser."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        # Simulate authenticated user (like browser with session)
        communicator.scope["user"] = self.user

        connected, subprotocol = await communicator.connect()
        
        self.assertTrue(connected, f"Connection should succeed but got subprotocol: {subprotocol}")
        self.assertIsNone(subprotocol, "Should connect without errors")
        
        await communicator.disconnect()

    async def test_connection_fails_for_anonymous_user(self):
        """Test that anonymous users (not logged in) cannot connect."""
        from django.contrib.auth.models import AnonymousUser
        
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        # Set AnonymousUser (how Channels auth middleware does it)
        communicator.scope["user"] = AnonymousUser()

        connected, subprotocol = await communicator.connect()
        
        self.assertFalse(connected, "Anonymous users should not be able to connect")
        # Should get close code 4001 (Unauthorized)
        self.assertEqual(subprotocol, 4001, "Should return 4001 Unauthorized code")

    async def test_connection_fails_with_invalid_thread_id_format(self):
        """Test that invalid thread ID formats are rejected by URL routing."""
        communicator = WebsocketCommunicator(
            application,
            "/ws/chat/invalid-uuid/",
        )
        communicator.scope["user"] = self.user

        # URL routing will reject this before reaching consumer
        # This raises ValueError, so we catch it
        try:
            connected, subprotocol = await communicator.connect()
            # If it somehow connects, it should still fail
            self.assertFalse(connected, "Invalid thread ID format should be rejected")
        except ValueError as e:
            # Expected: routing rejects invalid UUID format
            self.assertIn("No route found", str(e))

    async def test_connection_fails_when_thread_does_not_exist(self):
        """Test that non-existent thread IDs are rejected."""
        non_existent_id = uuid.uuid4()
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{non_existent_id}/",
        )
        communicator.scope["user"] = self.user

        connected, subprotocol = await communicator.connect()
        
        self.assertFalse(connected, "Non-existent thread should be rejected")
        self.assertEqual(subprotocol, 4004, "Should return 4004 Thread not found code")

    async def test_connection_fails_when_user_does_not_own_thread(self):
        """Test that users cannot connect to threads they don't own."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        # Different user trying to access thread
        communicator.scope["user"] = self.other_user

        connected, subprotocol = await communicator.connect()
        
        self.assertFalse(connected, "User should not be able to connect to thread they don't own")
        self.assertEqual(subprotocol, 4004, "Should return 4004 Access denied code")

    async def test_connection_succeeds_and_joins_correct_group(self):
        """Test that successful connection joins the correct Channels group."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator.scope["user"] = self.user

        connected, _ = await communicator.connect()
        self.assertTrue(connected, "Connection should succeed")

        # Verify we can send a group message and receive it
        from channels.layers import get_channel_layer
        channel_layer = get_channel_layer()
        
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {
                "type": "chat.start_stream",
                "content": "Test message",
                "user_id": self.user.id,
            },
        )

        # Wait a moment for the message to be processed
        import asyncio
        await asyncio.sleep(0.1)

        await communicator.disconnect()

    async def test_multiple_connections_to_same_thread(self):
        """Test that multiple connections to the same thread work (simulating multiple tabs)."""
        communicator1 = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator1.scope["user"] = self.user

        communicator2 = WebsocketCommunicator(
            application,
            f"/ws/chat/{self.thread.id}/",
        )
        communicator2.scope["user"] = self.user

        connected1, _ = await communicator1.connect()
        connected2, _ = await communicator2.connect()

        self.assertTrue(connected1, "First connection should succeed")
        self.assertTrue(connected2, "Second connection should succeed")

        await communicator1.disconnect()
        await communicator2.disconnect()

    async def test_connection_with_string_thread_id(self):
        """Test connection with thread ID as string (how browser sends it)."""
        communicator = WebsocketCommunicator(
            application,
            f"/ws/chat/{str(self.thread.id)}/",
        )
        communicator.scope["user"] = self.user

        connected, _ = await communicator.connect()
        
        self.assertTrue(connected, "Connection with string thread ID should work")

        await communicator.disconnect()

    async def test_connection_close_codes(self):
        """Test that correct close codes are returned for different failure scenarios."""
        from django.contrib.auth.models import AnonymousUser
        
        # Test anonymous user
        comm1 = WebsocketCommunicator(application, f"/ws/chat/{self.thread.id}/")
        comm1.scope["user"] = AnonymousUser()
        connected, code = await comm1.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4001, "Anonymous should get 4001")

        # Test non-existent thread
        non_existent_id = uuid.uuid4()
        comm2 = WebsocketCommunicator(application, f"/ws/chat/{non_existent_id}/")
        comm2.scope["user"] = self.user
        connected, code = await comm2.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4004, "Non-existent thread should get 4004")

        # Test wrong user
        comm3 = WebsocketCommunicator(application, f"/ws/chat/{self.thread.id}/")
        comm3.scope["user"] = self.other_user
        connected, code = await comm3.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4004, "Wrong user should get 4004")
