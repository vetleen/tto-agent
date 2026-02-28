"""Tests for ProjectChatConsumer WebSocket consumer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.test import override_settings
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase

from chat.consumers import ProjectChatConsumer
from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from channels.routing import URLRouter
from documents.models import Project

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
        self.project = Project.objects.create(
            name="Test", slug="test-project", created_by=self.user
        )
        self.other_user = User.objects.create_user(
            email="other@example.com", password="pass123"
        )

    async def _communicator(self, project_uuid, user=None):
        app = make_application()
        communicator = WebsocketCommunicator(
            app, f"/ws/projects/{project_uuid}/chat/"
        )
        if user:
            communicator.scope["user"] = user
        return communicator

    async def test_unauthenticated_rejected(self):
        from django.contrib.auth.models import AnonymousUser

        communicator = await self._communicator(self.project.uuid, AnonymousUser())
        connected, code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4401)
        await communicator.disconnect()

    async def test_wrong_owner_rejected(self):
        communicator = await self._communicator(self.project.uuid, self.other_user)
        connected, code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4404)
        await communicator.disconnect()

    async def test_connect_succeeds(self):
        communicator = await self._communicator(self.project.uuid, self.user)
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        await communicator.disconnect()

    async def test_nonexistent_project_rejected(self):
        import uuid

        fake_uuid = uuid.uuid4()
        communicator = await self._communicator(fake_uuid, self.user)
        connected, code = await communicator.connect()
        self.assertFalse(connected)
        self.assertEqual(code, 4404)
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
        self.project = Project.objects.create(
            name="Test", slug="test-project", created_by=self.user
        )

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(
            app, f"/ws/projects/{self.project.uuid}/chat/"
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
