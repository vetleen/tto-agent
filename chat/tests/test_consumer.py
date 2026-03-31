"""Tests for ChatConsumer WebSocket consumer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.testing import WebsocketCommunicator
from django.test import override_settings
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase

from accounts.models import Membership, Organization
from chat.consumers import ChatConsumer
from chat.models import ChatAttachment, ChatCanvas, ChatMessage, ChatThread, ChatThreadDataRoom
from chat.routing import websocket_urlpatterns
from channels.routing import URLRouter
from documents.models import DataRoom

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


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ValidateDataRoomTests(TransactionTestCase):
    """Test _validate_data_room access control including shared rooms."""

    def setUp(self):
        self.owner = User.objects.create_user(email="owner@example.com", password="pass")
        self.colleague = User.objects.create_user(email="colleague@example.com", password="pass")
        self.outsider = User.objects.create_user(email="outsider@example.com", password="pass")
        self.org = Organization.objects.create(name="TestOrg", slug="testorg-consumer")
        Membership.objects.create(user=self.owner, org=self.org)
        Membership.objects.create(user=self.colleague, org=self.org)

    def _make_consumer(self, user):
        consumer = ChatConsumer()
        consumer.scope = {"user": user}
        consumer.user = user
        return consumer

    async def test_owner_can_access_own_room(self):
        room = await database_sync_to_async(DataRoom.objects.create)(
            name="Owner Room", slug="owner-room-consumer", created_by=self.owner,
        )
        consumer = self._make_consumer(self.owner)
        result = await consumer._validate_data_room(room.pk)
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], room.pk)

    async def test_colleague_can_access_shared_room(self):
        room = await database_sync_to_async(DataRoom.objects.create)(
            name="Shared Room", slug="shared-room-consumer",
            created_by=self.owner, is_shared=True,
        )
        consumer = self._make_consumer(self.colleague)
        result = await consumer._validate_data_room(room.pk)
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], room.pk)

    async def test_outsider_denied_shared_room(self):
        room = await database_sync_to_async(DataRoom.objects.create)(
            name="Shared Room", slug="shared-deny-consumer",
            created_by=self.owner, is_shared=True,
        )
        consumer = self._make_consumer(self.outsider)
        result = await consumer._validate_data_room(room.pk)
        self.assertIsNone(result)

    async def test_nonexistent_room_returns_none(self):
        consumer = self._make_consumer(self.owner)
        result = await consumer._validate_data_room(999999)
        self.assertIsNone(result)


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class CanvasOwnershipTests(TransactionTestCase):
    """Test that canvas operations enforce thread ownership."""

    def setUp(self):
        self.owner = User.objects.create_user(email="canvas-owner@example.com", password="pass")
        self.attacker = User.objects.create_user(email="canvas-attacker@example.com", password="pass")

    def _make_consumer(self, user):
        consumer = ChatConsumer()
        consumer.scope = {"user": user}
        consumer.user = user
        return consumer

    async def test_resolve_canvas_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Secret Doc", content="Confidential",
        )
        consumer = self._make_consumer(self.attacker)
        result = await database_sync_to_async(consumer._resolve_canvas_id)(
            str(thread.id), canvas.pk,
        )
        self.assertIsNone(result)

    async def test_resolve_canvas_allows_own_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="My Doc", content="OK",
        )
        consumer = self._make_consumer(self.owner)
        result = await database_sync_to_async(consumer._resolve_canvas_id)(
            str(thread.id), canvas.pk,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.pk, canvas.pk)

    async def test_resolve_canvas_rejects_cross_thread_active_canvas(self):
        """active_canvas pointing to another thread's canvas must not be returned."""
        thread_a = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        thread_b = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        canvas_b = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread_b, title="Thread B Canvas", content="Private",
        )
        # Simulate a stale/corrupt active_canvas FK pointing to another thread's canvas
        await database_sync_to_async(
            lambda: ChatThread.objects.filter(pk=thread_a.pk).update(active_canvas=canvas_b)
        )()
        consumer = self._make_consumer(self.owner)
        result = await database_sync_to_async(consumer._resolve_canvas_id)(
            str(thread_a.id),
        )
        # Must NOT return canvas_b (belongs to thread_b)
        if result is not None:
            self.assertNotEqual(result.pk, canvas_b.pk)

    async def test_load_all_canvases_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Secret", content="No",
        )
        consumer = self._make_consumer(self.attacker)
        result = await consumer._load_all_canvases(str(thread.id))
        self.assertIsNone(result)

    async def test_get_canvases_for_prompt_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Secret", content="No",
        )
        consumer = self._make_consumer(self.attacker)
        result = await consumer._get_canvases_for_prompt(str(thread.id))
        self.assertIsNone(result)

    async def test_switch_canvas_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        canvas = await database_sync_to_async(ChatCanvas.objects.create)(
            thread=thread, title="Secret", content="No",
        )
        consumer = self._make_consumer(self.attacker)
        result = await consumer._switch_canvas(str(thread.id), canvas.pk)
        self.assertIsNone(result)

    async def test_get_or_create_canvas_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        consumer = self._make_consumer(self.attacker)
        result = await consumer._get_or_create_canvas(str(thread.id))
        self.assertIsNone(result)
        # Ensure no canvas was created
        count = await database_sync_to_async(
            ChatCanvas.objects.filter(thread=thread).count
        )()
        self.assertEqual(count, 0)

    async def test_save_canvas_denies_other_users_thread(self):
        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.owner,
        )
        consumer = self._make_consumer(self.attacker)
        # _save_canvas is already wrapped with @database_sync_to_async
        await consumer._save_canvas(
            str(thread.id), "Injected", "Malicious content",
        )
        # Ensure no canvas was created
        count = await database_sync_to_async(
            ChatCanvas.objects.filter(thread=thread).count
        )()
        self.assertEqual(count, 0)


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class PayloadDataRoomValidationTests(TransactionTestCase):
    """Test that data_room_ids in message payload are validated before persisting."""

    def setUp(self):
        self.user = User.objects.create_user(email="payload@example.com", password="pass")
        self.other = User.objects.create_user(email="victim@example.com", password="pass")

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(app, "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    @patch("llm.get_llm_service")
    async def test_inaccessible_data_room_ids_filtered_from_payload(self, mock_get_service):
        """data_room_ids in payload must be validated — inaccessible rooms must not be linked."""
        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            return
            yield

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        # Create rooms: one owned by user, one by other user (inaccessible)
        own_room = await database_sync_to_async(DataRoom.objects.create)(
            name="My Room", slug="payload-own", created_by=self.user,
        )
        other_room = await database_sync_to_async(DataRoom.objects.create)(
            name="Secret Room", slug="payload-secret", created_by=self.other,
        )

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
            "data_room_ids": [own_room.pk, other_room.pk],
        })

        # Consume thread.created
        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "thread.created")
        thread_id = resp["thread_id"]

        # Verify only the accessible room was linked
        linked_room_ids = await database_sync_to_async(
            lambda: set(
                ChatThreadDataRoom.objects.filter(thread_id=thread_id)
                .values_list("data_room_id", flat=True)
            )
        )()
        self.assertIn(own_room.pk, linked_room_ids)
        self.assertNotIn(other_room.pk, linked_room_ids)

        await communicator.disconnect()

    @patch("llm.get_llm_service")
    async def test_inaccessible_skill_id_not_attached_from_payload(self, mock_get_service):
        """skill_id in payload must be validated — inaccessible skills must not be attached."""
        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            return
            yield

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        from agent_skills.models import AgentSkill

        # Create an org-level skill that belongs to a different org
        other_org = await database_sync_to_async(Organization.objects.create)(
            name="OtherOrg", slug="otherorg-skill-test",
        )
        skill = await database_sync_to_async(AgentSkill.objects.create)(
            name="Secret Skill", slug="secret-skill",
            level="org", organization=other_org,
            instructions="Secret instructions",
        )

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "Hello",
            "skill_id": str(skill.pk),
        })

        resp = await communicator.receive_json_from(timeout=5)
        self.assertEqual(resp["event_type"], "thread.created")
        thread_id = resp["thread_id"]

        # Verify the skill was NOT attached to the thread
        thread_skill = await database_sync_to_async(
            lambda: ChatThread.objects.get(pk=thread_id).skill_id
        )()
        self.assertIsNone(thread_skill)

        await communicator.disconnect()


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class LoadAttachmentsAccessTests(TransactionTestCase):
    """Test that _load_attachments only returns the user's own attachments."""

    def setUp(self):
        self.user = User.objects.create_user(email="att-owner@example.com", password="pass")
        self.other = User.objects.create_user(email="att-other@example.com", password="pass")

    def _make_consumer(self, user):
        consumer = ChatConsumer()
        consumer.scope = {"user": user}
        consumer.user = user
        return consumer

    async def test_load_attachments_excludes_other_users(self):
        """_load_attachments must not return attachments uploaded by other users."""
        from django.core.files.uploadedfile import SimpleUploadedFile

        own_thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user,
        )
        other_thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.other,
        )

        own_att = await database_sync_to_async(ChatAttachment.objects.create)(
            thread=own_thread,
            uploaded_by=self.user,
            file=SimpleUploadedFile("mine.txt", b"my data", content_type="text/plain"),
            original_filename="mine.txt",
            content_type="text/plain",
            size_bytes=7,
        )
        other_att = await database_sync_to_async(ChatAttachment.objects.create)(
            thread=other_thread,
            uploaded_by=self.other,
            file=SimpleUploadedFile("secret.txt", b"secret", content_type="text/plain"),
            original_filename="secret.txt",
            content_type="text/plain",
            size_bytes=6,
        )

        consumer = self._make_consumer(self.user)
        result = await consumer._load_attachments([str(own_att.id), str(other_att.id)])
        self.assertIn(str(own_att.id), result)
        self.assertNotIn(str(other_att.id), result)
