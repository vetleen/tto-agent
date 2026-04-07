"""Tests for hidden seed messages and the pending_initial_turn dispatch path
in the chat consumer + view.

These cover the consumer-side half of the "edit skill in chat" feature; the
agent_skills side (view, fork-on-write) lives in
``agent_skills.tests.test_edit_in_chat``.
"""

from unittest.mock import MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings
from django.urls import reverse

from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns

User = get_user_model()


def make_application():
    return URLRouter(websocket_urlpatterns)


# ----- View-side rendering ----------------------------------------------


@override_settings(ALLOWED_HOSTS=["testserver"])
class HiddenMessageRenderTests(TestCase):
    """The chat_home view must filter is_hidden_from_user messages out."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="hidden@example.com", password="pw"
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])

    def test_hidden_message_excluded_from_chat_home(self):
        thread = ChatThread.objects.create(created_by=self.user)
        ChatMessage.objects.create(
            thread=thread,
            role="user",
            content="VISIBLE_USER_MESSAGE_TOKEN",
        )
        ChatMessage.objects.create(
            thread=thread,
            role="user",
            content="HIDDEN_SEED_TOKEN",
            is_hidden_from_user=True,
        )

        self.client.force_login(self.user)
        response = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "VISIBLE_USER_MESSAGE_TOKEN")
        self.assertNotContains(response, "HIDDEN_SEED_TOKEN")


# ----- Consumer-side dispatch -------------------------------------------


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class PendingInitialTurnDispatchTests(TransactionTestCase):
    """Loading a thread with pending_initial_turn=True triggers exactly one
    assistant turn against the existing seed message and clears the flag."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="seed@example.com", password="pw"
        )

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(app, "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    @patch("llm.get_llm_service")
    async def test_pending_flag_triggers_assistant_turn_and_clears(self, mock_get_service):
        from llm.types.streaming import StreamEvent

        events = [
            StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
            StreamEvent(event_type="token", data={"text": "Hi there"}, sequence=1, run_id="r1"),
            StreamEvent(event_type="message_end", data={}, sequence=2, run_id="r1"),
        ]

        astream_calls = {"count": 0}
        mock_service = MagicMock()

        async def mock_astream(*args, **kwargs):
            astream_calls["count"] += 1
            for e in events:
                yield e

        mock_service.astream = mock_astream
        mock_get_service.return_value = mock_service

        # Pre-create the thread with a hidden seed message and pending flag,
        # mirroring what skills_edit_in_chat does.
        @database_sync_to_async
        def setup():
            thread = ChatThread.objects.create(
                created_by=self.user,
                title="Editing X",
                metadata={"pending_initial_turn": True},
            )
            ChatMessage.objects.create(
                thread=thread,
                role="user",
                content="Greet the user about skill X.",
                is_hidden_from_user=True,
            )
            return thread

        thread = await setup()

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.load_thread",
            "thread_id": str(thread.id),
        })

        # Receive the expected sequence: thread.loaded, then the streamed
        # assistant turn (message_start, token, message_end), then post-stream
        # cost update. We assert types in order to verify the dispatch path.
        loaded = await communicator.receive_json_from(timeout=5)
        self.assertEqual(loaded.get("event_type"), "thread.loaded")

        msg_start = await communicator.receive_json_from(timeout=5)
        self.assertEqual(msg_start.get("event_type"), "message_start")

        tok = await communicator.receive_json_from(timeout=5)
        self.assertEqual(tok.get("event_type"), "token")
        self.assertEqual(tok["data"]["text"], "Hi there")

        msg_end = await communicator.receive_json_from(timeout=5)
        self.assertEqual(msg_end.get("event_type"), "message_end")

        cost = await communicator.receive_json_from(timeout=5)
        self.assertEqual(cost.get("event_type"), "thread.cost_updated")

        # The LLM was called exactly once (the seed turn).
        self.assertEqual(astream_calls["count"], 1)

        # Flag has been cleared on the thread — guarantees that any future
        # reconnect goes through ``_consume_pending_initial_turn`` and finds
        # nothing pending, so no double-trigger is possible.
        thread = await database_sync_to_async(
            lambda: ChatThread.objects.get(pk=thread.pk)
        )()
        self.assertFalse(thread.metadata.get("pending_initial_turn", False))

        # 1 hidden user (seed) + 1 assistant reply = 2
        msg_count = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(thread=thread).count()
        )()
        self.assertEqual(msg_count, 2)

        assistant = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(thread=thread, role="assistant").first()
        )()
        self.assertIsNotNone(assistant)
        self.assertEqual(assistant.content, "Hi there")


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ConsumePendingInitialTurnUnitTests(TransactionTestCase):
    """The ``_consume_pending_initial_turn`` helper is the idempotency
    guarantee — even with the consumer mocked out, calling it twice on the
    same thread should only ever return True once."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="cpit@example.com", password="pw"
        )

    async def test_atomic_read_and_clear(self):
        from chat.consumers import ChatConsumer

        @database_sync_to_async
        def setup():
            return ChatThread.objects.create(
                created_by=self.user,
                metadata={"pending_initial_turn": True, "source_skill_id": "abc"},
            )

        thread = await setup()

        consumer = ChatConsumer()
        consumer.user = self.user

        first = await consumer._consume_pending_initial_turn(str(thread.id))
        second = await consumer._consume_pending_initial_turn(str(thread.id))
        self.assertTrue(first)
        self.assertFalse(second)

        @database_sync_to_async
        def reload():
            t = ChatThread.objects.get(pk=thread.pk)
            return t.metadata

        meta = await reload()
        self.assertNotIn("pending_initial_turn", meta)
        # Other metadata keys are preserved.
        self.assertEqual(meta.get("source_skill_id"), "abc")
