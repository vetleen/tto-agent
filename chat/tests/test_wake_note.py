"""A server-seeded wake-up turn never starts on the agent's own messages.

2026-10-15 prod: a sub-agent outcome that landed mid-stream sorted before the
previous turn's last tool round, the woken turn's history ended on an assistant
reply, and Gemini rejected its second tool round ("missing a thought_signature")
— the per-request provider repair only covered rounds that end on an assistant
message. The wake note is appended once at turn start, so every round carries it.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat.models import ChatMessage, ChatThread
from chat.prompts import SUBAGENT_WAKE_NOTE
from llm.types.streaming import StreamEvent

User = get_user_model()


class _CapturingService:
    def __init__(self):
        self.requests = []

    async def astream(self, pipeline_id, request, **kwargs):
        self.requests.append(request)
        for i, (event_type, data) in enumerate([
            ("message_start", {}), ("token", {"text": "Carrying on."}), ("message_end", {}),
        ]):
            yield StreamEvent(event_type=event_type, data=data, sequence=i, run_id="r1")


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class SeededTurnWakeNoteTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="wake@example.com", password="pass123")
        from guardrails.service import GuardrailVerdict

        gr = patch(
            "guardrails.service.run_classifier_pipeline",
            new_callable=AsyncMock,
            return_value=GuardrailVerdict(action="allow"),
        )
        gr.start()
        self.addCleanup(gr.stop)

    @database_sync_to_async
    def _thread_ending_on_assistant(self):
        thread = ChatThread.objects.create(created_by=self.user, title="woken")
        ChatMessage.objects.create(thread=thread, role="user", content="Fill the table")
        ChatMessage.objects.create(
            thread=thread, role="assistant", content="",
            metadata={"tool_calls": [{"id": "c1", "name": "canvas_edit", "arguments": {}}]},
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(
            thread=thread, role="tool", content='{"status": "ok"}', tool_call_id="c1",
            is_hidden_from_user=True,
        )
        ChatMessage.objects.create(thread=thread, role="assistant", content="Updated the table.")
        return thread

    async def _consumer(self, thread):
        """A real, connected consumer (connect() sets the state a turn needs)
        with the transport mocked out."""
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.scope = {"user": self.user}
        consumer.channel_name = "test-channel"
        consumer.channel_layer = MagicMock()
        consumer.channel_layer.group_add = AsyncMock()
        consumer.channel_layer.group_discard = AsyncMock()
        consumer.channel_layer.group_send = AsyncMock()
        consumer.accept = AsyncMock()
        consumer.close = AsyncMock()
        consumer.send = AsyncMock()
        await consumer.connect()
        consumer._current_thread_id = str(thread.id)
        consumer._active_thread_id = str(thread.id)

        class _Sink:
            wants_heartbeats = False

            async def send_event(self, event):
                pass

        consumer._sink = _Sink()
        return consumer

    @patch("llm.get_llm_service")
    async def test_seeded_turn_ending_on_assistant_sends_wake_note_last(self, get_service):
        from chat.consumers import _TurnState

        service = _CapturingService()
        get_service.return_value = service
        thread = await self._thread_ending_on_assistant()
        consumer = await self._consumer(thread)
        turn = _TurnState(cancel_event=threading.Event(), stream_finished=asyncio.Event())
        consumer._turn = turn

        await consumer._stream_and_finalize(
            thread, content="", requested_model=None, thinking_level=None,
            resolved_model=None, max_context_tokens=None,
            history_mode="conversational", turn=turn, seed_mode=True,
        )

        self.assertEqual(len(service.requests), 1)
        last = service.requests[0].messages[-1]
        self.assertEqual(last.role, "user")
        self.assertTrue(last.content.endswith(SUBAGENT_WAKE_NOTE))
        # The per-turn preamble lands on the wake note, not on a buried message.
        self.assertIn("# User Message", last.content)
        self.assertEqual(
            [m.role for m in service.requests[0].messages[-4:]],
            ["assistant", "tool", "assistant", "user"],
        )
        # Never persisted.
        stored = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(thread=thread, content__contains="Automatic wake-up").count()
        )()
        self.assertEqual(stored, 0)
