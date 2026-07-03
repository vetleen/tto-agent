"""End-to-end: a background sub-agent completing on a *fresh* loop thread must
trigger a continuation turn scoped to the current pass — not a replay of the
whole 1:1 loop thread (which grows unbounded across passes and overflowed the
model's context window in production, surfacing as "request was too large").

This drives the real ChatConsumer over a WebSocket, then simulates the Celery
worker's ``subagent.completed`` notification exactly like
``chat.tasks._notify_consumer`` does, and inspects the messages the continuation
turn sends to the (mocked) LLM.
"""

import asyncio
from datetime import timedelta
from unittest.mock import MagicMock, patch

from channels.db import database_sync_to_async
from channels.layers import get_channel_layer
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings
from django.utils import timezone

from chat.models import ChatMessage, ChatThread, Loop, SubAgentRun
from chat.routing import websocket_urlpatterns
from core.tokens import count_tokens

User = get_user_model()


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class LoopSubagentContinuationScopingTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="loopcont@example.com", password="pass123")
        self.thread = ChatThread.objects.create(created_by=self.user, title="AI news watch")
        self.loop = Loop.objects.create(
            thread=self.thread, created_by=self.user, prompt="CURRENT_PROMPT",
            history_mode=Loop.HistoryMode.FRESH, cadence_kind=Loop.Cadence.INTERVAL,
            interval_seconds=3600, next_run=timezone.now(), max_runs=100,
        )
        base = timezone.now() - timedelta(hours=1)

        def _msg(content, minutes, role="user", metadata=None, hidden=False):
            m = ChatMessage.objects.create(
                thread=self.thread, role=role, content=content,
                token_count=count_tokens(content),
                metadata=metadata or {}, is_hidden_from_user=hidden,
            )
            ChatMessage.objects.filter(pk=m.pk).update(
                created_at=base + timedelta(minutes=minutes)
            )
            return m

        # Prior pass — must be EXCLUDED from the continuation.
        _msg("old prompt", 0, metadata={"loop_run": True})
        _msg("PRIOR_PASS_REPLY", 1, role="assistant")
        # Current pass — must be INCLUDED.
        _msg("CURRENT_PROMPT", 10, metadata={"loop_run": True})
        _msg("Started the scan.", 11, role="assistant")

        # A completed background sub-agent + its hidden result message (unreported).
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="scan the web",
            status=SubAgentRun.Status.COMPLETED, result="SUBAGENT_FINDINGS",
        )
        _msg(
            f"[Sub-agent result: {str(self.run.id)[:8]}]\nSUBAGENT_FINDINGS", 12,
            metadata={"source": "subagent", "subagent_run_id": str(self.run.id)},
            hidden=True,
        )

    async def _connect(self):
        communicator = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    async def _recv_until(self, comm, event_type, max_events=10):
        for _ in range(max_events):
            evt = await comm.receive_json_from(timeout=5)
            if evt.get("event_type") == event_type:
                return evt
        raise AssertionError(f"{event_type} not received")

    @patch("llm.get_llm_service")
    async def test_continuation_scopes_to_current_pass(self, mock_get_service):
        captured = {}
        service = MagicMock()

        async def astream(mode, request, **kwargs):
            captured["messages"] = request.messages
            return
            yield  # make it an async generator

        service.astream = astream
        mock_get_service.return_value = service

        communicator = await self._connect()
        # Load the thread so the consumer joins thread_<id> and sets _current_thread_id.
        await communicator.send_json_to({
            "type": "chat.load_thread", "thread_id": str(self.thread.id),
        })
        await self._recv_until(communicator, "thread.loaded")

        # Simulate the worker notifying that the sub-agent finished.
        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {"type": "subagent.completed", "run_id": str(self.run.id),
             "thread_id": str(self.thread.id)},
        )

        # Wait until the continuation turn calls the LLM. Polling (rather than
        # draining events by timeout) avoids asgiref cancelling the app on a
        # receive timeout.
        for _ in range(50):
            if "messages" in captured:
                break
            await asyncio.sleep(0.1)
        self.assertIn("messages", captured, "continuation turn never called the LLM")

        blob = " ".join(str(m.content) for m in captured["messages"])
        # Scoped to the current pass:
        self.assertIn("CURRENT_PROMPT", blob)
        self.assertIn("SUBAGENT_FINDINGS", blob)
        # Prior pass excluded:
        self.assertNotIn("PRIOR_PASS_REPLY", blob)
        self.assertNotIn("old prompt", blob)

        await communicator.disconnect()

    @patch("llm.get_llm_service")
    async def test_conversational_loop_not_scoped(self, mock_get_service):
        """A conversational loop keeps full history on the continuation."""
        await database_sync_to_async(
            Loop.objects.filter(pk=self.loop.pk).update
        )(history_mode=Loop.HistoryMode.CONVERSATIONAL)

        captured = {}
        service = MagicMock()

        async def astream(mode, request, **kwargs):
            captured["messages"] = request.messages
            return
            yield

        service.astream = astream
        mock_get_service.return_value = service

        communicator = await self._connect()
        await communicator.send_json_to({
            "type": "chat.load_thread", "thread_id": str(self.thread.id),
        })
        await self._recv_until(communicator, "thread.loaded")

        channel_layer = get_channel_layer()
        await channel_layer.group_send(
            f"thread_{self.thread.id}",
            {"type": "subagent.completed", "run_id": str(self.run.id),
             "thread_id": str(self.thread.id)},
        )

        for _ in range(50):
            if "messages" in captured:
                break
            await asyncio.sleep(0.1)
        self.assertIn("messages", captured)

        blob = " ".join(str(m.content) for m in captured["messages"])
        self.assertIn("PRIOR_PASS_REPLY", blob)  # full history retained
        self.assertIn("SUBAGENT_FINDINGS", blob)

        await communicator.disconnect()
