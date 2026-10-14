"""End-to-end: a sub-agent outcome that lands while a seeded continuation is
already streaming must still reach the model.

Run A finishing seeds an orchestrator turn. Run B finishes mid-stream: its
hidden outcome message is written and its completion event is ignored (a
stream is running). The turn's final reply is persisted AFTER B's message, and
the old "reported" test — any later reply tagged as a sub-agent response —
counted that reply as B's report. Nothing seeded again, the watchdog saw no
active run and retired, and B was orphaned until the user typed something.

Reporting is now per run (``SubAgentRun.delivered_at``, stamped for exactly the
outcomes snapshotted when the turn's history was loaded), so the post-stream
claim finds B and seeds its own continuation.

Drives the real ChatConsumer over a WebSocket with a mocked LLM, like
``test_loop_continuation``.
"""

import asyncio
from unittest.mock import MagicMock, patch

from channels.db import database_sync_to_async
from channels.layers import get_channel_layer
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat.models import ChatMessage, ChatThread, SubAgentRun
from chat.routing import websocket_urlpatterns
from core.tokens import count_tokens
from llm.types.streaming import StreamEvent

User = get_user_model()


def _reply_events(text, run_id):
    return [
        StreamEvent(event_type="message_start", data={}, sequence=0, run_id=run_id),
        StreamEvent(event_type="token", data={"text": text}, sequence=1, run_id=run_id),
        StreamEvent(event_type="message_end", data={}, sequence=2, run_id=run_id),
    ]


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class SubagentMidTurnContinuationTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="midturn@example.com", password="pass123")
        self.thread = ChatThread.objects.create(created_by=self.user, title="Two sub-agents")
        for role, content in (
            ("user", "Research X and Y."),
            ("assistant", "Dispatched two sub-agents."),
        ):
            ChatMessage.objects.create(
                thread=self.thread, role=role, content=content,
                token_count=count_tokens(content),
            )
        # The post-stream seed is gated on the sub-agent tool being allowed.
        _ht = patch("chat.consumers.ChatConsumer._has_tool", return_value=True)
        _ht.start()
        self.addCleanup(_ht.stop)

    @database_sync_to_async
    def _finish_run(self, findings):
        """What the worker does when a run completes: COMPLETED + hidden message."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt=findings,
            status=SubAgentRun.Status.COMPLETED, result=findings,
        )
        content = f"[Sub-agent result: {str(run.id)[:8]}]\n{findings}"
        ChatMessage.objects.create(
            thread=self.thread, role="user", content=content,
            token_count=count_tokens(content),
            metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            is_hidden_from_user=True,
        )
        return run

    async def _notify_completed(self, run):
        await get_channel_layer().group_send(
            f"thread_{self.thread.id}",
            {"type": "subagent.completed", "run_id": str(run.id),
             "thread_id": str(self.thread.id)},
        )

    @database_sync_to_async
    def _fresh(self, run):
        return SubAgentRun.objects.get(pk=run.pk)

    @database_sync_to_async
    def _final_replies(self):
        return list(
            ChatMessage.objects.filter(
                thread=self.thread, role="assistant", is_hidden_from_user=False,
            ).order_by("created_at").values_list("content", "metadata")
        )

    async def _connect(self):
        communicator = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    async def _recv_until(self, comm, event_type, max_events=30):
        for _ in range(max_events):
            evt = await comm.receive_json_from(timeout=5)
            if evt.get("event_type") == event_type:
                return evt
        raise AssertionError(f"{event_type} not received")

    async def _load(self, communicator):
        await communicator.send_json_to({
            "type": "chat.load_thread", "thread_id": str(self.thread.id),
        })
        await self._recv_until(communicator, "thread.loaded")

    async def _wait_for(self, predicate, seconds=10.0):
        """Poll (never drain events by timeout: a receive timeout cancels the app)."""
        for _ in range(int(seconds / 0.1)):
            if predicate():
                return True
            await asyncio.sleep(0.1)
        return predicate()

    @patch("llm.get_llm_service")
    async def test_outcome_landing_mid_turn_seeds_its_own_continuation(self, mock_get_service):
        calls = []
        state = {}
        service = MagicMock()

        async def astream(mode, request, **kwargs):
            calls.append(" ".join(str(m.content) for m in request.messages))
            if len(calls) == 1:
                # Run B finishes while A's continuation is streaming: the worker
                # writes its outcome and notifies; the consumer must ignore the
                # event for now (a stream is running) and pick B up afterwards.
                state["run_b"] = await self._finish_run("B_FINDINGS")
                await self._notify_completed(state["run_b"])
                await asyncio.sleep(0.3)
                for event in _reply_events("Reporting A.", "r1"):
                    yield event
            else:
                for event in _reply_events("Reporting B.", "r2"):
                    yield event

        service.astream = astream
        mock_get_service.return_value = service

        communicator = await self._connect()
        await self._load(communicator)

        run_a = await self._finish_run("A_FINDINGS")
        await self._notify_completed(run_a)

        self.assertTrue(
            await self._wait_for(lambda: len(calls) >= 2),
            f"second continuation never ran ({len(calls)} LLM call(s))",
        )
        self.assertIn("A_FINDINGS", calls[0])
        self.assertNotIn("B_FINDINGS", calls[0])
        self.assertIn("B_FINDINGS", calls[1])

        delivered = {}

        async def _both_delivered():
            delivered["a"] = (await self._fresh(run_a)).delivered_at
            delivered["b"] = (await self._fresh(state["run_b"])).delivered_at
            return delivered["a"] is not None and delivered["b"] is not None

        for _ in range(100):
            if await _both_delivered():
                break
            await asyncio.sleep(0.1)
        self.assertIsNotNone(delivered["a"])
        self.assertIsNotNone(delivered["b"])

        # Exactly one continuation per outcome, no third turn.
        await asyncio.sleep(0.5)
        self.assertEqual(len(calls), 2)
        replies = await self._final_replies()
        self.assertEqual(
            [(content, (meta or {}).get("subagent_delivered_run_ids")) for content, meta in replies[-2:]],
            [("Reporting A.", [str(run_a.id)]), ("Reporting B.", [str(state["run_b"].id)])],
        )

        await communicator.disconnect()

    @patch("llm.get_llm_service")
    async def test_errored_continuation_leaves_outcome_undelivered(self, mock_get_service):
        """No final reply persisted → nothing marked delivered; the claim lease
        expiring lets the watchdog / next thread load re-seed, as before."""
        service = MagicMock()

        async def astream(mode, request, **kwargs):
            yield StreamEvent(event_type="error", data={"message": "boom"}, sequence=0, run_id="r1")

        service.astream = astream
        mock_get_service.return_value = service

        communicator = await self._connect()
        await self._load(communicator)

        run_a = await self._finish_run("A_FINDINGS")
        await self._notify_completed(run_a)
        await self._recv_until(communicator, "error")

        claimed = {}

        async def _claimed():
            claimed["run"] = await self._fresh(run_a)
            return claimed["run"].reported_at is not None

        for _ in range(50):
            if await _claimed():
                break
            await asyncio.sleep(0.1)
        self.assertIsNotNone(claimed["run"].reported_at, "the seeded turn claimed the run")
        self.assertIsNone(claimed["run"].delivered_at)

        await communicator.disconnect()
