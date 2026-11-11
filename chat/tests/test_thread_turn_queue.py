"""One turn at a time per thread: a second turn on the same thread waits.

A page reloaded mid-turn (the turn finishes detached), a second tab or a second
device could start a turn on a thread whose previous turn was still running; it
ran concurrently, without that reply in its history. Now every turn holds a
per-thread lease (chat/thread_turn_lease.py) until its reply is saved and a turn
that finds it taken waits (``turn.queued`` reason ``thread_busy``), then runs on
a history that has the reply.

Conventions as in ``test_turn_gate_consumer``: never wait on a socket timeout.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TransactionTestCase, override_settings

from chat import thread_turn_lease, turn_gate
from chat.consumers import ChatConsumer
from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from chat.tests import test_turn_gate_consumer as tg

User = get_user_model()

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                      "LOCATION": "thread-turn-queue-tests"}}


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
    CACHES=LOCMEM,
)
class ThreadTurnQueueTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="queue-owner@example.com", password="pass123")
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)
        cache.clear()
        self.addCleanup(cache.clear)
        poll = patch.object(thread_turn_lease, "POLL_INTERVAL_S", 0.01)
        poll.start()
        self.addCleanup(poll.stop)

        from guardrails.service import GuardrailVerdict

        gr = patch(
            "guardrails.service.run_classifier_pipeline",
            new_callable=AsyncMock,
            return_value=GuardrailVerdict(action="allow"),
        )
        gr.start()
        self.addCleanup(gr.stop)
        title = patch.object(ChatConsumer, "_generate_thread_title", new=AsyncMock())
        title.start()
        self.addCleanup(title.stop)

    # -- helpers ------------------------------------------------------------

    async def _connect(self):
        communicator = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        return communicator

    def _service(self, dispatcher):
        service = MagicMock()
        service.astream = dispatcher.astream
        return service

    async def _read_until(self, tab, event_type, limit=40):
        seen = []
        for _ in range(limit):
            event = await tab.receive_json_from(timeout=5)
            seen.append(event.get("event_type"))
            if event.get("event_type") == event_type:
                return event
        self.fail(f"{event_type} never arrived; saw {seen}")

    async def _start_first_turn(self, dispatcher):
        tab1 = await self._connect()
        await tab1.send_json_to({"type": "chat.message", "content": "First question"})
        created = await self._read_until(tab1, "thread.created")
        await tg._until(lambda: dispatcher.streams[0].started.is_set(),
                        message="first turn never reached the LLM")
        return tab1, created["thread_id"]

    @database_sync_to_async
    def _visible(self, thread_id):
        return [
            (m.role, m.content)
            for m in ChatMessage.objects.filter(thread_id=thread_id, is_hidden_from_user=False)
            .order_by("created_at")
        ]

    async def _wait_visible(self, thread_id, predicate, message):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rows = await self._visible(thread_id)
            if predicate(rows):
                return rows
            await asyncio.sleep(0.02)
        self.fail(message)

    async def _settle(self, *tabs):
        await tg._until(lambda: turn_gate.stats()["active"] == 0, message="a turn kept its slot")
        await asyncio.sleep(0.2)
        for tab in tabs:
            await tab.disconnect()

    def _second_reply(self, dispatcher):
        dispatcher.streams[1].events = tg._events("SECOND")

    # -- tests --------------------------------------------------------------

    @patch("llm.get_llm_service")
    async def test_second_tab_waits_then_runs_with_the_first_reply(self, get_service):
        dispatcher = tg.StreamDispatcher(2)
        self._second_reply(dispatcher)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.message", "content": "Second question",
                                 "thread_id": thread_id})
        queued = await self._read_until(tab2, "turn.queued")
        self.assertEqual(queued["data"]["reason"], "thread_busy")
        await asyncio.sleep(0.1)
        self.assertEqual(dispatcher.streams[1].calls, 0)  # still waiting

        dispatcher.streams[0].release.set()
        await tg._until(lambda: dispatcher.streams[1].started.is_set(),
                        message="queued turn never started after the first finished")
        dispatcher.streams[1].release.set()

        rows = await self._wait_visible(
            thread_id, lambda rows: ("assistant", "SECOND") in rows, "second reply missing",
        )
        await self._settle(tab1, tab2)
        # The queued message was moved after the reply it waited for.
        self.assertEqual(rows, [
            ("user", "First question"), ("assistant", "Hi"),
            ("user", "Second question"), ("assistant", "SECOND"),
        ])
        request = dispatcher.streams[1].requests[0]
        roles = [m.role for m in request.messages if m.role != "system"]
        self.assertEqual(roles[-1], "user")
        self.assertIn("Hi", " ".join(str(m.content) for m in request.messages if m.role == "assistant"))
        self.assertIsNone(thread_turn_lease.current(thread_id))

    @patch("llm.get_llm_service")
    async def test_a_detached_turn_still_holds_the_thread(self, get_service):
        dispatcher = tg.StreamDispatcher(2)
        self._second_reply(dispatcher)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        await tab1.disconnect()  # reload mid-turn: the turn finishes detached

        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.message", "content": "Again?",
                                 "thread_id": thread_id})
        queued = await self._read_until(tab2, "turn.queued")
        self.assertEqual(queued["data"]["reason"], "thread_busy")
        self.assertEqual(dispatcher.streams[1].calls, 0)

        dispatcher.release_all()
        rows = await self._wait_visible(
            thread_id, lambda rows: ("assistant", "SECOND") in rows, "second reply missing",
        )
        await self._settle(tab2)
        self.assertEqual([r[1] for r in rows], ["First question", "Hi", "Again?", "SECOND"])

    @patch("llm.get_llm_service")
    async def test_queued_turn_survives_its_tab_closing(self, get_service):
        """Tab 2 queues behind tab 1, then closes (reload, tab closed): its message
        is still answered, detached, after tab 1's reply — not left unanswered."""
        dispatcher = tg.StreamDispatcher(2)
        self._second_reply(dispatcher)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.message", "content": "Follow-up",
                                 "thread_id": thread_id})
        queued = await self._read_until(tab2, "turn.queued")
        self.assertEqual(queued["data"]["reason"], "thread_busy")
        await tab2.disconnect()

        dispatcher.streams[0].release.set()
        await tg._until(lambda: dispatcher.streams[1].started.is_set(),
                        message="the queued turn was dropped when its tab closed")
        dispatcher.streams[1].release.set()
        rows = await self._wait_visible(
            thread_id, lambda rows: ("assistant", "SECOND") in rows, "queued reply missing",
        )
        await self._settle(tab1)
        self.assertEqual([r[1] for r in rows], ["First question", "Hi", "Follow-up", "SECOND"])
        self.assertIsNone(thread_turn_lease.current(thread_id))

    @override_settings(CHAT_THREAD_TURN_WAIT_SECONDS=0.2)
    @patch("llm.get_llm_service")
    async def test_timeout_reports_thread_busy_and_keeps_the_message(self, get_service):
        dispatcher = tg.StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.message", "content": "Waiting in vain",
                                 "thread_id": thread_id})
        error = await self._read_until(tab2, "error")
        self.assertEqual(error["data"]["error_code"], "thread_busy_timeout")
        self.assertEqual(dispatcher.streams[1].calls, 0)
        self.assertIn(("user", "Waiting in vain"), await self._visible(thread_id))

        dispatcher.streams[0].release.set()
        await self._wait_visible(
            thread_id, lambda rows: ("assistant", "Hi") in rows, "first reply missing",
        )
        await self._settle(tab1, tab2)

    @patch("llm.get_llm_service")
    async def test_stop_while_queued_leaves_the_running_turn_alone(self, get_service):
        dispatcher = tg.StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.message", "content": "Never mind",
                                 "thread_id": thread_id})
        await self._read_until(tab2, "turn.queued")
        await tab2.send_json_to({"type": "chat.stop", "thread_id": thread_id})
        await self._read_until(tab2, "stream.cancelled")

        holder = thread_turn_lease.current(thread_id)
        self.assertIsNotNone(holder)  # still tab 1's turn
        dispatcher.streams[0].release.set()
        await self._wait_visible(
            thread_id, lambda rows: ("assistant", "Hi") in rows, "first reply missing",
        )
        await self._settle(tab1, tab2)
        self.assertEqual(dispatcher.streams[1].calls, 0)
        self.assertIsNone(thread_turn_lease.current(thread_id))

    @patch("llm.get_llm_service")
    async def test_superseded_turn_releases_the_thread(self, get_service):
        """Same tab, mid-stream message: the old turn is cancelled and frees the lease."""
        dispatcher = tg.StreamDispatcher(2)
        self._second_reply(dispatcher)
        get_service.return_value = self._service(dispatcher)

        tab1, thread_id = await self._start_first_turn(dispatcher)
        await tab1.send_json_to({"type": "chat.message", "content": "Instead",
                                 "thread_id": thread_id})
        await tg._until(lambda: dispatcher.streams[1].started.is_set(),
                        message="the superseding turn waited on the cancelled turn's lease")
        dispatcher.streams[1].release.set()
        await self._wait_visible(
            thread_id, lambda rows: ("assistant", "SECOND") in rows, "second reply missing",
        )
        await self._settle(tab1)
        self.assertIsNone(thread_turn_lease.current(thread_id))


@override_settings(CACHES=LOCMEM)
class LoopTurnLeaseTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="loop-lease@example.com", password="pass123")
        self.thread = ChatThread.objects.create(created_by=self.user, title="loop")
        cache.clear()
        self.addCleanup(cache.clear)
        poll = patch.object(thread_turn_lease, "POLL_INTERVAL_S", 0.01)
        poll.start()
        self.addCleanup(poll.stop)

    def _runner(self):
        from chat.loop_service import HeadlessTurnRunner
        from chat.sinks import NullSink

        runner = HeadlessTurnRunner(user=self.user, thread_id=self.thread.id, sink=NullSink())
        runner._load_membership = AsyncMock()
        runner._check_suspension = AsyncMock(return_value=False)
        runner._check_budget_exceeded = AsyncMock(return_value=None)
        runner._resolve_preferences = AsyncMock(return_value=None)
        runner.run_turn_to_completion = AsyncMock()
        return runner

    async def _run(self, runner):
        return await runner.run_loop_turn(
            self.thread, "loop prompt", history_mode="full", data_room_ids=[], active_skill_ids=[],
        )

    async def test_loop_turn_waits_for_a_running_turn(self):
        holder = thread_turn_lease.try_acquire(self.thread.id, kind="chat")
        runner = self._runner()
        task = asyncio.create_task(self._run(runner))
        await asyncio.sleep(0.1)
        self.assertFalse(task.done())
        runner.run_turn_to_completion.assert_not_awaited()

        await thread_turn_lease.aio_release(holder)
        self.assertTrue(await asyncio.wait_for(task, timeout=5))
        runner.run_turn_to_completion.assert_awaited_once()
        self.assertIsNone(thread_turn_lease.current(self.thread.id))

    @override_settings(CHAT_THREAD_TURN_WAIT_SECONDS=0.1)
    async def test_loop_turn_skips_when_the_thread_stays_busy(self):
        thread_turn_lease.try_acquire(self.thread.id, kind="chat")
        runner = self._runner()
        self.assertFalse(await self._run(runner))
        runner.run_turn_to_completion.assert_not_awaited()

        @database_sync_to_async
        def prompt_rows():
            return ChatMessage.objects.filter(thread=self.thread, content="loop prompt").count()

        self.assertEqual(await prompt_rows(), 0)
