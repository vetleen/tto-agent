"""A page that opens a thread while a turn runs elsewhere shows it, then gets the reply.

``thread.loaded`` carries ``turn_in_flight`` (the lease of a turn another tab,
device or process is running on the thread); the page shows a locked "still
working" bubble, and when that lease is released the consumer sends
``turn.finished`` with the saved reply (``_thread_turn_watch_loop``).

Conventions as in ``test_turn_gate_consumer``: never wait on a socket timeout.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from chat import thread_turn_lease, turn_gate
from chat.consumers import ChatConsumer, _TurnState
from chat.routing import websocket_urlpatterns
from chat.tests import test_turn_gate_consumer as tg

User = get_user_model()

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                      "LOCATION": "thread-turn-live-view-tests"}}


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
    CACHES=LOCMEM,
)
class ThreadTurnLiveViewTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="live-view@example.com", password="pass123")
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)
        cache.clear()
        self.addCleanup(cache.clear)
        for target, attr, value in (
            (thread_turn_lease, "POLL_INTERVAL_S", 0.01),
            (ChatConsumer, "THREAD_TURN_WATCH_INTERVAL_SECONDS", 0.02),
        ):
            p = patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)

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

    async def _connect(self):
        communicator = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        return communicator

    async def _read_until(self, tab, event_type, limit=40):
        seen = []
        for _ in range(limit):
            event = await tab.receive_json_from(timeout=5)
            seen.append(event.get("event_type"))
            if event.get("event_type") == event_type:
                return event, seen
        self.fail(f"{event_type} never arrived; saw {seen}")

    async def _start_turn(self, get_service):
        stream = tg.BlockingStream()
        service = MagicMock()
        service.astream = stream.astream
        get_service.return_value = service
        tab1 = await self._connect()
        await tab1.send_json_to({"type": "chat.message", "content": "Hello"})
        created, _ = await self._read_until(tab1, "thread.created")
        await tg._until(lambda: stream.started.is_set(), message="turn never reached the LLM")
        return tab1, stream, created["thread_id"]

    async def _load(self, tab, thread_id):
        await tab.send_json_to({"type": "chat.load_thread", "thread_id": thread_id})
        loaded, _ = await self._read_until(tab, "thread.loaded")
        return loaded

    async def _settle(self, *tabs):
        await tg._until(lambda: turn_gate.stats()["active"] == 0, message="a turn kept its slot")
        await asyncio.sleep(0.2)
        for tab in tabs:
            await tab.disconnect()

    @patch("llm.get_llm_service")
    async def test_second_tab_sees_the_turn_then_gets_the_reply(self, get_service):
        tab1, stream, thread_id = await self._start_turn(get_service)
        tab2 = await self._connect()
        loaded = await self._load(tab2, thread_id)
        in_flight = loaded["turn_in_flight"]
        self.assertEqual(in_flight["token"], thread_turn_lease.current(thread_id)["token"])
        self.assertEqual(in_flight["kind"], "chat")

        stream.release.set()
        finished, seen = await self._read_until(tab2, "turn.finished")
        self.assertEqual(finished["data"]["token"], in_flight["token"])
        self.assertEqual(finished["data"]["reply"]["content"], "Hi")
        # A turn in another tab is not broadcast token by token.
        self.assertNotIn("token", seen)
        await self._settle(tab1, tab2)

    @patch("llm.get_llm_service")
    async def test_reloaded_page_follows_a_detached_turn(self, get_service):
        tab1, stream, thread_id = await self._start_turn(get_service)
        await tab1.disconnect()  # the turn finishes detached

        tab2 = await self._connect()
        loaded = await self._load(tab2, thread_id)
        self.assertIsNotNone(loaded["turn_in_flight"])

        stream.release.set()
        finished, seen = await self._read_until(tab2, "turn.finished")
        self.assertIn("message_end", seen)  # streamed live via loop.event first
        self.assertEqual(finished["data"]["reply"]["content"], "Hi")
        await self._settle(tab2)

    @patch("llm.get_llm_service")
    async def test_own_turn_is_not_reported_in_flight(self, get_service):
        tab1, stream, thread_id = await self._start_turn(get_service)
        loaded = await self._load(tab1, thread_id)
        self.assertIsNone(loaded["turn_in_flight"])
        stream.release.set()
        await self._settle(tab1)

    @patch("llm.get_llm_service")
    async def test_vanished_lease_finishes_with_no_reply(self, get_service):
        tab1, stream, thread_id = await self._start_turn(get_service)
        tab2 = await self._connect()
        loaded = await self._load(tab2, thread_id)
        self.assertIsNotNone(loaded["turn_in_flight"])

        cache.delete(thread_turn_lease._key(thread_id))  # crashed holder / expired
        finished, _ = await self._read_until(tab2, "turn.finished")
        self.assertIsNone(finished["data"]["reply"])

        stream.release.set()
        await self._settle(tab1, tab2)

    async def test_idle_thread_has_no_turn_in_flight(self):
        from channels.db import database_sync_to_async

        from chat.models import ChatThread

        thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user, title="idle",
        )
        tab = await self._connect()
        loaded = await self._load(tab, str(thread.id))
        self.assertIsNone(loaded["turn_in_flight"])
        await tab.disconnect()


@override_settings(CACHES=LOCMEM)
class ThreadTurnWatchUnitTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)

    async def test_foreign_vs_own_lease(self):
        consumer = ChatConsumer()
        consumer._turn = None
        lease = thread_turn_lease.try_acquire("t1", kind="chat")
        foreign = await consumer._foreign_turn_in_flight("t1")
        self.assertEqual(foreign["token"], lease.token)

        consumer._turn = _TurnState(cancel_event=threading.Event(), stream_finished=asyncio.Event())
        consumer._turn.thread_lease = lease
        self.assertIsNone(await consumer._foreign_turn_in_flight("t1"))
        self.assertIsNone(await consumer._foreign_turn_in_flight("t2"))

    async def test_cancel_stops_the_watch(self):
        consumer = ChatConsumer()
        consumer._thread_turn_watch_task = asyncio.create_task(asyncio.sleep(30))
        task = consumer._thread_turn_watch_task
        await consumer._cancel_thread_turn_watch()
        self.assertTrue(task.cancelled())
        self.assertIsNone(consumer._thread_turn_watch_task)
        await consumer._cancel_thread_turn_watch()  # idempotent, no task
