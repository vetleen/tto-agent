"""A chat turn whose WebSocket dies must finish detached, not vanish.

Two ways a dropped connection used to lose the reply: a send on the dead
socket raised out of ``WebSocketSink`` and aborted the stream (while the model
ran on for nobody), and a clean ``disconnect()`` cancelled the stream task
mid-reply. Now the consumer detaches — the rest of the turn is broadcast to
the thread group, the reply is persisted — and a detached session starts no
follow-up turns.

Same conventions as ``test_turn_gate_consumer``: a ``receive_json_from``
timeout cancels the ASGI application, so waiting is done on Python-side
conditions and socket events are only read once they are known to be coming.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TransactionTestCase, override_settings

from chat import turn_gate
from chat.consumers import ChatConsumer, _TurnState
from chat.models import ChatMessage
from chat.routing import websocket_urlpatterns
from chat.sinks import BroadcastSink, NullSink, WebSocketSink
# Module import (not `from ... import`) so its TestCase classes are not
# re-discovered under this module's name.
from chat.tests import test_turn_gate_consumer as tg

User = get_user_model()


class _RecordingLayer:
    def __init__(self):
        self.sent = []

    async def group_send(self, group, message):
        self.sent.append((group, message))


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class DetachedTurnTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="detach-owner@example.com", password="pass123",
        )
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)

        from guardrails.service import GuardrailVerdict

        gr = patch(
            "guardrails.service.run_classifier_pipeline",
            new_callable=AsyncMock,
            return_value=GuardrailVerdict(action="allow"),
        )
        gr.start()
        self.addCleanup(gr.stop)
        # Title generation calls the (fake) service's `arun`, which a MagicMock
        # cannot await — an unrelated ERROR log that would trip assertNoLogs.
        title = patch.object(ChatConsumer, "_generate_thread_title", new=AsyncMock())
        title.start()
        self.addCleanup(title.stop)

    async def _connect(self):
        communicator = WebsocketCommunicator(
            URLRouter(websocket_urlpatterns), "/ws/chat/",
        )
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        return communicator

    async def _start_turn(self, get_service):
        """A connected tab whose turn is parked inside the (fake) LLM stream."""
        stream = tg.BlockingStream()
        service = MagicMock()
        service.astream = stream.astream
        get_service.return_value = service

        tab = await self._connect()
        await tab.send_json_to({"type": "chat.message", "content": "Hello"})
        created = await tab.receive_json_from(timeout=5)
        self.assertEqual(created["event_type"], "thread.created")
        await tg._until(lambda: stream.started.is_set(), message="turn never reached the LLM")
        return tab, stream, created["thread_id"]

    async def _wait_for_reply(self, thread_id):
        @database_sync_to_async
        def _has_reply():
            return ChatMessage.objects.filter(
                thread_id=thread_id, role="assistant", content="Hi",
            ).exists()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if await _has_reply():
                break
            await asyncio.sleep(0.02)
        else:
            self.fail("the detached turn never persisted its reply")
        # The slot is released right after the stream; wait for it so the turn
        # is past its streaming phase before the test goes on.
        await tg._until(lambda: turn_gate.stats()["active"] == 0, message="turn kept its slot")

    @patch("llm.get_llm_service")
    async def test_failed_send_finishes_the_turn_detached(self, get_service):
        """Path A: the socket is dead when the next event is sent."""
        tab, stream, thread_id = await self._start_turn(get_service)

        layer = _RecordingLayer()
        closed = AsyncMock(side_effect=RuntimeError("Attempt to send on a closed protocol"))
        with patch("channels.layers.get_channel_layer", return_value=layer), \
                patch.object(ChatConsumer, "send", new=closed), \
                self.assertNoLogs("chat.consumers", level="ERROR"):
            stream.release.set()
            await self._wait_for_reply(thread_id)

        # The rest of the turn went to the thread group, loop-turn style.
        forwarded = [
            msg["event"]["event_type"]
            for group, msg in layer.sent
            if group == f"thread_{thread_id}" and msg["type"] == "loop.event"
        ]
        self.assertIn("message_end", forwarded)

        await asyncio.sleep(0.2)  # let the post-stream work (title, cost) settle
        await tab.disconnect()

    @patch("llm.get_llm_service")
    async def test_disconnect_mid_turn_finishes_and_reaches_another_tab(self, get_service):
        """Path B: a clean disconnect while the model is still streaming."""
        tab1, stream, thread_id = await self._start_turn(get_service)

        # A second tab (the "reconnected" page) viewing the same thread.
        tab2 = await self._connect()
        await tab2.send_json_to({"type": "chat.load_thread", "thread_id": thread_id})
        loaded = await tab2.receive_json_from(timeout=5)
        self.assertEqual(loaded["event_type"], "thread.loaded")
        follow = await tab2.receive_json_from(timeout=5)
        self.assertEqual(follow["event_type"], "tasks.loaded")

        await tab1.disconnect()  # the turn is still parked in the stream
        self.assertEqual(stream.calls, 1)

        stream.release.set()
        await self._wait_for_reply(thread_id)

        # tab2 joined the thread group on load, so it got the rest of the turn live.
        seen = []
        for _ in range(8):
            event = await tab2.receive_json_from(timeout=5)
            seen.append(event["event_type"])
            if event["event_type"] == "message_end":
                break
        self.assertIn("message_start", seen)
        self.assertIn("message_end", seen)

        await asyncio.sleep(0.2)
        await tab2.disconnect()


class DisconnectTests(SimpleTestCase):
    """``disconnect()`` on a hand-built consumer: what is torn down, what is kept."""

    def _consumer(self, thread_id="t1"):
        consumer = ChatConsumer()
        consumer._current_thread_id = thread_id
        consumer._turn = None
        consumer._stream_task = None
        consumer._guardrail_task = None
        consumer._subagent_watch_task = None
        consumer._cancel_event = None
        consumer._stopped = False
        consumer._detached = False
        consumer._sink = WebSocketSink(consumer)
        consumer._safe_group_discard = AsyncMock()
        return consumer

    def _turn(self, slot_held):
        return _TurnState(
            cancel_event=threading.Event(), stream_finished=asyncio.Event(), slot_held=slot_held,
        )

    async def test_running_turn_is_detached_not_cancelled(self):
        consumer = self._consumer()
        gate = asyncio.Event()
        consumer._turn = self._turn(slot_held=True)
        consumer._stream_task = asyncio.create_task(gate.wait())
        consumer._guardrail_task = asyncio.create_task(gate.wait())
        consumer._cancel_event = consumer._turn.cancel_event

        layer = _RecordingLayer()
        with patch("channels.layers.get_channel_layer", return_value=layer), \
                self.assertLogs("chat.consumers", level="INFO") as cm:
            await consumer.disconnect(1006)

        self.assertTrue(consumer._detached)
        self.assertIsInstance(consumer._sink, BroadcastSink)
        self.assertIn("finishing the turn detached", cm.output[0])
        self.assertFalse(consumer._cancel_event.is_set())
        self.assertFalse(consumer._stream_task.cancelled())
        self.assertFalse(consumer._guardrail_task.cancelled())
        consumer._safe_group_discard.assert_awaited_once_with("thread_t1")

        gate.set()
        await consumer._stream_task
        await consumer._guardrail_task
        # The swapped-in sink publishes to the thread group.
        await consumer._sink.send_event({"event_type": "token"})
        self.assertEqual(layer.sent[0][0], "thread_t1")

    async def test_queued_turn_is_dropped_as_before(self):
        """No slot yet → nothing generated: the departed user leaves the line."""
        consumer = self._consumer()
        consumer._turn = self._turn(slot_held=False)
        consumer._stream_task = asyncio.create_task(asyncio.Event().wait())
        consumer._cancel_event = consumer._turn.cancel_event

        await consumer.disconnect(1006)

        self.assertTrue(consumer._cancel_event.is_set())
        self.assertTrue(consumer._stream_task.cancelled())
        self.assertFalse(consumer._detached)
        self.assertIsInstance(consumer._sink, WebSocketSink)

    async def test_idle_disconnect_still_cancels_the_guardrail_scan(self):
        consumer = self._consumer()
        consumer._guardrail_task = asyncio.create_task(asyncio.sleep(30))

        await consumer.disconnect(1000)

        self.assertTrue(consumer._guardrail_task.cancelled())
        self.assertFalse(consumer._detached)
        self.assertIsInstance(consumer._sink, WebSocketSink)

    async def test_socket_lost_is_idempotent_and_needs_no_thread(self):
        consumer = self._consumer(thread_id=None)
        await consumer._socket_lost(RuntimeError("closed"))
        self.assertTrue(consumer._detached)
        self.assertIsInstance(consumer._sink, NullSink)

        first_sink = consumer._sink
        consumer._detach_from_socket("again")
        self.assertIs(consumer._sink, first_sink)


class SeedGuardTests(SimpleTestCase):
    def _consumer(self):
        consumer = ChatConsumer()
        consumer._stopped = False
        consumer._detached = False
        consumer.resolved_prefs = None
        return consumer

    def test_detached_session_never_seeds_a_continuation(self):
        consumer = self._consumer()
        turn = _TurnState(cancel_event=threading.Event(), stream_finished=asyncio.Event())
        consumer._turn = turn
        with patch.object(ChatConsumer, "_has_tool", return_value=True):
            self.assertTrue(consumer._should_seed_continuation(turn))
            consumer._detached = True
            self.assertFalse(consumer._should_seed_continuation(turn))

    def test_other_guards_unchanged(self):
        consumer = self._consumer()
        turn = _TurnState(cancel_event=threading.Event(), stream_finished=asyncio.Event())
        consumer._turn = turn
        with patch.object(ChatConsumer, "_has_tool", return_value=True):
            consumer._stopped = True
            self.assertFalse(consumer._should_seed_continuation(turn))
            consumer._stopped = False
            turn.cancel_event.set()
            self.assertFalse(consumer._should_seed_continuation(turn))
            turn.cancel_event.clear()
            consumer._turn = None  # superseded
            self.assertFalse(consumer._should_seed_continuation(turn))
