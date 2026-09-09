"""Consumer-level tests for the chat-turn concurrency gate.

These prove the behaviour a user actually sees: an over-cap turn waits instead of
being refused, it can still be stopped, and it never leaves a slot behind.

Two conventions matter here and are easy to get wrong:

* A ``receive_json_from`` timeout **cancels the ASGI application** (asgiref), so
  waiting is done by polling a Python-side condition — ``turn_gate.stats()`` or a
  ``BlockingStream`` flag — and events are only read when the sequence is known.
  See the same note at ``chat/tests/test_consumer.py`` and
  ``chat/tests/test_loop_continuation.py``.
* The caps are module-level ``os.environ`` reads, so they are set with
  ``@patch("chat.turn_gate.<CONST>", n)`` rather than ``override_settings``.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat import turn_gate
from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from llm.types.streaming import StreamEvent

User = get_user_model()


def _events(text="Hi"):
    return [
        StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
        StreamEvent(event_type="token", data={"text": text}, sequence=1, run_id="r1"),
        StreamEvent(event_type="message_end", data={}, sequence=2, run_id="r1"),
    ]


class BlockingStream:
    """A controllable stand-in for ``LLMService.astream``.

    ``started`` fires when a turn actually reaches the LLM, which is to say when
    it won a slot; the stream then parks until the test sets ``release``. That
    lets a test pin turns inside the gate and assert "this turn has NOT started"
    as a Python-side condition, instead of draining socket events on a timeout.

    The repo had no blocking stream mock before this; every other fake ``astream``
    yields its events immediately and so cannot hold a turn open.
    """

    def __init__(self, events=None):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.requests = []
        self.events = _events() if events is None else events

    async def astream(self, pipeline_id, request, **kwargs):
        self.calls += 1
        self.requests.append(request)
        self.started.set()
        await self.release.wait()
        for event in self.events:
            yield event


class StreamDispatcher:
    """Hands each successive turn its own :class:`BlockingStream`."""

    def __init__(self, count):
        self.streams = [BlockingStream() for _ in range(count)]
        self._next = 0

    @property
    def calls(self):
        return sum(s.calls for s in self.streams)

    def astream(self, pipeline_id, request, **kwargs):
        stream = self.streams[min(self._next, len(self.streams) - 1)]
        self._next += 1
        return stream.astream(pipeline_id, request, **kwargs)

    def release_all(self):
        for stream in self.streams:
            stream.release.set()


async def _until(predicate, timeout=5.0, message="condition not met"):
    """Poll a Python-side condition. Never drains socket events by timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("%s (waited %.1fs)" % (message, timeout))


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class TurnGateConsumerTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="gate-owner@example.com", password="pass123",
        )
        self.other = User.objects.create_user(
            email="gate-other@example.com", password="pass123",
        )
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)

        # Isolate these assertions from the guardrail pipeline's own LLM calls,
        # exactly as ConsumerMessageTests does.
        from guardrails.service import GuardrailVerdict

        gr = patch(
            "guardrails.service.run_classifier_pipeline",
            new_callable=AsyncMock,
            return_value=GuardrailVerdict(action="allow"),
        )
        gr.start()
        self.addCleanup(gr.stop)

    async def _connect(self, user=None):
        communicator = WebsocketCommunicator(
            URLRouter(websocket_urlpatterns), "/ws/chat/",
        )
        communicator.scope["user"] = user or self.user
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        return communicator

    def _service(self, dispatcher):
        service = MagicMock()
        service.astream = dispatcher.astream
        return service

    async def _send(self, communicator, content="Hello"):
        await communicator.send_json_to({"type": "chat.message", "content": content})
        created = await communicator.receive_json_from(timeout=5)
        self.assertEqual(created["event_type"], "thread.created")
        return created["thread_id"]

    # -- queueing ---------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_second_turn_for_same_user_is_queued_then_streams(self, get_service):
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(
            lambda: dispatcher.streams[0].started.is_set(),
            message="first turn never reached the LLM",
        )

        await self._send(tab2)
        queued = await tab2.receive_json_from(timeout=5)
        self.assertEqual(queued["event_type"], "turn.queued")
        self.assertEqual(queued["data"]["reason"], "user")
        self.assertEqual(queued["data"]["position"], 1)
        # Queued means queued: it has not called the LLM at all.
        self.assertEqual(dispatcher.streams[1].calls, 0)

        # Releasing the first turn lets the second start on its own.
        dispatcher.streams[0].release.set()
        await _until(
            lambda: dispatcher.streams[1].started.is_set(),
            message="queued turn never started after the slot freed",
        )
        dispatcher.streams[1].release.set()

        start = await tab2.receive_json_from(timeout=5)
        self.assertEqual(start["event_type"], "message_start")

        await tab1.disconnect()
        await tab2.disconnect()

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    @patch("llm.get_llm_service")
    async def test_system_cap_queues_across_users(self, get_service):
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect(self.user)
        tab2 = await self._connect(self.other)

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())

        await self._send(tab2)
        queued = await tab2.receive_json_from(timeout=5)
        self.assertEqual(queued["event_type"], "turn.queued")
        self.assertEqual(queued["data"]["reason"], "system")

        dispatcher.release_all()
        await _until(lambda: dispatcher.streams[1].started.is_set())

        await tab1.disconnect()
        await tab2.disconnect()

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 0)
    @patch("llm.get_llm_service")
    async def test_zero_caps_disable_the_gate(self, get_service):
        dispatcher = StreamDispatcher(3)
        get_service.return_value = self._service(dispatcher)

        tabs = [await self._connect() for _ in range(3)]
        for tab in tabs:
            await self._send(tab)

        await _until(
            lambda: all(s.started.is_set() for s in dispatcher.streams),
            message="a turn was queued even though both caps are disabled",
        )
        self.assertEqual(turn_gate.stats()["waiting"], 0)

        dispatcher.release_all()
        for tab in tabs:
            await tab.disconnect()

    # -- getting out of the queue -----------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_stop_cancels_a_queued_turn(self, get_service):
        """The headline case.

        A queued turn must remain stoppable, which is only true because the wait
        happens inside the background task rather than in the dispatch loop. If
        it ever moves inline, this test fails by timing out.
        """
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())

        thread_id = await self._send(tab2)
        queued = await tab2.receive_json_from(timeout=5)
        self.assertEqual(queued["event_type"], "turn.queued")
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        await tab2.send_json_to({"type": "chat.stop", "thread_id": thread_id})
        cancelled = await tab2.receive_json_from(timeout=5)
        self.assertEqual(cancelled["event_type"], "subagents.updated")
        cancelled = await tab2.receive_json_from(timeout=5)
        self.assertEqual(cancelled["event_type"], "stream.cancelled")

        await _until(
            lambda: turn_gate.stats()["waiting"] == 0,
            message="stopping a queued turn left it in the line",
        )
        self.assertEqual(dispatcher.streams[1].calls, 0)

        dispatcher.release_all()
        await tab1.disconnect()
        await tab2.disconnect()

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_disconnect_releases_a_queued_slot(self, get_service):
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())
        await self._send(tab2)
        await tab2.receive_json_from(timeout=5)  # turn.queued
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        await tab2.disconnect()
        await _until(
            lambda: turn_gate.stats()["waiting"] == 0,
            message="closing the tab left its turn in the line",
        )

        dispatcher.release_all()
        await tab1.disconnect()

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_new_message_supersedes_a_queued_turn(self, get_service):
        dispatcher = StreamDispatcher(3)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())

        thread_id = await self._send(tab2, content="first try")
        await tab2.receive_json_from(timeout=5)  # turn.queued
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        # A second message on the same socket replaces the queued turn rather
        # than stacking a second one behind it.
        await tab2.send_json_to({
            "type": "chat.message", "content": "second try", "thread_id": thread_id,
        })
        # The cancelled turn's cleanup resyncs the sub-agent bar on its way out
        # (pre-existing behaviour of the `finally`), then the new turn queues.
        resync = await tab2.receive_json_from(timeout=5)
        self.assertEqual(resync["event_type"], "subagents.updated")
        queued = await tab2.receive_json_from(timeout=5)
        self.assertEqual(queued["event_type"], "turn.queued")
        self.assertEqual(turn_gate.stats()["waiting"], 1)

        dispatcher.streams[0].release.set()
        await _until(lambda: dispatcher.streams[1].started.is_set())

        # Only the newer message reached the LLM.
        sent = dispatcher.streams[1].requests[0]
        self.assertIn("second try", sent.messages[-1].content)

        dispatcher.release_all()
        await tab1.disconnect()
        await tab2.disconnect()

    # -- freshness --------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_queued_turn_assembles_history_after_admission(self, get_service):
        """A turn that waited must not stream the history it saw on arrival."""
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())

        thread_id = await self._send(tab2, content="question")
        await tab2.receive_json_from(timeout=5)  # turn.queued
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        # Something lands in the thread while the turn sits in the queue.
        @database_sync_to_async
        def _add_message():
            thread = ChatThread.objects.get(id=thread_id)
            ChatMessage.objects.create(
                thread=thread, role="assistant", content="ARRIVED-WHILE-QUEUED",
            )

        await _add_message()

        dispatcher.streams[0].release.set()
        await _until(lambda: dispatcher.streams[1].started.is_set())

        request = dispatcher.streams[1].requests[0]
        transcript = " ".join(str(m.content or "") for m in request.messages)
        self.assertIn("ARRIVED-WHILE-QUEUED", transcript)

        dispatcher.release_all()
        await tab1.disconnect()
        await tab2.disconnect()

    # -- timeout ----------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("chat.turn_gate.CHAT_TURN_QUEUE_TIMEOUT_SECONDS", 0.2)
    @patch("llm.get_llm_service")
    async def test_queue_timeout_reports_busy_and_keeps_the_user_message(self, get_service):
        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        tab1 = await self._connect()
        tab2 = await self._connect()

        await self._send(tab1)
        await _until(lambda: dispatcher.streams[0].started.is_set())

        await self._send(tab2, content="please answer")
        queued = await tab2.receive_json_from(timeout=5)
        self.assertEqual(queued["event_type"], "turn.queued")

        error = await tab2.receive_json_from(timeout=5)
        self.assertEqual(error["event_type"], "error")
        self.assertEqual(error["data"]["error_code"], "turn_queue_timeout")

        # The user's message survives, so resending is one click.
        @database_sync_to_async
        def _user_messages():
            return list(
                ChatMessage.objects.filter(role="user").values_list("content", flat=True)
            )

        self.assertIn("please answer", await _user_messages())
        self.assertEqual(turn_gate.stats()["waiting"], 0)

        dispatcher.release_all()
        await tab1.disconnect()
        await tab2.disconnect()

    # -- heartbeats -------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_heartbeats_run_while_a_turn_is_queued(self, get_service):
        """Without these the browser's 11-minute stream timeout would fire."""
        from chat.consumers import ChatConsumer

        dispatcher = StreamDispatcher(2)
        get_service.return_value = self._service(dispatcher)

        original = ChatConsumer._send_heartbeats

        async def fast_heartbeats(self, interval=0.05):
            await original(self, interval)

        with patch.object(ChatConsumer, "_send_heartbeats", fast_heartbeats):
            tab1 = await self._connect()
            tab2 = await self._connect()

            await self._send(tab1)
            await _until(lambda: dispatcher.streams[0].started.is_set())
            await self._send(tab2)

            queued = await tab2.receive_json_from(timeout=5)
            self.assertEqual(queued["event_type"], "turn.queued")

            # Only heartbeats can arrive here: the first turn still holds the
            # slot, so nothing else is coming down this socket.
            beat = await tab2.receive_json_from(timeout=5)
            self.assertEqual(beat["event_type"], "heartbeat")

            dispatcher.release_all()
            await tab1.disconnect()
            await tab2.disconnect()

    # -- isolation --------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    @patch("llm.get_llm_service")
    async def test_headless_loop_turn_is_not_gated(self, get_service):
        """Loop turns run on the worker and must never queue behind web chat.

        They reach the turn machinery through ``run_turn_to_completion``, which
        does not go through ``_stream_and_finalize`` and so never takes a slot.
        """
        from chat.loop_service import HeadlessTurnRunner

        stream = BlockingStream()
        stream.release.set()  # no need to hold this one open
        service = MagicMock()
        service.astream = stream.astream
        get_service.return_value = service

        @database_sync_to_async
        def _make_thread():
            return ChatThread.objects.create(created_by=self.user, title="loop")

        thread = await _make_thread()

        # Saturate the gate: every slot is taken, and the line is not empty.
        gate = turn_gate._gate()
        gate._enqueue("web-user")
        self.assertEqual(turn_gate.stats()["active"], 1)

        class _NullSink:
            wants_heartbeats = False

            async def send_event(self, event):
                return

        runner = HeadlessTurnRunner(
            user=self.user, thread_id=str(thread.id), sink=_NullSink(),
        )
        await asyncio.wait_for(
            runner.run_turn_to_completion(thread, "scheduled prompt"),
            timeout=10,
        )

        self.assertEqual(stream.calls, 1)
        self.assertEqual(turn_gate.stats()["waiting"], 0)

    # -- seeded continuations ---------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_QUEUE_TIMEOUT_SECONDS", 0.2)
    async def test_seeded_turn_times_out_silently(self):
        """A sub-agent continuation has no user message to resend.

        Telling a human to "send again" for a turn they never sent would be
        worse than saying nothing; the reported_at lease re-claims it instead.
        """
        from chat.consumers import ChatConsumer, _TurnState
        import threading

        @database_sync_to_async
        def _make_thread():
            return ChatThread.objects.create(created_by=self.user, title="seeded")

        thread = await _make_thread()

        consumer = ChatConsumer()
        consumer.scope = {"user": self.user}
        consumer.user = self.user
        consumer._stopped = False
        consumer._turn = None
        consumer._guardrail_task = None
        consumer._stream_task = None
        consumer._subagent_watch_task = None
        consumer._current_thread_id = str(thread.id)
        consumer._active_thread_id = str(thread.id)
        consumer.data_room_ids = []
        consumer.active_skill_ids = []
        consumer.resolved_prefs = None

        sent = []

        class _CapturingSink:
            wants_heartbeats = False

            async def send_event(self, event):
                sent.append(event)

        consumer._sink = _CapturingSink()

        turn = _TurnState(
            cancel_event=threading.Event(), stream_finished=asyncio.Event(),
        )
        consumer._turn = turn

        # Occupy the only slot so the seeded turn has to wait, then expire.
        gate = turn_gate._gate()
        gate._enqueue("someone-else")

        await consumer._stream_and_finalize(
            thread,
            content="",
            requested_model=None,
            thinking_level=None,
            resolved_model=None,
            max_context_tokens=None,
            history_mode="conversational",
            turn=turn,
            seed_mode=True,
        )

        self.assertEqual(
            [e for e in sent if e.get("event_type") == "error"], [],
            "a seeded continuation must not tell the user to send again",
        )
