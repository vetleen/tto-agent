"""A delivered reply must be saved even when the next message arrives at once.

The browser re-enables the composer on the turn's single final ``message_end``.
The stream itself ends only later — after ``LLMService.stream`` writes the
LLMCallLog (seconds on a big thread) — and the reply used to be saved only then.
A message sent in that gap made ``_handle_chat_message`` cancel the turn and the
reply was silently lost (reproduced 7/7 with Playwright). Now the reply is saved
as soon as ``message_end`` is forwarded, and a follow-up that still lands before
the save waits for it (``_await_reply_persisted``).

Same conventions as ``test_turn_gate_consumer``: a ``receive_json_from`` timeout
cancels the ASGI application, so waiting is done on Python-side conditions and
socket events are only read once they are known to be coming.
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
# Module import (not `from ... import`) so its TestCase classes are not
# re-discovered under this module's name.
from chat.tests import test_turn_gate_consumer as tg
from llm.types.streaming import StreamEvent

User = get_user_model()


def _ev(event_type, data=None, seq=0):
    return StreamEvent(event_type=event_type, data=data or {}, sequence=seq, run_id="r1")


def _answer(text):
    return [_ev("message_start"), _ev("token", {"text": text}, 1), _ev("message_end", {}, 2)]


class ParkAfterStream:
    """Yields its events at once, then parks before ending — like the real
    stream, which only ends after the LLMCallLog write (``log_stream``)."""

    def __init__(self, events):
        self.events = events
        self.done_yielding = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []

    async def astream(self, pipeline_id, request, **kwargs):
        self.requests.append(request)
        for event in self.events:
            yield event
        self.done_yielding.set()
        await self.release.wait()


class Dispatcher:
    """Hands each successive turn the next stream object."""

    def __init__(self, *streams):
        self.streams = list(streams)
        self._next = 0

    def astream(self, pipeline_id, request, **kwargs):
        stream = self.streams[min(self._next, len(self.streams) - 1)]
        self._next += 1
        return stream.astream(pipeline_id, request, **kwargs)


class ImmediateStream:
    def __init__(self, events):
        self.events = events
        self.requests = []

    async def astream(self, pipeline_id, request, **kwargs):
        self.requests.append(request)
        for event in self.events:
            yield event


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ReplyPersistRaceTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="race-owner@example.com", password="pass123")
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
        # cannot await.
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

    def _service(self, astream):
        service = MagicMock()
        service.astream = astream
        return service

    async def _send(self, tab, content, thread_id=None):
        payload = {"type": "chat.message", "content": content}
        if thread_id:
            payload["thread_id"] = thread_id
        await tab.send_json_to(payload)

    async def _read_until(self, tab, event_type, limit=40):
        seen = []
        for _ in range(limit):
            event = await tab.receive_json_from(timeout=5)
            seen.append(event.get("event_type"))
            if event.get("event_type") == event_type:
                return event, seen
        self.fail(f"{event_type} never arrived; saw {seen}")

    @database_sync_to_async
    def _assistant_rows(self, thread_id):
        return list(
            ChatMessage.objects.filter(thread_id=thread_id, role="assistant")
            .order_by("created_at")
            .values_list("content", flat=True)
        )

    @database_sync_to_async
    def _rows(self, thread_id):
        return list(
            ChatMessage.objects.filter(thread_id=thread_id)
            .order_by("created_at")
            .values("role", "content", "created_at", "is_hidden_from_user", "metadata")
        )

    async def _wait_rows(self, thread_id, predicate, message):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rows = await self._rows(thread_id)
            if predicate(rows):
                return rows
            await asyncio.sleep(0.02)
        self.fail(message)

    async def _settle(self, *tabs):
        await tg._until(lambda: turn_gate.stats()["active"] == 0, message="a turn kept its slot")
        await asyncio.sleep(0.2)
        for tab in tabs:
            await tab.disconnect()

    # -- tests --------------------------------------------------------------

    @patch("llm.get_llm_service")
    async def test_reply_is_saved_at_message_end_before_the_stream_ends(self, get_service):
        stream = ParkAfterStream(_answer("ALPHA-REPLY"))
        get_service.return_value = self._service(stream.astream)

        tab = await self._connect()
        await self._send(tab, "Hello")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        await self._read_until(tab, "message_end")
        await tg._until(lambda: stream.done_yielding.is_set())

        # The stream is still parked (its "LLMCallLog write" not done) — the
        # reply must already be in the DB.
        rows = await self._wait_rows(
            thread_id,
            lambda rows: any(r["role"] == "assistant" for r in rows),
            "reply not saved while the stream was still open",
        )
        self.assertEqual([r["content"] for r in rows if r["role"] == "assistant"], ["ALPHA-REPLY"])

        stream.release.set()
        await self._settle(tab)
        # Saved once, not again by the post-loop fallback.
        self.assertEqual(await self._assistant_rows(thread_id), ["ALPHA-REPLY"])

    @patch("llm.get_llm_service")
    async def test_follow_up_right_after_message_end_keeps_the_reply(self, get_service):
        """The original bug: message 2 arrives before reply 1 is written."""
        first = ParkAfterStream(_answer("ALPHA-REPLY"))
        second = ImmediateStream(_answer("BETA-REPLY"))
        get_service.return_value = self._service(Dispatcher(first, second).astream)

        # Make the assistant save slow so message 2 reliably lands before it.
        # The raw SyncToAsync object: class attribute access would bind it to None.
        original = ChatConsumer.__dict__["_create_message"]

        async def slow_create(consumer, thread, role, content, *args, **kwargs):
            if role == "assistant" and not kwargs.get("metadata", {}).get("tool_calls"):
                await asyncio.sleep(0.4)
            return await original(consumer, thread, role, content, *args, **kwargs)

        with patch.object(ChatConsumer, "_create_message", new=slow_create):
            tab = await self._connect()
            await self._send(tab, "First question")
            created, _ = await self._read_until(tab, "thread.created")
            thread_id = created["thread_id"]
            await self._read_until(tab, "message_end")
            # Immediately — the composer just re-enabled.
            await self._send(tab, "Second question", thread_id)

            rows = await self._wait_rows(
                thread_id,
                lambda rows: [r["content"] for r in rows if r["role"] == "assistant"]
                == ["ALPHA-REPLY", "BETA-REPLY"],
                "a reply was lost or the order is wrong",
            )
            await self._settle(tab)

        visible = [(r["role"], r["content"]) for r in rows if not r["is_hidden_from_user"]]
        self.assertEqual(
            visible,
            [("user", "First question"), ("assistant", "ALPHA-REPLY"),
             ("user", "Second question"), ("assistant", "BETA-REPLY")],
        )
        # Turn 2 was built on a history that already had reply 1.
        transcript = " ".join(
            str(m.content) for m in second.requests[0].messages if m.role == "assistant"
        )
        self.assertIn("ALPHA-REPLY", transcript)

    @patch("llm.get_llm_service")
    async def test_mid_stream_follow_up_still_supersedes(self, get_service):
        """Unchanged: a message sent while the answer is still streaming cancels it."""
        first = tg.BlockingStream(_answer("ALPHA-REPLY"))  # parks before any event
        second = ImmediateStream(_answer("BETA-REPLY"))
        get_service.return_value = self._service(Dispatcher(first, second).astream)

        tab = await self._connect()
        await self._send(tab, "First question")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        await tg._until(lambda: first.started.is_set())

        await self._send(tab, "Second question", thread_id)
        await self._wait_rows(
            thread_id,
            lambda rows: any(r["content"] == "BETA-REPLY" for r in rows),
            "the second turn never answered",
        )
        await self._settle(tab)
        self.assertEqual(await self._assistant_rows(thread_id), ["BETA-REPLY"])

    @patch("llm.get_llm_service")
    async def test_error_path_saves_nothing(self, get_service):
        events = [
            _ev("message_start"),
            _ev("token", {"text": "partial"}, 1),
            _ev("error", {"message": "boom"}, 2),
            _ev("message_end", {"output_tokens": 3}, 3),  # usage-only end after an error
        ]
        stream = ParkAfterStream(events)
        get_service.return_value = self._service(stream.astream)

        tab = await self._connect()
        await self._send(tab, "Hello")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        await self._read_until(tab, "message_end")
        stream.release.set()
        await self._settle(tab)
        self.assertEqual(await self._assistant_rows(thread_id), [])

    @patch("llm.get_llm_service")
    async def test_refusal_saves_nothing(self, get_service):
        events = [_ev("message_start"), _ev("message_end", {"stop_reason": "refusal"}, 1)]
        stream = ParkAfterStream(events)
        get_service.return_value = self._service(stream.astream)

        tab = await self._connect()
        await self._send(tab, "Hello")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        error, _ = await self._read_until(tab, "error")
        self.assertEqual(error["data"]["error_code"], "model_refusal")
        stream.release.set()
        await self._settle(tab)
        self.assertEqual(await self._assistant_rows(thread_id), [])

    @patch("llm.get_llm_service")
    async def test_stream_without_message_end_still_saves_after_the_loop(self, get_service):
        stream = ImmediateStream([_ev("message_start"), _ev("token", {"text": "NO-END"}, 1)])
        get_service.return_value = self._service(stream.astream)

        tab = await self._connect()
        await self._send(tab, "Hello")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        await self._wait_rows(
            thread_id,
            lambda rows: any(r["content"] == "NO-END" for r in rows),
            "fallback save after the loop did not happen",
        )
        await self._settle(tab)
        self.assertEqual(await self._assistant_rows(thread_id), ["NO-END"])

    @patch("llm.get_llm_service")
    async def test_tool_loop_rows_precede_the_final_reply(self, get_service):
        events = [
            _ev("message_start"),
            _ev("token", {"text": "Let me look."}, 1),
            _ev("tool_start", {"tool_call_id": "c1", "tool_name": "web_search", "arguments": {}}, 2),
            _ev("tool_end", {"tool_call_id": "c1", "tool_name": "web_search", "result": "{}"}, 3),
            _ev("message_start", {}, 4),
            _ev("token", {"text": "FINAL"}, 5),
            _ev("message_end", {}, 6),
        ]
        stream = ParkAfterStream(events)
        get_service.return_value = self._service(stream.astream)

        tab = await self._connect()
        await self._send(tab, "Search")
        created, _ = await self._read_until(tab, "thread.created")
        thread_id = created["thread_id"]
        await self._read_until(tab, "message_end")

        rows = await self._wait_rows(
            thread_id,
            lambda rows: any(r["content"] == "FINAL" for r in rows),
            "final reply not saved while the stream was open",
        )
        stream.release.set()
        await self._settle(tab)

        roles = [
            (r["role"], bool((r["metadata"] or {}).get("tool_calls"))) for r in rows
        ]
        self.assertEqual(
            roles,
            [("user", False), ("assistant", True), ("tool", False), ("assistant", False)],
        )
        self.assertEqual(await self._assistant_rows(thread_id), ["Let me look.", "FINAL"])


class AwaitReplyPersistedTests(SimpleTestCase):
    def _turn(self, delivered):
        turn = _TurnState(cancel_event=threading.Event(), stream_finished=asyncio.Event())
        turn.answer_delivered = delivered
        return turn

    async def test_returns_at_once_without_a_delivered_answer(self):
        consumer = ChatConsumer()
        await asyncio.wait_for(consumer._await_reply_persisted(None), timeout=0.5)
        await asyncio.wait_for(consumer._await_reply_persisted(self._turn(False)), timeout=0.5)
        turn = self._turn(True)
        turn.reply_persisted.set()
        await asyncio.wait_for(consumer._await_reply_persisted(turn), timeout=0.5)

    async def test_waits_until_the_reply_is_saved(self):
        consumer = ChatConsumer()
        turn = self._turn(True)
        waiter = asyncio.create_task(consumer._await_reply_persisted(turn))
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        turn.reply_persisted.set()
        await asyncio.wait_for(waiter, timeout=0.5)

    @override_settings(CHAT_REPLY_PERSIST_WAIT_SECONDS=0.05)
    async def test_gives_up_after_the_timeout(self):
        consumer = ChatConsumer()
        with self.assertLogs("chat.consumers", level="WARNING") as cm:
            await asyncio.wait_for(consumer._await_reply_persisted(self._turn(True)), timeout=1)
        self.assertIn("previous reply not saved", cm.output[0])
