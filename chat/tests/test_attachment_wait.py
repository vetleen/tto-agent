"""The turn hold: a chat turn waits until the attachments on its message have
finished worker-side processing (``ChatAttachment.processing_state``).

Unit tests drive ``_wait_for_attachments`` on a bare consumer; the socket tests
follow the conventions of ``test_turn_gate_consumer.py`` (poll Python-side
conditions, only read socket events whose sequence is known).
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings

from chat import turn_gate
from chat.consumers import ChatConsumer, _TurnState
from chat.models import ChatAttachment, ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from chat.tests.test_turn_gate_consumer import StreamDispatcher, _until

User = get_user_model()

_IN_MEMORY_STORAGE = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
_PDF_MIME = "application/pdf"
State = ChatAttachment.ProcessingState


def _make_turn(ids):
    return _TurnState(
        cancel_event=threading.Event(),
        stream_finished=asyncio.Event(),
        attachment_ids=[str(i) for i in ids],
    )


@_IN_MEMORY_STORAGE
@override_settings(CHAT_ATTACHMENT_READY_TIMEOUT_SECONDS=5)
@patch("chat.consumers._ATTACHMENT_POLL_INTERVAL_S", 0.01)
class WaitForAttachmentsTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="wait@example.com", password="pw")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _consumer(self, turn):
        c = ChatConsumer()
        c.scope = {"user": self.user}
        c.user = self.user
        c._stopped = False
        c._turn = turn
        return c

    async def _att(self, name="a.pdf", state=State.PENDING, message=None, user=None):
        return await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread,
            message=message,
            uploaded_by=user or self.user,
            file=SimpleUploadedFile(name, b"%PDF-1.4 x", content_type=_PDF_MIME),
            original_filename=name,
            content_type=_PDF_MIME,
            size_bytes=10,
            processing_state=state,
        )

    async def _set_state(self, att, state):
        await database_sync_to_async(
            ChatAttachment.objects.filter(pk=att.pk).update
        )(processing_state=state)

    async def test_ready_attachments_do_not_wait_or_announce(self):
        att = await self._att(state=State.READY)
        turn = _make_turn([att.id])
        on_waiting = AsyncMock()
        ok = await self._consumer(turn)._wait_for_attachments(
            self.thread, turn, seed_mode=False, on_waiting=on_waiting,
        )
        self.assertTrue(ok)
        on_waiting.assert_not_called()
        self.assertEqual(turn.attachments_not_ready, set())

    async def test_pending_attachment_holds_then_releases_when_ready(self):
        att = await self._att(name="deck.pdf")
        turn = _make_turn([att.id])
        on_waiting = AsyncMock()

        async def flip_soon():
            await asyncio.sleep(0.05)
            await self._set_state(att, State.READY)

        flipper = asyncio.create_task(flip_soon())
        ok = await self._consumer(turn)._wait_for_attachments(
            self.thread, turn, seed_mode=False, on_waiting=on_waiting,
        )
        await flipper
        self.assertTrue(ok)
        on_waiting.assert_awaited_once_with(["deck.pdf"])
        self.assertEqual(turn.attachments_not_ready, set())

    @override_settings(CHAT_ATTACHMENT_READY_TIMEOUT_SECONDS=0.05)
    async def test_timeout_proceeds_and_records_the_pending_ids(self):
        att = await self._att()
        turn = _make_turn([att.id])
        on_waiting = AsyncMock()
        ok = await self._consumer(turn)._wait_for_attachments(
            self.thread, turn, seed_mode=False, on_waiting=on_waiting,
        )
        self.assertTrue(ok)
        on_waiting.assert_awaited_once()
        self.assertEqual(turn.attachments_not_ready, {str(att.id)})

    async def test_cancel_while_waiting_returns_false(self):
        att = await self._att()
        turn = _make_turn([att.id])
        on_waiting = AsyncMock()

        async def cancel_soon():
            await asyncio.sleep(0.03)
            turn.cancel_event.set()

        canceller = asyncio.create_task(cancel_soon())
        ok = await self._consumer(turn)._wait_for_attachments(
            self.thread, turn, seed_mode=False, on_waiting=on_waiting,
        )
        await canceller
        self.assertFalse(ok)

    @override_settings(CHAT_ATTACHMENT_READY_TIMEOUT_SECONDS=0)
    async def test_hold_disabled_never_queries(self):
        att = await self._att()
        turn = _make_turn([att.id])
        c = self._consumer(turn)
        with patch.object(ChatConsumer, "_pending_attachments") as pending:
            ok = await c._wait_for_attachments(self.thread, turn, seed_mode=False, on_waiting=AsyncMock())
        self.assertTrue(ok)
        pending.assert_not_called()

    async def test_other_users_and_failed_rows_never_hold(self):
        other = await database_sync_to_async(User.objects.create_user)(email="o@example.com", password="pw")
        foreign = await self._att(user=other)
        failed = await self._att(state=State.FAILED)
        turn = _make_turn([foreign.id, failed.id])
        ok = await self._consumer(turn)._wait_for_attachments(
            self.thread, turn, seed_mode=False, on_waiting=AsyncMock(),
        )
        self.assertTrue(ok)

    async def test_seed_mode_waits_only_for_files_after_the_last_assistant_message(self):
        create = database_sync_to_async(ChatMessage.objects.create)
        old_user = await create(thread=self.thread, role="user", content="earlier")
        old_att = await self._att(name="old.pdf", message=old_user)
        await create(thread=self.thread, role="assistant", content="reply")
        disclaimer = await create(thread=self.thread, role="user", content="Files attached")
        new_att = await self._att(name="new.pdf", message=disclaimer)
        turn = _make_turn([])
        c = self._consumer(turn)
        self.assertEqual(await c._seed_attachment_ids(str(self.thread.id)), [str(new_att.id)])

        on_waiting = AsyncMock()

        async def flip_soon():
            await asyncio.sleep(0.05)
            await self._set_state(new_att, State.READY)

        flipper = asyncio.create_task(flip_soon())
        ok = await c._wait_for_attachments(self.thread, turn, seed_mode=True, on_waiting=on_waiting)
        await flipper
        self.assertTrue(ok)
        on_waiting.assert_awaited_once_with(["new.pdf"])
        # The old (still pending) file never held the turn.
        old_att = await database_sync_to_async(ChatAttachment.objects.get)(pk=old_att.pk)
        self.assertEqual(old_att.processing_state, State.PENDING)


@_IN_MEMORY_STORAGE
@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}},
    CHAT_ATTACHMENT_READY_TIMEOUT_SECONDS=10,
)
@patch("chat.consumers._ATTACHMENT_POLL_INTERVAL_S", 0.02)
class AttachmentHoldConsumerTests(TransactionTestCase):
    """Over the socket: a pending attachment holds the turn (no LLM call), the
    client hears ``turn.waiting_for_attachments``, and the turn runs once the
    row is ready — or never, when the user stops it meanwhile."""

    def setUp(self):
        self.user = User.objects.create_user(email="hold@example.com", password="pass123")
        self.thread = ChatThread.objects.create(created_by=self.user)
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

    async def _pending_pdf(self, name="deck.pdf"):
        return await database_sync_to_async(ChatAttachment.objects.create)(
            thread=self.thread,
            uploaded_by=self.user,
            file=SimpleUploadedFile(name, b"%PDF-1.4 x", content_type=_PDF_MIME),
            original_filename=name,
            content_type=_PDF_MIME,
            size_bytes=10,
            processing_state=State.PENDING,
        )

    async def _send_with_attachment(self, communicator, att):
        await communicator.send_json_to({
            "type": "chat.message",
            "content": "What is in the file?",
            "thread_id": str(self.thread.id),
            "attachment_ids": [str(att.id)],
        })
        # The hold announces itself before anything else streams; read until it
        # shows up (bounded, the event arrives on the first poll).
        for _ in range(8):
            event = await communicator.receive_json_from(timeout=5)
            if event.get("event_type") == "turn.waiting_for_attachments":
                return event
        raise AssertionError("turn.waiting_for_attachments never arrived")

    @patch("llm.get_llm_service")
    async def test_turn_waits_for_processing_then_streams(self, get_service):
        dispatcher = StreamDispatcher(1)
        get_service.return_value = self._service(dispatcher)
        att = await self._pending_pdf()
        comm = await self._connect()

        event = await self._send_with_attachment(comm, att)
        self.assertEqual(event["data"], {"names": ["deck.pdf"], "count": 1})
        # Held means held: the LLM has not been called.
        await asyncio.sleep(0.1)
        self.assertEqual(dispatcher.calls, 0)

        await database_sync_to_async(
            ChatAttachment.objects.filter(pk=att.pk).update
        )(processing_state=State.READY)
        await _until(lambda: dispatcher.streams[0].started.is_set(), message="held turn never started")
        dispatcher.release_all()

        for _ in range(20):
            evt = await comm.receive_json_from(timeout=5)
            if evt.get("event_type") == "message_end":
                break
        else:
            raise AssertionError("message_end never arrived")
        self.assertEqual(dispatcher.calls, 1)
        await comm.disconnect()

    @patch("llm.get_llm_service")
    async def test_stop_during_the_hold_never_calls_the_llm(self, get_service):
        dispatcher = StreamDispatcher(1)
        get_service.return_value = self._service(dispatcher)
        att = await self._pending_pdf()
        comm = await self._connect()

        await self._send_with_attachment(comm, att)
        await comm.send_json_to({"type": "chat.stop"})
        await asyncio.sleep(0.1)
        await database_sync_to_async(
            ChatAttachment.objects.filter(pk=att.pk).update
        )(processing_state=State.READY)
        await asyncio.sleep(0.2)
        self.assertEqual(dispatcher.calls, 0)
        await comm.disconnect()
