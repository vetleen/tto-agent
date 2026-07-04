"""Lifecycle/teardown safety tests for OpenAIRealtimeSession.

Covers the review-batch fixes:
- C8: a reconnect must not reopen a session after aclose().
- C11: aclose() must not block forever on a full event queue.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from django.test import TestCase

from meetings.services.realtime_session import (
    OpenAIRealtimeSession,
    RealtimeSessionError,
    SessionStatus,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_session(**kwargs):
    return OpenAIRealtimeSession(
        model_id="openai/gpt-4o-mini-transcribe",
        _api_key="sk-fake",
        **kwargs,
    )


class ReconnectAfterCloseTests(TestCase):
    def test_open_refuses_when_closed(self):
        session = _make_session()
        session._closed = True

        async def run():
            with self.assertRaises(RealtimeSessionError):
                await session._open()

        _run(run())

    def test_reconnect_aborts_when_closed_during_backoff(self):
        # Simulate aclose() landing during the backoff sleep: the post-sleep
        # _closed re-check must abort before reopening a fresh socket.
        opened = {"count": 0}

        async def _fake_open(self):
            opened["count"] += 1

        async def run():
            session = _make_session()

            async def _sleep_then_close(_delay):
                session._closed = True  # aclose() happened mid-backoff

            with patch("meetings.services.realtime_session.asyncio.sleep", _sleep_then_close), \
                    patch.object(OpenAIRealtimeSession, "_open", _fake_open):
                await session._try_reconnect()

            # _open must NOT have run — no leaked reopened socket.
            self.assertEqual(opened["count"], 0)

        _run(run())

    def test_aclose_cancels_pending_reconnect(self):
        async def run():
            session = _make_session()

            async def _park():
                await asyncio.sleep(3600)

            session._reconnect_task = asyncio.create_task(_park())
            await asyncio.sleep(0)  # let it start
            await session.aclose()
            self.assertTrue(
                session._reconnect_task is None
                or session._reconnect_task.cancelled()
            )

        _run(run())


class AcloseQueueTests(TestCase):
    def test_aclose_does_not_block_on_full_event_queue(self):
        async def run():
            session = _make_session()
            # Fill the bounded (256) event queue so a blocking put would hang.
            for _ in range(session._events.maxsize):
                session._events.put_nowait(SessionStatus(state="reconnecting"))

            # Must complete promptly (courtesy disconnected event is dropped).
            await asyncio.wait_for(session.aclose(), timeout=1.0)
            self.assertTrue(session._closed)

        _run(run())
