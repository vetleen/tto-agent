"""Tests for the provider-agnostic RealtimeTranscriptionSession.

The OpenAI implementation is exercised against a fake AsyncRealtime
connection so these tests never touch the live API. The fake mimics the
subset of the SDK interface that ``OpenAIRealtimeSession`` relies on:
``send``, iteration over server events, and the
``input_audio_buffer.append`` helper.
"""
from __future__ import annotations

import asyncio
import base64

from django.test import TestCase

from meetings.services.realtime_session import (
    OpenAIRealtimeSession,
    SessionError,
    SessionStatus,
    TranscriptCompleted,
    TranscriptDelta,
    UnsupportedModelError,
    build_realtime_session,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Bag-of-attributes standing in for the SDK's typed server events."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeInputAudioBuffer:
    def __init__(self, conn):
        self.conn = conn

    async def append(self, *, audio: str) -> None:
        self.conn.appended_b64.append(audio)

    async def commit(self) -> None:
        self.conn.commit_calls += 1


class _FakeConnection:
    """Stand-in for AsyncRealtimeConnection.

    Drives an async queue of server events that tests push onto; async
    iteration over the connection yields them in order.
    """

    def __init__(self, script: list[_FakeEvent]):
        self._script = list(script)
        self._queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.appended_b64: list[str] = []
        self.commit_calls = 0
        self.closed = False
        self.input_audio_buffer = _FakeInputAudioBuffer(self)
        for evt in self._script:
            self._queue.put_nowait(evt)

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        # Unblock any pending iteration.
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        evt = await self._queue.get()
        if evt is None:
            raise StopAsyncIteration
        return evt

    # Tests push additional events via this:
    def push(self, evt):
        self._queue.put_nowait(evt)


class _FakeConnectionManager:
    def __init__(self, conn: _FakeConnection):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeRealtimeNamespace:
    def __init__(self, conn: _FakeConnection):
        self.conn = conn

    def connect(self, *, model: str):
        return _FakeConnectionManager(self.conn)


class _FakeBeta:
    def __init__(self, conn: _FakeConnection):
        self.realtime = _FakeRealtimeNamespace(conn)


class _FakeAsyncOpenAI:
    def __init__(self, conn: _FakeConnection):
        self.beta = _FakeBeta(conn)


def _factory_for(conn: _FakeConnection):
    def _f():
        return _FakeAsyncOpenAI(conn)
    return _f


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect_events(session, *, until_kinds: set[type], limit: int = 5):
    """Consume ``session.events()`` until N events of the given types arrive."""
    seen = []
    async for evt in session.events():
        seen.append(evt)
        if any(isinstance(evt, k) for k in until_kinds) and len(seen) >= limit:
            return seen
        if len(seen) >= limit * 3:
            return seen
    return seen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class FactoryTests(TestCase):
    def test_unknown_model_raises(self):
        with self.assertRaises(UnsupportedModelError):
            build_realtime_session(model_id="openai/not-a-real-model")

    def test_diarize_model_rejected_for_live(self):
        with self.assertRaises(UnsupportedModelError):
            build_realtime_session(model_id="openai/gpt-4o-transcribe-diarize")

    def test_mini_transcribe_builds_openai_session(self):
        # Build it but don't connect — just prove the factory dispatches
        # without touching the network.
        sess = build_realtime_session(model_id="openai/gpt-4o-mini-transcribe")
        self.assertIsInstance(sess, OpenAIRealtimeSession)


class OpenAIRealtimeSessionTests(TestCase):
    def test_connect_sends_transcription_session_update(self):
        conn = _FakeConnection(script=[
            _FakeEvent(type="transcription_session.created", session=_FakeEvent(id="sess_1")),
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            prompt="OncoBio Therapeutics meeting",
            language="en",
            _client_factory=_factory_for(conn),
        )

        async def run():
            await session.connect()
            # Let the recv loop drain the queued event.
            await asyncio.sleep(0)
            await session.aclose()

        _run(run())

        # The SDK's "send" accepts a dict for raw event shapes.
        self.assertEqual(len(conn.sent), 1)
        sent = conn.sent[0]
        self.assertEqual(sent["type"], "transcription_session.update")
        cfg = sent["session"]
        self.assertEqual(cfg["input_audio_format"], "pcm16")
        self.assertEqual(cfg["input_audio_transcription"]["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(cfg["input_audio_transcription"]["language"], "en")
        self.assertIn("OncoBio Therapeutics", cfg["input_audio_transcription"]["prompt"])

    def test_deltas_and_completed_surface_as_structured_events(self):
        conn = _FakeConnection(script=[
            _FakeEvent(type="transcription_session.created", session=_FakeEvent(id="sess_1")),
            _FakeEvent(
                type="conversation.item.input_audio_transcription.delta",
                item_id="item_1",
                delta="Hello ",
            ),
            _FakeEvent(
                type="conversation.item.input_audio_transcription.delta",
                item_id="item_1",
                delta="world.",
            ),
            _FakeEvent(
                type="conversation.item.input_audio_transcription.completed",
                item_id="item_1",
                transcript="Hello world.",
                usage=_FakeEvent(input_tokens=10, output_tokens=4, total_tokens=14),
            ),
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _client_factory=_factory_for(conn),
        )

        async def run():
            await session.connect()
            deltas: list[str] = []
            completed: list[TranscriptCompleted] = []
            async for evt in session.events():
                if isinstance(evt, TranscriptDelta):
                    deltas.append(evt.text)
                elif isinstance(evt, TranscriptCompleted):
                    completed.append(evt)
                    break
            await session.aclose()
            return deltas, completed

        deltas, completed = _run(run())
        self.assertEqual(deltas, ["Hello ", "world."])
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].text, "Hello world.")
        self.assertEqual(completed[0].usage, {
            "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
        })

    def test_send_pcm_appends_base64_audio(self):
        conn = _FakeConnection(script=[
            _FakeEvent(type="transcription_session.created", session=_FakeEvent(id="sess_1")),
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _client_factory=_factory_for(conn),
        )

        async def run():
            await session.connect()
            await session.send_pcm(b"\x01\x02\x03\x04")
            await session.send_pcm(b"\x05\x06\x07\x08")
            await session.aclose()

        _run(run())
        self.assertEqual(len(conn.appended_b64), 2)
        self.assertEqual(base64.b64decode(conn.appended_b64[0]), b"\x01\x02\x03\x04")

    def test_fatal_error_event_marks_session_error(self):
        conn = _FakeConnection(script=[
            _FakeEvent(type="transcription_session.created", session=_FakeEvent(id="sess_1")),
            _FakeEvent(
                type="error",
                error=_FakeEvent(code="invalid_api_key", message="auth failed"),
            ),
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _client_factory=_factory_for(conn),
        )

        async def run():
            await session.connect()
            errors: list[SessionError] = []
            async for evt in session.events():
                if isinstance(evt, SessionError):
                    errors.append(evt)
                    break
            await session.aclose()
            return errors

        errors = _run(run())
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].code, "invalid_api_key")
        self.assertTrue(errors[0].fatal)

    def test_aclose_emits_disconnected_status(self):
        conn = _FakeConnection(script=[
            _FakeEvent(type="transcription_session.created", session=_FakeEvent(id="sess_1")),
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _client_factory=_factory_for(conn),
        )

        async def run():
            await session.connect()
            await session.aclose()
            # Drain the event queue after close — should end with disconnected.
            seen = []
            async for evt in session.events():
                seen.append(evt)
            return seen

        seen = _run(run())
        self.assertTrue(any(
            isinstance(e, SessionStatus) and e.state == "disconnected"
            for e in seen
        ))
