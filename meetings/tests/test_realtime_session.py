"""Tests for the provider-agnostic RealtimeTranscriptionSession.

``OpenAIRealtimeSession`` opens a raw WebSocket to
``wss://api.openai.com/v1/realtime?intent=transcription`` and speaks
OpenAI's **GA** Realtime transcription dialect: no ``OpenAI-Beta`` header,
a ``session.update`` message with config nested under
``session.audio.input`` (audio ``format`` as an object), and
``session.created``/``session.updated`` lifecycle events. The retired beta
shape (``transcription_session.update`` + flat keys) closed every connect
with ``invalid_request_error.beta_api_shape_disabled``. The test seam is a
factory that returns a fake websocket with ``send``, async iteration, and
``close``.
"""
from __future__ import annotations

import asyncio
import base64
import json

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


class _FakeWebSocket:
    """Minimal stand-in for a websockets.asyncio.client connection.

    Exposes the subset used by OpenAIRealtimeSession: ``send`` (text frames),
    async iteration over inbound messages, and ``close``. Tests push server
    events via ``push_server_json`` — they're JSON-serialised to match the
    real wire format.
    """

    def __init__(self, script: list[dict] | None = None):
        self._queue: asyncio.Queue = asyncio.Queue()
        self.sent: list[str] = []
        self.closed = False
        for evt in (script or []):
            self._queue.put_nowait(json.dumps(evt))

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self, *args, **kwargs) -> None:
        self.closed = True
        # Unblock any pending iteration.
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    def push_server_json(self, event: dict) -> None:
        self._queue.put_nowait(json.dumps(event))


def _factory_for(ws: _FakeWebSocket, captured: dict | None = None):
    """Return a connect-factory that yields ``ws`` and optionally records args."""
    async def factory(api_key: str, url: str):
        if captured is not None:
            captured["api_key"] = api_key
            captured["url"] = url
        return ws
    return factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


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
    def test_connect_hits_transcription_url_and_sends_session_update(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        captured: dict = {}
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            prompt="OncoBio Therapeutics meeting",
            language="en",
            _ws_connect_factory=_factory_for(ws, captured),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await asyncio.sleep(0)  # let recv loop drain the queued event
            await session.aclose()

        _run(run())

        self.assertEqual(
            captured["url"],
            "wss://api.openai.com/v1/realtime?intent=transcription",
        )
        self.assertEqual(captured["api_key"], "sk-fake")

        # First (and only) sent frame should be the GA session.update event.
        self.assertEqual(len(ws.sent), 1)
        payload = json.loads(ws.sent[0])
        self.assertEqual(payload["type"], "session.update")
        cfg = payload["session"]
        self.assertEqual(cfg["type"], "transcription")
        audio_input = cfg["audio"]["input"]
        self.assertEqual(audio_input["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(audio_input["transcription"]["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(audio_input["transcription"]["language"], "en")
        self.assertIn("OncoBio Therapeutics", audio_input["transcription"]["prompt"])
        self.assertEqual(audio_input["turn_detection"]["type"], "server_vad")
        self.assertEqual(audio_input["noise_reduction"], {"type": "near_field"})

    def test_long_prompt_is_capped_at_1024(self):
        # OpenAI's Realtime transcription endpoint closes the session with
        # ``string_above_max_length`` if the prompt exceeds 1024 chars.
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            prompt="x" * 5000,
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await session.aclose()

        _run(run())

        payload = json.loads(ws.sent[0])
        prompt = payload["session"]["audio"]["input"]["transcription"]["prompt"]
        self.assertEqual(len(prompt), 1024)

    def test_missing_api_key_raises(self):
        # Patch the env so even a machine running with a real OPENAI_API_KEY
        # still exercises the missing-key branch.
        import os
        from unittest.mock import patch
        ws = _FakeWebSocket()

        async def run():
            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
                session = OpenAIRealtimeSession(
                    model_id="openai/gpt-4o-mini-transcribe",
                    _ws_connect_factory=_factory_for(ws),
                )
                await session.connect()

        from meetings.services.realtime_session import RealtimeSessionError
        with self.assertRaises(RealtimeSessionError):
            _run(run())

    def test_deltas_and_completed_surface_as_structured_events(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "Hello ",
            },
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "item_1",
                "delta": "world.",
            },
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "Hello world.",
                "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            },
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
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

    def test_send_pcm_encodes_as_input_audio_buffer_append(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await session.send_pcm(b"\x01\x02\x03\x04")
            await session.send_pcm(b"\x05\x06\x07\x08")
            await session.aclose()

        _run(run())

        audio_frames = [json.loads(m) for m in ws.sent if '"input_audio_buffer.append"' in m]
        self.assertEqual(len(audio_frames), 2)
        self.assertEqual(audio_frames[0]["type"], "input_audio_buffer.append")
        self.assertEqual(base64.b64decode(audio_frames[0]["audio"]), b"\x01\x02\x03\x04")

    def test_fatal_error_event_marks_session_error(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "error", "error": {"code": "invalid_api_key", "message": "auth failed"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
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

    def test_invalid_request_error_is_fatal_not_reconnected(self):
        # A config/request rejection (e.g. the retired beta shape, which closed
        # with invalid_request_error) is identical on reconnect — classifying it
        # off error.type keeps it fatal so we don't loop the backoff into the same
        # rejection (the "Reconnecting…" storm). The code field may be absent or
        # an unrecognised string, so error.type is what makes it fatal here.
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "error", "error": {
                "type": "invalid_request_error",
                "code": "beta_api_shape_disabled",
                "message": "The beta Realtime API shape is no longer supported.",
            }},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
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
        self.assertTrue(errors[0].fatal)

    def test_commit_empty_error_is_benign_not_surfaced(self):
        # Under server-VAD the server auto-commits on end-of-speech, so the
        # defensive commit finalize() sends often hits an already-empty buffer
        # -> ``input_audio_buffer_commit_empty`` (an ``invalid_request_error``).
        # That's an expected race, not a failure: it must NOT surface as a
        # SessionError (which the consumer treats as a reason to tear realtime
        # down and fall back to chunked). A real transcript queued *after* the
        # error proves the session stayed alive and the error was swallowed.
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "error", "error": {
                "type": "invalid_request_error",
                "code": "input_audio_buffer_commit_empty",
                "message": "Error committing input audio buffer: buffer is empty.",
            }},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "still here.",
            },
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            seen: list = []
            async for evt in session.events():
                seen.append(evt)
                if isinstance(evt, TranscriptCompleted):
                    break
            await session.aclose()
            return seen

        seen = _run(run())
        self.assertFalse(
            any(isinstance(e, SessionError) for e in seen),
            "commit_empty must not surface as a SessionError",
        )
        self.assertTrue(any(
            isinstance(e, TranscriptCompleted) and e.text == "still here."
            for e in seen
        ))

    def test_aclose_emits_disconnected_status(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await session.aclose()
            seen = []
            async for evt in session.events():
                seen.append(evt)
            return seen

        seen = _run(run())
        self.assertTrue(any(
            isinstance(e, SessionStatus) and e.state == "disconnected"
            for e in seen
        ))


class FinalizeTests(TestCase):
    """finalize() must wait for the server's post-commit transcription so the
    user's last utterance isn't dropped on Stop — bounded by `timeout`."""

    def test_finalize_waits_for_post_commit_completed(self):
        class _CommitRespondingWS(_FakeWebSocket):
            async def send(self, payload):
                await super().send(payload)
                if json.loads(payload).get("type") == "input_audio_buffer.commit":
                    # Server reacts to the commit with a final transcription —
                    # the case the old drain loop returned too early to catch.
                    self.push_server_json({
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "item_final",
                        "transcript": "the last words.",
                    })

        ws = _CommitRespondingWS(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            completed: list[TranscriptCompleted] = []

            async def consume():
                async for evt in session.events():
                    if isinstance(evt, TranscriptCompleted):
                        completed.append(evt)
                    elif isinstance(evt, SessionStatus) and evt.state == "disconnected":
                        return

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0.05)  # let session.created drain (queue empties)
            await session.finalize(timeout=2.0)
            await session.aclose()
            await consumer
            return completed

        completed = _run(run())
        self.assertTrue(
            any(e.text == "the last words." for e in completed),
            "finalize() must wait for the post-commit .completed event",
        )
        self.assertTrue(
            any(json.loads(s).get("type") == "input_audio_buffer.commit" for s in ws.sent)
        )

    def test_finalize_returns_within_timeout_when_server_silent(self):
        # No server response to the commit — finalize must not hang; it returns
        # at the deadline (the failsafe ceiling).
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await asyncio.sleep(0.05)
            # A short timeout keeps the test fast; the assertion is that this
            # await completes at all (no hang) and the commit was sent.
            await asyncio.wait_for(session.finalize(timeout=0.3), timeout=2.0)
            await session.aclose()

        _run(run())
        self.assertTrue(
            any(json.loads(s).get("type") == "input_audio_buffer.commit" for s in ws.sent)
        )
