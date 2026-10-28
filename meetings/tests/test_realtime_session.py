"""Tests for the realtime transcription session.

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
from unittest.mock import patch

from django.test import TestCase

from meetings.services.realtime_session import (
    OpenAIRealtimeSession,
    RealtimeSessionError,
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


_FRAME_BYTES = 9600  # 200 ms of 24 kHz PCM16, what PcmPipe forwards


def _tone_ms(ms: int, amp: float = 0.2) -> bytes:
    import numpy as np
    t = np.arange(24 * ms) / 24_000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def _silence_ms(ms: int) -> bytes:
    return b"\x00" * (48 * ms)


async def _send_audio(session, audio: bytes) -> None:
    for i in range(0, len(audio), _FRAME_BYTES):
        await session.send_pcm(audio[i:i + _FRAME_BYTES])


def _sent_types(ws) -> list[str]:
    return [json.loads(m).get("type") for m in ws.sent]


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
        # We commit turns ourselves: turn_detection must be an explicit null
        # (an omitted key leaves the session's default server VAD on).
        self.assertIn("turn_detection", audio_input)
        self.assertIsNone(audio_input["turn_detection"])
        # far_field: meetings are captured by laptop / room mics, not headsets.
        self.assertEqual(audio_input["noise_reduction"], {"type": "far_field"})

    def test_completed_reports_utterance_start_offset_not_end(self):
        """start_offset_seconds must reflect where the utterance STARTED (the
        speech onset the turn detector saw), not the larger total forwarded by
        the time the .completed event arrives; duration_seconds is what the
        commit covered."""
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
            # 1 s of silence, 2 s of speech, then the pause that closes it.
            await _send_audio(session, _silence_ms(1000) + _tone_ms(2000) + _silence_ms(600))
            ws.push_server_json({"type": "input_audio_buffer.committed", "item_id": "item_1"})
            # More audio arrives before the transcript does.
            await _send_audio(session, _silence_ms(2000))
            ws.push_server_json({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "hello",
                "usage": {"type": "duration", "seconds": 4},
            })
            evt = await session._events.get()
            while not isinstance(evt, TranscriptCompleted):
                evt = await session._events.get()
            await session.aclose()
            return evt

        completed = _run(run())
        self.assertIsInstance(completed, TranscriptCompleted)
        self.assertAlmostEqual(completed.start_offset_seconds, 1.0)
        self.assertAlmostEqual(completed.duration_seconds, 3.6)
        self.assertEqual(completed.usage["seconds"], 4)
        self.assertEqual(completed.usage["type"], "duration")

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
            "type": None, "seconds": None,
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

    def test_failed_utterance_surfaces_nonfatal_error_and_keeps_counters_paired(self):
        # A committed buffer whose transcription fails server-side emits
        # ``conversation.item.input_audio_transcription.failed``. It must
        # (a) surface as a NON-fatal SessionError (the utterance is lost but
        # the session is healthy), (b) advance the completed counter so
        # finalize() doesn't wait out its full timeout for a .completed that
        # will never come, and (c) forget the item's commit bookkeeping.
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "input_audio_buffer.committed", "item_id": "item_1"},
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": "item_1",
                "error": {"type": "server_error", "message": "processing failed"},
            },
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
        self.assertEqual(errors[0].code, "utterance_failed")
        self.assertFalse(errors[0].fatal)
        self.assertEqual(session._committed_count, session._completed_count)
        self.assertEqual(session._items, {})

    def test_reconnect_resets_commit_completed_pairing(self):
        # Commits orphaned by a dropped socket never produce a .completed on
        # the new one. If _open() didn't reset the counters, finalize() would
        # burn its full timeout on every Stop after any reconnect.
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
            session._committed_count = 3
            session._completed_count = 1
            old_recv = session._recv_task
            await session._open()  # what _try_reconnect calls
            # The old recv task iterates the same fake socket; retire it so it
            # doesn't outlive the test loop.
            old_recv.cancel()
            try:
                await old_recv
            except (asyncio.CancelledError, Exception):
                pass
            await session.aclose()

        _run(run())
        self.assertEqual(session._committed_count, 0)
        self.assertEqual(session._completed_count, 0)

    def test_clean_server_close_triggers_reconnect(self):
        # The websockets library ends iteration *normally* on a clean close
        # (e.g. the server retiring the session at its lifetime cap). That
        # must schedule a reconnect exactly like an abnormal drop — the old
        # code only reconnected from the exception path, so a clean close
        # became a silent stall: audio kept flowing into a dead session.
        from unittest.mock import patch as mock_patch

        ws1 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        ws2 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_2"}},
        ])
        sockets = iter([ws1, ws2])

        async def factory(api_key: str, url: str):
            return next(sockets)

        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=factory,
            _api_key="sk-fake",
        )

        async def run():
            with mock_patch(
                "meetings.services.realtime_session._RECONNECT_BACKOFFS",
                (0.0,) * 8,
            ):
                await session.connect()
                # Server closes ws1 cleanly: iteration ends without an exception.
                ws1._queue.put_nowait(None)
                states: list[str] = []
                async for evt in session.events():
                    if isinstance(evt, SessionStatus):
                        states.append(evt.state)
                        # initial connect + reconnecting + reconnected
                        if states.count("connected") == 2:
                            break
                await session.aclose()
                return states

        states = _run(run())
        self.assertIn("reconnecting", states)
        self.assertEqual(states.count("connected"), 2)

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

    def test_turn_lifecycle_surfaces_as_session_activity(self):
        # A long open utterance may produce no final text until it closes;
        # these three events let the consumer (and the client's stall
        # watchdog) tell "listening" from "stalled". speech_started/stopped
        # come from our detector, committed from the server's ack.
        from meetings.services.realtime_session import SessionActivity

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
            await _send_audio(session, _tone_ms(1000) + _silence_ms(600))
            ws.push_server_json({"type": "input_audio_buffer.committed", "item_id": "item_1"})
            kinds: list[str] = []
            async for evt in session.events():
                if isinstance(evt, SessionActivity):
                    kinds.append(evt.kind)
                    if len(kinds) == 3:
                        break
            items = dict(session._items)
            await session.aclose()
            return kinds, items

        kinds, items = _run(run())
        self.assertEqual(kinds, ["speech_started", "speech_stopped", "committed"])
        self.assertEqual(session._committed_count, 1)
        self.assertIn("item_1", items)
        self.assertEqual(_sent_types(ws).count("input_audio_buffer.commit"), 1)

    def test_server_vad_events_are_ignored(self):
        from meetings.services.realtime_session import SessionActivity

        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": "item_1",
                "transcript": "done",
            },
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            seen = []
            async for evt in session.events():
                seen.append(evt)
                if isinstance(evt, TranscriptCompleted):
                    break
            await session.aclose()
            return seen

        with self.assertLogs("meetings.services.realtime_session", level="WARNING") as logs:
            seen = _run(run())
        self.assertFalse(any(isinstance(e, SessionActivity) for e in seen))
        self.assertEqual(sum("unexpected server VAD event" in line for line in logs.output), 1)

    def test_speech_then_silence_sends_commit_and_silence_alone_does_not(self):
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
            await _send_audio(session, _silence_ms(3000))
            before = _sent_types(ws).count("input_audio_buffer.commit")
            await _send_audio(session, _tone_ms(1400) + _silence_ms(600))
            after = _sent_types(ws).count("input_audio_buffer.commit")
            await session.aclose()
            return before, after

        before, after = _run(run())
        self.assertEqual(before, 0)
        self.assertEqual(after, 1)
        # The commit follows the frame that closed the turn, never precedes it.
        types = _sent_types(ws)
        self.assertEqual(types[-1], "input_audio_buffer.commit")
        self.assertEqual(session._uncommitted_bytes, 0)

    def test_long_silence_is_cleared_upstream(self):
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
            await _send_audio(session, _silence_ms(16_000))
            await session.aclose()

        _run(run())
        types = _sent_types(ws)
        self.assertEqual(types.count("input_audio_buffer.clear"), 1)
        self.assertNotIn("input_audio_buffer.commit", types)
        self.assertLess(session._uncommitted_bytes, 2 * 48_000)

    def test_reconnect_replays_uncommitted_turn_and_commits_pending(self):
        # A turn that ends while the socket is down is replayed on the new
        # socket and committed at its boundary; audio after it stays
        # uncommitted. Already-committed audio is never replayed.
        ws1 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        ws2 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_2"}},
        ])
        sockets = iter([ws1, ws2])

        async def factory(api_key: str, url: str):
            return next(sockets)

        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=factory,
            _api_key="sk-fake",
        )
        committed_turn = _tone_ms(1000) + _silence_ms(600)
        offline_turn = _tone_ms(1000) + _silence_ms(600)
        trailing = _silence_ms(400)

        async def run():
            with patch(
                "meetings.services.realtime_session._RECONNECT_BACKOFFS",
                (0.2,) * 8,
            ):
                await session.connect()
                await _send_audio(session, committed_turn)
                # Socket drops; audio keeps arriving during the backoff.
                ws1._queue.put_nowait(None)
                await asyncio.sleep(0.05)
                await _send_audio(session, offline_turn + trailing)
                states: list[str] = []
                async for evt in session.events():
                    if isinstance(evt, SessionStatus):
                        states.append(evt.state)
                        if states.count("connected") == 2:
                            break
                await session.aclose()

        _run(run())
        self.assertEqual(_sent_types(ws1).count("input_audio_buffer.commit"), 1)
        sent2 = [json.loads(m) for m in ws2.sent]
        types2 = [m["type"] for m in sent2]
        self.assertEqual(types2.count("input_audio_buffer.commit"), 1)
        replayed = b"".join(
            base64.b64decode(m["audio"]) for m in sent2 if m["type"] == "input_audio_buffer.append"
        )
        self.assertEqual(replayed, offline_turn + trailing)
        # The commit sits exactly at the offline turn's boundary.
        commit_at = types2.index("input_audio_buffer.commit")
        before_commit = b"".join(
            base64.b64decode(m["audio"]) for m in sent2[:commit_at]
            if m["type"] == "input_audio_buffer.append"
        )
        self.assertEqual(before_commit, offline_turn)
        # The trailing silence was replayed but no commit covers it yet.
        self.assertEqual(session._uncommitted_bytes, len(trailing))


    def test_update_language_resends_session_update_when_open(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            language="en",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await session.update_language("no")
            await session.aclose()

        _run(run())

        updates = [
            json.loads(m) for m in ws.sent if json.loads(m).get("type") == "session.update"
        ]
        self.assertEqual(len(updates), 2)
        self.assertEqual(
            updates[0]["session"]["audio"]["input"]["transcription"]["language"], "en",
        )
        self.assertEqual(
            updates[1]["session"]["audio"]["input"]["transcription"]["language"], "no",
        )
        self.assertEqual(session.language, "no")

    def test_update_language_to_auto_reopens_without_language(self):
        # session.update merges, so re-sending the config without a language
        # key would not reliably clear a pinned one. Auto-detect reopens the
        # socket instead; the reconnect path sends a fresh config with no
        # language, and the self-inflicted close is logged as routine.
        ws1 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        ws2 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_2"}},
        ])
        sockets = iter([ws1, ws2])

        async def factory(api_key: str, url: str):
            return next(sockets)

        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            language="no",
            _ws_connect_factory=factory,
            _api_key="sk-fake",
        )

        async def run():
            with patch(
                "meetings.services.realtime_session._RECONNECT_BACKOFFS",
                (0.0,) * 8,
            ):
                await session.connect()
                await session.update_language(None)
                states: list[str] = []
                async for evt in session.events():
                    if isinstance(evt, SessionStatus):
                        states.append(evt.state)
                        if states.count("connected") == 2:
                            break
                await session.aclose()
                return states

        with self.assertLogs("meetings.services.realtime_session", level="INFO") as logs:
            states = _run(run())

        self.assertTrue(ws1.closed)
        self.assertIn("reconnecting", states)
        self.assertIsNone(session.language)
        cfg = json.loads(ws2.sent[0])
        self.assertEqual(cfg["type"], "session.update")
        self.assertNotIn("language", cfg["session"]["audio"]["input"]["transcription"])
        self.assertTrue(any("reopening to apply config" in line for line in logs.output))
        self.assertFalse(any("server closed the socket" in line for line in logs.output))

    def test_pcm_during_reopen_backoff_is_buffered_not_warned(self):
        # Seen on the local run: during the reopen's backoff the pump kept
        # writing to the dead socket — one WARNING per 250 ms frame. The
        # socket must be forgotten before the backoff so frames buffer
        # silently and replay once the new socket is up.
        ws1 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        ws2 = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_2"}},
        ])
        sockets = iter([ws1, ws2])

        async def factory(api_key: str, url: str):
            return next(sockets)

        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            language="no",
            _ws_connect_factory=factory,
            _api_key="sk-fake",
        )

        async def run():
            with patch(
                "meetings.services.realtime_session._RECONNECT_BACKOFFS",
                (0.3,) * 8,
            ):
                await session.connect()
                await session.update_language(None)
                await asyncio.sleep(0.05)  # inside the backoff: no socket yet
                await session.send_pcm(b"\x01\x02\x03\x04")
                states: list[str] = []
                async for evt in session.events():
                    if isinstance(evt, SessionStatus):
                        states.append(evt.state)
                        if states.count("connected") == 2:
                            break
                await session.aclose()

        with self.assertLogs("meetings.services.realtime_session", level="INFO") as logs:
            _run(run())

        self.assertFalse(any("append failed" in line for line in logs.output), logs.output)
        # The frame sent during the backoff was replayed on the new socket.
        replayed = [
            json.loads(m) for m in ws2.sent if '"input_audio_buffer.append"' in m
        ]
        self.assertEqual(len(replayed), 1)
        self.assertEqual(base64.b64decode(replayed[0]["audio"]), b"\x01\x02\x03\x04")

    def test_update_language_while_disconnected_defers_to_reconnect(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            language="en",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            sent_before = len(ws.sent)
            session._ws = None  # mid-reconnect: nothing to send on
            await session.update_language("no")
            frames_sent = len(ws.sent) - sent_before
            session._ws = ws
            await session.aclose()
            return frames_sent

        self.assertEqual(_run(run()), 0)
        # Stored for the next _open(), which re-sends the config.
        self.assertEqual(session.language, "no")

    def test_session_echo_logs_effective_language(self):
        ws = _FakeWebSocket(script=[
            {"type": "session.created", "session": {"id": "sess_1"}},
            {
                "type": "session.updated",
                "session": {
                    "id": "sess_1",
                    "audio": {"input": {"transcription": {"language": "no"}}},
                },
            },
        ])
        session = OpenAIRealtimeSession(
            model_id="openai/gpt-4o-mini-transcribe",
            _ws_connect_factory=_factory_for(ws),
            _api_key="sk-fake",
        )

        async def run():
            await session.connect()
            await asyncio.sleep(0.05)
            await session.aclose()

        with self.assertLogs("meetings.services.realtime_session", level="INFO") as logs:
            _run(run())
        self.assertTrue(any("session.updated (id=sess_1 language=no)" in line for line in logs.output))
        self.assertTrue(any("session.created (id=sess_1 language=auto)" in line for line in logs.output))


class FinalizeTests(TestCase):
    """finalize() must commit the open turn and wait for the server's
    transcription so the user's last utterance isn't dropped on Stop —
    bounded by `timeout`."""

    def test_finalize_waits_for_post_commit_completed(self):
        class _CommitRespondingWS(_FakeWebSocket):
            async def send(self, payload):
                await super().send(payload)
                if json.loads(payload).get("type") == "input_audio_buffer.commit":
                    # Server acks the commit, then sends the final transcription.
                    self.push_server_json({"type": "input_audio_buffer.committed", "item_id": "item_final"})
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
            await _send_audio(session, _tone_ms(1000))  # an open turn with speech
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
            await _send_audio(session, _tone_ms(1000))
            await asyncio.wait_for(session.finalize(timeout=0.3), timeout=2.0)
            await session.aclose()

        _run(run())
        self.assertTrue(
            any(json.loads(s).get("type") == "input_audio_buffer.commit" for s in ws.sent)
        )

    def test_finalize_waits_for_completed_that_trails_committed_ack(self):
        """The production Stop-drop race: under server-VAD the manual commit
        gets a fast ``input_audio_buffer.committed`` ack, but the real
        ``.completed`` transcript trails it. The old finalize bailed on the ack
        (+0.4s settle) and tore the socket down before the transcript arrived.

        Here the completion trails by 0.6s — longer than that old settle window
        — so this fails on the old code and passes once finalize keys off the
        committed-vs-completed gap.
        """
        class _AckThenDelayedCompletionWS(_FakeWebSocket):
            async def send(self, payload):
                await super().send(payload)
                if json.loads(payload).get("type") == "input_audio_buffer.commit":
                    # Fast ack...
                    self.push_server_json({"type": "input_audio_buffer.committed", "item_id": "item_final"})

                    # ...then the real transcript, well after the old settle.
                    async def _delayed():
                        await asyncio.sleep(0.6)
                        self.push_server_json({
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": "item_final",
                            "transcript": "the trailing words.",
                        })
                    asyncio.create_task(_delayed())

        ws = _AckThenDelayedCompletionWS(script=[
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
            await asyncio.sleep(0.05)  # drain session.created
            await _send_audio(session, _tone_ms(1000))  # an open turn with speech
            await session.finalize(timeout=2.0)
            # finalize must have blocked until the trailing .completed landed;
            # only then do we close (which would otherwise cancel the recv loop).
            await session.aclose()
            await consumer
            return completed

        completed = _run(run())
        self.assertTrue(
            any(e.text == "the trailing words." for e in completed),
            "finalize() must wait for the .completed that trails the committed ack",
        )

    def test_finalize_skips_commit_when_turn_is_silent(self):
        # Committing silence only bills and invites a hallucinated tail.
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
            await _send_audio(session, _silence_ms(2000))
            started = asyncio.get_running_loop().time()
            await session.finalize(timeout=2.0)
            elapsed = asyncio.get_running_loop().time() - started
            await session.aclose()
            return elapsed

        elapsed = _run(run())
        self.assertNotIn("input_audio_buffer.commit", _sent_types(ws))
        self.assertLess(elapsed, 0.5)
