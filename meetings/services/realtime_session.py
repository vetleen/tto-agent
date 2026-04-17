"""Provider-agnostic streaming transcription session for the live path.

The existing chunked live-transcription path batches audio every ~30s and
calls ``/v1/audio/transcriptions``. Latency to first text on screen is
~15s in the common case. This module wraps OpenAI's Realtime API
(``"type": "transcription"`` session) so audio can stream in continuously
and text deltas stream back within ~1s of speech.

The abstract base ``RealtimeTranscriptionSession`` defines the interface
the WebSocket consumer talks to. The concrete
``OpenAIRealtimeSession`` is the only shipping implementation; the
abstraction exists so a future ``deepgram/*`` or ``assemblyai/*``
registry entry can plug in without touching the consumer or ffmpeg
pipeline. Factory ``build_realtime_session`` dispatches on the provider
declared in the transcription registry.

Event shape emitted to the consumer (kept provider-neutral so alternate
backends can conform):

* ``TranscriptDelta(item_id, text)`` — interim token slice.
* ``TranscriptCompleted(item_id, text, usage)`` — finalized utterance.
* ``SessionError(code, message, fatal)`` — recoverable or terminal.
* ``SessionStatus(state)`` — e.g. ``"connected"``, ``"reconnecting"``.

The OpenAI implementation handles reconnect with exponential backoff
and replays the last few seconds of PCM from a bounded ring buffer so
short outages don't drop audio.
"""
from __future__ import annotations

import abc
import asyncio
import base64
import logging
import random
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from llm.transcription_registry import get_transcription_model_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured events — shape used by consumers, independent of provider
# ---------------------------------------------------------------------------


@dataclass
class TranscriptDelta:
    item_id: str
    text: str


@dataclass
class TranscriptCompleted:
    item_id: str
    text: str
    usage: dict | None = None


@dataclass
class SessionError:
    code: str
    message: str
    fatal: bool = False


@dataclass
class SessionStatus:
    state: str  # "connected" | "reconnecting" | "disconnected"


SessionEvent = TranscriptDelta | TranscriptCompleted | SessionError | SessionStatus


class RealtimeSessionError(RuntimeError):
    """Raised when the session cannot be brought up or recovered."""


class UnsupportedModelError(RealtimeSessionError):
    """Raised when the selected model cannot drive a realtime session."""


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RealtimeTranscriptionSession(abc.ABC):
    """Bidirectional streaming transcription session (audio in → text out).

    Subclasses own one provider-specific WebSocket. The consumer interacts
    only with the methods on this base class.
    """

    def __init__(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        language: str | None = None,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.language = language
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=256)

    @abc.abstractmethod
    async def connect(self) -> None:
        """Open the session and prepare it to receive audio."""

    @abc.abstractmethod
    async def send_pcm(self, frame: bytes) -> None:
        """Forward a PCM16 frame (24kHz mono LE) to the provider."""

    @abc.abstractmethod
    async def finalize(self, timeout: float = 3.0) -> None:
        """Flush pending audio and wait briefly for final ``.completed`` events."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release all resources. Safe to call multiple times."""

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Yield structured events produced by the provider.

        Terminates when the session is closed. Consumers treat
        ``SessionError(fatal=True)`` as a signal to tear down.
        """
        while True:
            evt = await self._events.get()
            yield evt
            if isinstance(evt, SessionStatus) and evt.state == "disconnected":
                return


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


# Retained audio for replay after reconnect. At 24kHz mono PCM16 that's
# 24000 * 2 = 48_000 bytes/sec. 5s = 240 KB — bounded and small.
_REPLAY_BUFFER_SECONDS = 5
_REPLAY_BUFFER_BYTES = 24_000 * 2 * _REPLAY_BUFFER_SECONDS

# Reconnect schedule: cap at 4s so we recover quickly, max 8 attempts per
# meeting before giving up. The counter resets after 60s of success so a
# single reconnect doesn't permanently exhaust the budget.
_RECONNECT_BACKOFFS = (0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0)
_RECONNECT_COUNTER_RESET_SECONDS = 60


# Process-wide AsyncOpenAI client. Lazy so importing this module in test
# contexts without the API key doesn't blow up.
_client_lock = threading.Lock()
_async_client = None


def _get_async_client():
    global _async_client
    if _async_client is None:
        with _client_lock:
            if _async_client is None:
                from openai import AsyncOpenAI
                _async_client = AsyncOpenAI()
    return _async_client


class OpenAIRealtimeSession(RealtimeTranscriptionSession):
    """Wrap an ``AsyncOpenAI().beta.realtime.connect(...)`` connection.

    The SDK (openai==2.24.0) returns an ``AsyncRealtimeConnection`` with
    typed helpers for the client events we need: ``transcription_session
    .update`` to configure the session, ``input_audio_buffer.append`` to
    stream PCM, and async iteration over incoming server events.
    """

    def __init__(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        language: str | None = None,
        _client_factory=None,  # test seam
    ):
        super().__init__(model_id=model_id, prompt=prompt, language=language)
        info = get_transcription_model_info(model_id)
        if info is None:
            raise UnsupportedModelError(f"Unknown transcription model: {model_id}")
        if not info.supports_live_streaming:
            raise UnsupportedModelError(
                f"Model {model_id} does not support live streaming"
            )
        self._info = info
        self._api_model = info.api_model
        self._conn = None
        self._cm = None
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False
        self._session_id: str | None = None
        self._replay_buffer: deque[bytes] = deque()
        self._replay_bytes = 0
        self._reconnect_attempt = 0
        self._last_reconnect_success: float = 0.0
        self._client_factory = _client_factory or _get_async_client

    async def connect(self) -> None:
        await self._open()
        await self._events.put(SessionStatus(state="connected"))

    async def _open(self) -> None:
        client = self._client_factory()
        self._cm = client.beta.realtime.connect(model=self._api_model)
        self._conn = await self._cm.__aenter__()
        await self._send_session_update()
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def _send_session_update(self) -> None:
        """Configure the session for transcription-only.

        The SDK exposes ``transcription_session.update`` which accepts the
        same shape as ``/v1/realtime/transcription_sessions``. We ask for
        server VAD (so the model auto-commits on silence) and near-field
        noise reduction. Prompt and language are optional.
        """
        assert self._conn is not None
        transcription_cfg: dict = {"model": self._api_model}
        if self.prompt:
            transcription_cfg["prompt"] = self.prompt[:2048]
        if self.language:
            transcription_cfg["language"] = self.language
        session_cfg: dict = {
            "input_audio_format": "pcm16",
            "input_audio_transcription": transcription_cfg,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 600,
            },
            "input_audio_noise_reduction": {"type": "near_field"},
        }
        await self._conn.send({
            "type": "transcription_session.update",
            "session": session_cfg,
        })

    async def _receive_loop(self) -> None:
        """Drain server events and translate into provider-neutral events.

        Survives benign disconnects by scheduling a reconnect.
        """
        assert self._conn is not None
        try:
            async for event in self._conn:
                evt_type = getattr(event, "type", "")
                await self._dispatch_server_event(evt_type, event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OpenAIRealtimeSession: recv loop ended (%s)", exc)
            if not self._closed:
                asyncio.create_task(self._try_reconnect())

    async def _dispatch_server_event(self, evt_type: str, event) -> None:
        if evt_type in ("transcription_session.created", "transcription_session.updated"):
            sess = getattr(event, "session", None)
            if sess is not None:
                self._session_id = getattr(sess, "id", None)
            return
        if evt_type == "conversation.item.input_audio_transcription.delta":
            item_id = getattr(event, "item_id", "") or ""
            delta = getattr(event, "delta", "") or ""
            if delta:
                await self._events.put(TranscriptDelta(item_id=item_id, text=delta))
            return
        if evt_type == "conversation.item.input_audio_transcription.completed":
            item_id = getattr(event, "item_id", "") or ""
            text = getattr(event, "transcript", "") or ""
            usage = None
            usage_obj = getattr(event, "usage", None)
            if usage_obj is not None:
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
            await self._events.put(TranscriptCompleted(item_id=item_id, text=text, usage=usage))
            return
        if evt_type == "error":
            err = getattr(event, "error", None)
            code = getattr(err, "code", "") if err else ""
            message = getattr(err, "message", "") if err else ""
            # Authentication / model errors are fatal; others are recoverable.
            fatal = code in {"invalid_api_key", "model_not_found", "permission_denied"}
            await self._events.put(SessionError(code=code or "unknown", message=message, fatal=fatal))
            return
        # Other event types (rate_limits.updated etc.) are ignored for now.

    async def _try_reconnect(self) -> None:
        if self._closed:
            return
        # Reset counter if the previous session lasted long enough to be healthy.
        if time.monotonic() - self._last_reconnect_success > _RECONNECT_COUNTER_RESET_SECONDS:
            self._reconnect_attempt = 0
        if self._reconnect_attempt >= len(_RECONNECT_BACKOFFS):
            logger.error("OpenAIRealtimeSession: reconnect budget exhausted; giving up")
            await self._events.put(SessionError(code="reconnect_exhausted", message="Max reconnect attempts reached", fatal=True))
            await self._events.put(SessionStatus(state="disconnected"))
            return

        backoff = _RECONNECT_BACKOFFS[self._reconnect_attempt] + random.uniform(0, 0.25)
        self._reconnect_attempt += 1
        logger.warning("OpenAIRealtimeSession: reconnecting in %.1fs (attempt %d)", backoff, self._reconnect_attempt)
        await self._events.put(SessionStatus(state="reconnecting"))
        await asyncio.sleep(backoff)

        # Tear down the old handles so _open can re-enter cleanly.
        try:
            if self._cm is not None:
                await self._cm.__aexit__(None, None, None)
        except Exception:
            pass
        self._conn = None
        self._cm = None

        try:
            await self._open()
        except Exception as exc:
            logger.warning("OpenAIRealtimeSession: reconnect failed (%s); scheduling retry", exc)
            asyncio.create_task(self._try_reconnect())
            return

        # Replay the last few seconds of PCM so the remote side doesn't lose
        # audio that was in flight around the disconnect. Duplication at
        # utterance boundaries is acceptable — the VAD will coalesce.
        for frame in list(self._replay_buffer):
            try:
                await self._conn.input_audio_buffer.append(
                    audio=base64.b64encode(frame).decode("ascii")
                )
            except Exception:
                break

        self._last_reconnect_success = time.monotonic()
        await self._events.put(SessionStatus(state="connected"))

    async def send_pcm(self, frame: bytes) -> None:
        if self._closed:
            return
        self._remember_pcm(frame)
        if self._conn is None:
            return  # mid-reconnect; the ring buffer will replay once we're back.
        try:
            await self._conn.input_audio_buffer.append(
                audio=base64.b64encode(frame).decode("ascii")
            )
        except Exception as exc:
            logger.warning("OpenAIRealtimeSession: append failed (%s) — dropping and triggering reconnect", exc)
            # The recv loop will observe the close and kick off reconnect.

    def _remember_pcm(self, frame: bytes) -> None:
        self._replay_buffer.append(frame)
        self._replay_bytes += len(frame)
        while self._replay_bytes > _REPLAY_BUFFER_BYTES and self._replay_buffer:
            old = self._replay_buffer.popleft()
            self._replay_bytes -= len(old)

    async def finalize(self, timeout: float = 3.0) -> None:
        """Commit any pending audio and wait briefly for ``.completed`` events."""
        if self._closed or self._conn is None:
            return
        try:
            await self._conn.input_audio_buffer.commit()
        except Exception as exc:
            logger.debug("OpenAIRealtimeSession: commit raised (%s) — continuing", exc)
        # Let the server produce any last completed event — we don't wait
        # on specific events, we just give the recv loop a moment to drain.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._events.qsize() > 0:
            await asyncio.sleep(0.05)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        try:
            if self._conn is not None:
                await self._conn.close()
        except Exception:
            pass
        try:
            if self._cm is not None:
                await self._cm.__aexit__(None, None, None)
        except Exception:
            pass
        await self._events.put(SessionStatus(state="disconnected"))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_realtime_session(
    *,
    model_id: str,
    prompt: str | None = None,
    language: str | None = None,
) -> RealtimeTranscriptionSession:
    """Return a provider-appropriate ``RealtimeTranscriptionSession``.

    The consumer calls this with whatever model the user picked. We look
    up the provider from the registry so adding a Deepgram or
    AssemblyAI backend later is a one-line branch.
    """
    info = get_transcription_model_info(model_id)
    if info is None:
        raise UnsupportedModelError(f"Unknown transcription model: {model_id}")
    if not info.supports_live_streaming:
        raise UnsupportedModelError(f"Model {model_id} does not support live streaming")
    if info.provider == "openai":
        return OpenAIRealtimeSession(model_id=model_id, prompt=prompt, language=language)
    raise UnsupportedModelError(f"No realtime session implementation for provider {info.provider!r}")


__all__ = [
    "RealtimeTranscriptionSession",
    "OpenAIRealtimeSession",
    "build_realtime_session",
    "TranscriptDelta",
    "TranscriptCompleted",
    "SessionError",
    "SessionStatus",
    "SessionEvent",
    "RealtimeSessionError",
    "UnsupportedModelError",
]
