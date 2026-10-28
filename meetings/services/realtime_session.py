"""Provider-agnostic streaming transcription session for the live path.

The existing chunked live-transcription path batches audio every ~30s and
calls ``/v1/audio/transcriptions``. Latency to first text on screen is
~15s in the common case. This module wraps OpenAI's Realtime API
(``"type": "transcription"`` session) so audio can stream in continuously
and text deltas stream back within ~1s of speech.

``OpenAIRealtimeSession`` is the only shipping implementation and is what
the WebSocket consumer talks to. Factory ``build_realtime_session``
dispatches on the provider declared in the transcription registry, so a
future ``deepgram/*`` or ``assemblyai/*`` backend can plug in there
without touching the consumer or ffmpeg pipeline.

Event shape emitted to the consumer (kept provider-neutral so alternate
backends can conform):

* ``TranscriptDelta(item_id, text)`` — interim token slice.
* ``TranscriptCompleted(item_id, text, usage)`` — finalized utterance.
* ``SessionError(code, message, fatal)`` — recoverable or terminal.
* ``SessionStatus(state)`` — e.g. ``"connected"``, ``"reconnecting"``.
* ``SessionActivity(kind)`` — turn liveness (``"speech_started"``,
  ``"speech_stopped"``, ``"committed"``); lets the client tell an open
  utterance apart from a stalled pipeline.

Turn boundaries are decided here, not by the provider: ``TurnDetector``
(``meetings/services/turn_detector.py``) watches the PCM we forward and the
session sends ``input_audio_buffer.commit`` at the end of each turn, because
the realtime transcription models don't support server VAD. Audio sent since
the last commit is kept so a reconnect can replay it without losing or
duplicating a turn.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from llm.transcription_registry import get_transcription_model_info
from meetings.services.turn_detector import (
    PCM_BYTES_PER_SECOND,
    TurnDecision,
    TurnDetector,
    config_from_settings,
)

logger = logging.getLogger(__name__)


@dataclass
class _CommitMeta:
    """What our side knows about one commit, keyed by item_id once acked."""
    start_bytes: int
    turn_bytes: int
    # Commit order on the current socket (assigned when the commit is sent).
    seq: int = 0


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
    # Seconds of audio that preceded this utterance's start (not its end).
    # ``None`` if the provider couldn't determine it; the consumer falls back.
    start_offset_seconds: float | None = None
    # Seconds of audio the commit covered (what the provider transcribed).
    duration_seconds: float | None = None


@dataclass
class SessionError:
    code: str
    message: str
    fatal: bool = False


@dataclass
class SessionStatus:
    state: str  # "connected" | "reconnecting" | "disconnected"


@dataclass
class SessionActivity:
    """Turn lifecycle, forwarded as a liveness signal.

    ``speech_started`` / ``speech_stopped`` come from our own turn detector
    (they prove the ffmpeg pipe hears audio); ``committed`` is the provider's
    ack of a commit (it proves the upstream socket is alive). Three per
    utterance, so the client can tell "still listening" apart from "stalled".
    """
    kind: str  # "speech_started" | "speech_stopped" | "committed"


SessionEvent = (
    TranscriptDelta | TranscriptCompleted | SessionError | SessionStatus | SessionActivity
)


# Server "error" events that are expected and harmless. A commit of a buffer
# under 100 ms (e.g. a turn whose audio was all replayed into a socket that
# then dropped) surfaces as an ``invalid_request_error`` but is not a failure —
# swallowing it keeps a benign race from tearing down the realtime session.
_BENIGN_ERROR_CODES = frozenset({"input_audio_buffer_commit_empty"})


class RealtimeSessionError(RuntimeError):
    """Raised when the session cannot be brought up or recovered."""


class UnsupportedModelError(RealtimeSessionError):
    """Raised when the selected model cannot drive a realtime session."""


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


# Uncommitted audio kept for replay after a reconnect. A turn is at most the
# detector's hard cap (~20 s) and idle audio is cleared after ~15 s, so this
# only fills up during a long outage; then the oldest audio is dropped.
# At 24kHz mono PCM16 that's 48_000 bytes/sec: 60 s ≈ 2.9 MB.
_UNCOMMITTED_BUFFER_SECONDS = 60
_UNCOMMITTED_BUFFER_BYTES = PCM_BYTES_PER_SECOND * _UNCOMMITTED_BUFFER_SECONDS

# Reconnect schedule: cap at 4s so we recover quickly, max 8 attempts per
# meeting before giving up. The counter resets after 60s of success so a
# single reconnect doesn't permanently exhaust the budget.
_RECONNECT_BACKOFFS = (0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0)
_RECONNECT_COUNTER_RESET_SECONDS = 60

# How long a completion that arrived ahead of an earlier, still-outstanding
# turn is held back so the transcript stays in spoken order.
REORDER_WAIT_SECONDS = 8.0


_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


def _echoed_transcription_language(session: dict) -> str | None:
    """Pull ``session.audio.input.transcription.language`` out of a server echo."""
    audio = session.get("audio")
    if not isinstance(audio, dict):
        return None
    audio_input = audio.get("input")
    if not isinstance(audio_input, dict):
        return None
    transcription = audio_input.get("transcription")
    if not isinstance(transcription, dict):
        return None
    language = transcription.get("language")
    return language if isinstance(language, str) and language else None


async def _default_ws_connect(api_key: str, url: str):
    """Open a raw WebSocket to OpenAI's Realtime transcription endpoint.

    We talk to ``/v1/realtime?intent=transcription`` directly instead of
    going through the SDK wrapper, which forces a ``?model=`` URL parameter
    that the transcription path rejects with
    ``invalid_request_error.invalid_model``. The transcription model is
    configured in the ``session.update`` event below instead.

    GA note: the ``OpenAI-Beta: realtime=v1`` header is intentionally absent.
    OpenAI retired the beta Realtime dialect — sending that header together
    with the old ``transcription_session.update`` message now closes the
    socket with ``invalid_request_error.beta_api_shape_disabled``.
    """
    from websockets.asyncio.client import connect

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    return await connect(
        url,
        additional_headers=headers,
        max_size=2**24,  # 16MB — plenty of headroom for delta events
    )


class OpenAIRealtimeSession:
    """Open a direct WebSocket to OpenAI's Realtime transcription endpoint.

    Connects to ``wss://api.openai.com/v1/realtime?intent=transcription``
    using the process-wide ``OPENAI_API_KEY``. The transcription model
    (``gpt-4o-mini-transcribe`` / ``gpt-4o-transcribe``) is sent inside
    the GA ``session.update`` event, not in the URL.

    The ``_ws_connect_factory`` kwarg is a test seam that returns an
    awaitable which yields any websocket-like object. The interface used
    is the subset of the ``websockets`` library: ``send(data)``,
    async iteration over incoming messages, ``close(code)``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        language: str | None = None,
        _ws_connect_factory=None,   # test seam; replaces _default_ws_connect
        _api_key: str | None = None,
    ):
        self.model_id = model_id
        self.prompt = prompt
        self.language = language
        self._events: asyncio.Queue[SessionEvent] = asyncio.Queue(maxsize=256)
        info = get_transcription_model_info(model_id)
        if info is None:
            raise UnsupportedModelError(f"Unknown transcription model: {model_id}")
        if not info.supports_live_streaming:
            raise UnsupportedModelError(
                f"Model {model_id} does not support live streaming"
            )
        self._info = info
        self._api_model = info.api_model
        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        # Reconnect runs as a fire-and-forget task; we hold a reference so aclose
        # can cancel a reconnect that is sleeping in backoff (otherwise it would
        # wake after aclose and reopen a fresh, leaked, billable session).
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closed = False
        self._session_id: str | None = None
        # Turn detection: decides where each utterance ends (see module doc).
        self._turn = TurnDetector(config=config_from_settings())
        # Audio sent since the last commit/clear, replayed on reconnect. While
        # the socket is down, commits the detector decides are recorded as
        # byte boundaries in this buffer and sent during the replay.
        self._uncommitted: deque[bytes] = deque()
        self._uncommitted_bytes = 0
        self._offline_commits: deque[tuple[int, _CommitMeta]] = deque()
        # Set from just before a reconnect's _open() until its replay has
        # caught up; send_pcm only buffers meanwhile.
        self._replaying = False
        # _pcm_bytes_sent is a monotonic count of audio forwarded — the meeting
        # timeline that start offsets are measured on.
        self._pcm_bytes_sent = 0
        # Commits sent but not yet acked (FIFO: acks arrive in commit order),
        # then keyed by the item_id the ack names. Completions are looked up
        # by item_id, so they pair correctly even when they arrive out of order.
        self._pending_commits: deque[_CommitMeta] = deque()
        self._items: dict[str, _CommitMeta] = {}
        # In-order emission (see _emit_in_order): commits are numbered from 1
        # as they are sent; completions ahead of an outstanding turn wait here.
        self._commit_seq = 0
        self._next_emit_seq = 1
        self._held: dict[int, TranscriptCompleted | None] = {}
        self._reorder_task: Optional[asyncio.Task] = None
        self._warned_server_vad = False
        # Utterance bookkeeping for a clean Stop: every committed buffer should
        # produce one transcription. finalize() waits until completed catches up
        # to committed so the trailing utterance isn't dropped when the user
        # stops mid-transcription.
        self._committed_count = 0
        self._completed_count = 0
        self._reconnect_attempt = 0
        self._last_reconnect_success: float = 0.0
        # Set by update_language() when the only way to apply a config change
        # is a fresh session (clearing a pinned language). The receive loop
        # then treats the self-inflicted close as routine, not a server drop.
        self._reopen_requested = False
        self._ws_connect_factory = _ws_connect_factory or _default_ws_connect
        self._api_key = _api_key or os.environ.get("OPENAI_API_KEY", "")

    async def connect(self) -> None:
        await self._open()
        await self._events.put(SessionStatus(state="connected"))

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

    def pending_event_count(self) -> int:
        """Events produced but not yet consumed.

        Teardown uses this to give the event-consumer task a bounded moment to
        dequeue and persist the trailing ``.completed`` that ``finalize()``
        waited for, before it gets cancelled.
        """
        return self._events.qsize()

    async def _open(self) -> None:
        if self._closed:
            # A reconnect scheduled before aclose() must not resurrect the
            # session after it's been closed.
            raise RealtimeSessionError("session is closed")
        if not self._api_key:
            raise RealtimeSessionError("OPENAI_API_KEY is not set")
        # Commits orphaned by the dropped socket can never produce a .completed
        # on the new one. Reset the pairing so finalize()'s committed-vs-completed
        # wait doesn't stall on ghosts and burn its full timeout on every Stop
        # after a reconnect. The byte counter keeps running to preserve a
        # continuous meeting timeline.
        lost = len(self._pending_commits) + max(0, self._committed_count - self._completed_count)
        if lost:
            logger.warning(
                "OpenAIRealtimeSession: %d committed turn(s) lost with the dropped socket", lost,
            )
        self._pending_commits.clear()
        self._items.clear()
        self._committed_count = 0
        self._completed_count = 0
        # Lost turns will never complete: release what was waiting on them and
        # number the new socket's commits after everything sent so far.
        await self._flush_held()
        self._next_emit_seq = self._commit_seq + 1
        ws = await self._ws_connect_factory(self._api_key, _REALTIME_WS_URL)
        self._ws = ws
        await self._send_session_update()
        # Bind the loop to THIS socket: it must never pick up (or clear) a
        # socket that a later reopen/reconnect installed in self._ws.
        self._recv_task = asyncio.create_task(self._receive_loop(ws))

    async def _send_session_update(self) -> None:
        """Configure the session for transcription-only (GA Realtime shape).

        Tells OpenAI which transcription model to use and what audio format
        we'll send (PCM16 @ 24kHz). ``turn_detection`` is an explicit null:
        we commit each turn ourselves (the session defaults to server VAD
        otherwise, which the live transcription models reject).

        GA nests everything under ``session.audio.input`` and expects the
        audio format as an object (``{"type": "audio/pcm", "rate": 24000}``)
        rather than the old ``"pcm16"`` string. The message ``type`` is the
        generic ``session.update`` (the beta ``transcription_session.update``
        is gone), with ``session.type == "transcription"``.
        """
        assert self._ws is not None
        transcription_cfg: dict = {"model": self._api_model}
        if self.prompt:
            # OpenAI Realtime transcription prompts are capped at 1024 chars
            # (shorter than the batch transcription endpoint's 2048 limit).
            # Longer prompts close the session with string_above_max_length.
            transcription_cfg["prompt"] = self.prompt[:1024]
        if self.language:
            transcription_cfg["language"] = self.language
        audio_input_cfg: dict = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": transcription_cfg,
            "turn_detection": None,
            # far_field: meetings are typically captured by a laptop or
            # conference-room mic at a distance, not a close-talking headset
            # (which is what near_field is tuned for per OpenAI's docs).
            "noise_reduction": {"type": "far_field"},
        }
        payload = json.dumps({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": audio_input_cfg},
            },
        })
        await self._ws.send(payload)

    async def _receive_loop(self, ws) -> None:
        """Drain server events from ``ws`` and translate into provider-neutral events.

        Survives benign disconnects by scheduling a reconnect.
        """
        try:
            async for message in ws:
                # websockets lib delivers strings for text frames; be defensive.
                if isinstance(message, bytes):
                    message = message.decode("utf-8", errors="replace")
                try:
                    event = json.loads(message)
                except (json.JSONDecodeError, ValueError):
                    logger.warning("OpenAIRealtimeSession: non-JSON frame (%d bytes) — dropping", len(message))
                    continue
                await self._dispatch_server_event(event.get("type", ""), event)
            # Normal iteration end = the server closed the socket cleanly
            # (e.g. session lifetime reached). Reconnect just like an abnormal
            # drop — otherwise this is a silent stall: audio keeps flowing into
            # a dead session and no error ever surfaces.
            if not self._closed:
                if self._reopen_requested:
                    self._reopen_requested = False
                    logger.info("OpenAIRealtimeSession: reopening to apply config")
                else:
                    logger.warning("OpenAIRealtimeSession: server closed the socket; reconnecting")
                await self._drop_dead_socket(ws)
                self._reconnect_task = asyncio.create_task(self._try_reconnect())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OpenAIRealtimeSession: recv loop ended (%s)", exc)
            if not self._closed:
                await self._drop_dead_socket(ws)
                self._reconnect_task = asyncio.create_task(self._try_reconnect())

    async def _drop_dead_socket(self, ws) -> None:
        """Forget a socket whose receive loop has ended, before the reconnect backoff.

        While ``_ws`` still points at the dead socket, every 250 ms PCM frame
        the pump forwards during the backoff fails with its own WARNING (a
        storm of "append failed" lines, each a Sentry event). With it cleared,
        ``send_pcm`` buffers silently and the ring buffer replays on reconnect.
        Only clears ``_ws`` if it still is this socket, so a loop ending late
        can never discard a successor that a reopen already installed.
        """
        if self._ws is ws:
            self._ws = None
        try:
            await ws.close()
        except Exception:
            pass

    async def _dispatch_server_event(self, evt_type: str, event: dict) -> None:
        if evt_type in ("session.created", "session.updated"):
            session = event.get("session") or {}
            echoed_language = None
            if isinstance(session, dict):
                self._session_id = session.get("id")
                echoed_language = _echoed_transcription_language(session)
            # The echoed language is what the server will actually transcribe
            # in — the one place a live language switch can be verified.
            logger.info(
                "OpenAIRealtimeSession: %s (id=%s language=%s)",
                evt_type, self._session_id, echoed_language or "auto",
            )
            return
        if evt_type == "input_audio_buffer.committed":
            # Acks arrive in commit order, so the oldest unacked commit is the
            # one this ack names. One committed buffer ⇒ one transcription
            # owed; finalize() waits for completed to catch up.
            item_id = event.get("item_id", "") or ""
            meta = self._pending_commits.popleft() if self._pending_commits else None
            if meta is not None and item_id:
                self._items[item_id] = meta
            self._committed_count += 1
            logger.info("OpenAIRealtimeSession: committed (item=%s)", item_id or "-")
            # Forward as liveness: proves the upstream socket answers.
            await self._events.put(SessionActivity(kind="committed"))
            return
        if evt_type in ("input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"):
            # Server VAD is off (turn_detection null); our detector owns turns.
            if not self._warned_server_vad:
                self._warned_server_vad = True
                logger.warning("OpenAIRealtimeSession: unexpected server VAD event %s — ignored", evt_type)
            return
        if evt_type == "conversation.item.input_audio_transcription.delta":
            item_id = event.get("item_id", "") or ""
            delta = event.get("delta", "") or ""
            if delta:
                await self._events.put(TranscriptDelta(item_id=item_id, text=delta))
            return
        if evt_type == "conversation.item.input_audio_transcription.completed":
            item_id = event.get("item_id", "") or ""
            text = event.get("transcript", "") or ""
            usage_obj = event.get("usage") or {}
            usage = None
            if isinstance(usage_obj, dict) and usage_obj:
                usage = {
                    "type": usage_obj.get("type"),
                    "seconds": usage_obj.get("seconds"),
                    "input_tokens": usage_obj.get("input_tokens"),
                    "output_tokens": usage_obj.get("output_tokens"),
                    "total_tokens": usage_obj.get("total_tokens"),
                }
            # Pair by item_id with what we knew at commit time; fall back to
            # the current position if the ack was missed.
            meta = self._items.pop(item_id, None)
            if meta is None:
                logger.info("OpenAIRealtimeSession: completion for unknown item %s", item_id or "-")
            start_bytes = meta.start_bytes if meta else self._pcm_bytes_sent
            completed = TranscriptCompleted(
                item_id=item_id,
                text=text,
                usage=usage,
                start_offset_seconds=start_bytes / PCM_BYTES_PER_SECOND,
                duration_seconds=(meta.turn_bytes / PCM_BYTES_PER_SECOND) if meta else None,
            )
            self._completed_count += 1
            await self._emit_in_order(meta.seq if meta else None, completed)
            return
        if evt_type == "conversation.item.input_audio_transcription.failed":
            # A committed buffer whose transcription failed server-side. Without
            # this branch the event fell into the ignored catch-all: the
            # utterance was silently lost and _completed_count never caught up
            # to _committed_count, so every later finalize() waited out its
            # full timeout.
            err = event.get("error") or {}
            message = (err.get("message") or "") if isinstance(err, dict) else ""
            logger.warning(
                "OpenAIRealtimeSession: utterance transcription failed (item=%s): %s",
                event.get("item_id", "") or "", message,
            )
            # The commit will never complete — advance the completed counter so
            # finalize() doesn't stall, and let later turns past it in order.
            self._completed_count += 1
            failed_meta = self._items.pop(event.get("item_id", "") or "", None)
            if failed_meta is not None:
                await self._emit_in_order(failed_meta.seq, None)
            await self._events.put(SessionError(
                code="utterance_failed",
                message=message or "One utterance could not be transcribed.",
                fatal=False,
            ))
            return
        if evt_type == "error":
            err = event.get("error") or {}
            code = (err.get("code") or "") if isinstance(err, dict) else ""
            err_type = (err.get("type") or "") if isinstance(err, dict) else ""
            message = (err.get("message") or "") if isinstance(err, dict) else ""
            # Benign, expected races (e.g. committing an already-auto-committed
            # buffer under server-VAD) are not failures — log and move on so
            # they don't surface as a SessionError that tears down realtime.
            if code in _BENIGN_ERROR_CODES:
                logger.info("OpenAIRealtimeSession: benign server error %s (%s)", code, message)
                return
            # Fatal vs recoverable. A genuine request/config rejection
            # (``error.type == "invalid_request_error"`` — e.g. a bad
            # session.update or an unsupported model) will be rejected
            # identically on reconnect, so treat it as fatal rather than
            # looping the backoff schedule into the same rejection. Auth and
            # model codes are likewise fatal.
            fatal = (
                err_type == "invalid_request_error"
                or code in {"invalid_api_key", "model_not_found", "permission_denied"}
            )
            await self._events.put(
                SessionError(code=code or err_type or "unknown", message=message, fatal=fatal)
            )
            if fatal:
                # A fatal error ends the session. Emit a disconnected status so
                # the base events() generator self-terminates instead of relying
                # solely on the consumer returning after the SessionError.
                await self._events.put(SessionStatus(state="disconnected"))
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

        if self._closed:
            # aclose() ran during the backoff sleep — do not reopen a socket
            # that nothing will ever close.
            return

        # Tear down the old socket so _open can re-enter cleanly.
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None

        self._replaying = True
        try:
            await self._open()
        except Exception as exc:
            self._replaying = False
            logger.warning("OpenAIRealtimeSession: reconnect failed (%s); scheduling retry", exc)
            self._reconnect_task = asyncio.create_task(self._try_reconnect())
            return

        await self._replay_uncommitted()

        self._last_reconnect_success = time.monotonic()
        await self._events.put(SessionStatus(state="connected"))

    async def _replay_uncommitted(self) -> None:
        """Send the audio no commit has covered yet to the fresh socket.

        Everything since the last successful commit is replayed exactly once,
        so nothing is lost or transcribed twice. Commits the detector decided
        while the socket was down are sent at their byte boundaries, so turns
        stay separate. ``send_pcm`` keeps buffering while ``_replaying`` is
        set, and this loop drains until it has caught up, so live frames can't
        overtake the replay.
        """
        replayed = 0
        idx = 0
        pos = 0  # bytes of the buffer's head already sent on this socket
        try:
            while idx < len(self._uncommitted):
                frame = self._uncommitted[idx]
                await self._send_audio_frame(frame)
                idx += 1
                pos += len(frame)
                replayed += len(frame)
                while self._offline_commits and self._offline_commits[0][0] <= pos:
                    _, meta = self._offline_commits.popleft()
                    await self._send_commit(meta)
                    # The committed head no longer needs replaying.
                    for _ in range(idx):
                        self._uncommitted.popleft()
                    self._uncommitted_bytes -= pos
                    self._offline_commits = deque(
                        (p - pos, m) for p, m in self._offline_commits
                    )
                    idx = 0
                    pos = 0
        except Exception as exc:
            # The new socket died too; its receive loop schedules the next
            # reconnect, which replays from the same buffer.
            logger.warning("OpenAIRealtimeSession: replay interrupted (%s)", exc)
        finally:
            self._replaying = False
        if replayed:
            logger.info(
                "OpenAIRealtimeSession: replayed %.1fs of uncommitted audio",
                replayed / PCM_BYTES_PER_SECOND,
            )

    async def _send_audio_frame(self, frame: bytes) -> None:
        """Send a base64-encoded PCM frame as an ``input_audio_buffer.append`` event."""
        assert self._ws is not None
        payload = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(frame).decode("ascii"),
        })
        await self._ws.send(payload)

    async def send_pcm(self, frame: bytes) -> None:
        if self._closed:
            return
        self._pcm_bytes_sent += len(frame)
        self._keep_uncommitted(frame)
        decision = self._turn.feed(frame)
        if decision.speech_started:
            await self._events.put(SessionActivity(kind="speech_started"))
        if decision.speech_stopped:
            await self._events.put(SessionActivity(kind="speech_stopped"))
        await self._apply_decision(decision)

    async def _apply_decision(self, decision: TurnDecision) -> None:
        """Send the newest buffered frame and act on the detector's decision."""
        if self._ws is None or self._replaying:
            # Mid-reconnect: the frame is buffered and replayed once we're
            # back. A turn that ends now is committed during the replay.
            if decision.commit:
                self._offline_commits.append((self._uncommitted_bytes, self._commit_meta(decision)))
            elif decision.clear and not self._offline_commits and not self._replaying:
                self._drop_uncommitted()
            return
        frame = self._uncommitted[-1] if self._uncommitted else b""
        try:
            if frame:
                await self._send_audio_frame(frame)
            if decision.commit:
                await self._send_commit(self._commit_meta(decision))
                self._drop_uncommitted()
            elif decision.clear:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                self._drop_uncommitted()
        except Exception as exc:
            # The recv loop will observe the close and kick off reconnect,
            # which replays the buffer (and this commit, if one was due).
            logger.warning("OpenAIRealtimeSession: send failed (%s) — will replay after reconnect", exc)
            if decision.commit:
                self._offline_commits.append((self._uncommitted_bytes, self._commit_meta(decision)))

    def _keep_uncommitted(self, frame: bytes) -> None:
        self._uncommitted.append(frame)
        self._uncommitted_bytes += len(frame)
        if self._replaying:
            return  # the replay indexes into the buffer; trim after it.
        while self._uncommitted_bytes > _UNCOMMITTED_BUFFER_BYTES and len(self._uncommitted) > 1:
            old = self._uncommitted.popleft()
            self._uncommitted_bytes -= len(old)
            # Boundaries are offsets into the buffer; shift them with it.
            self._offline_commits = deque(
                (pos - len(old), meta) for pos, meta in self._offline_commits if pos - len(old) > 0
            )

    def _drop_uncommitted(self) -> None:
        self._uncommitted.clear()
        self._uncommitted_bytes = 0

    @staticmethod
    def _commit_meta(decision: TurnDecision) -> _CommitMeta:
        return _CommitMeta(
            start_bytes=decision.turn_start_bytes or 0,
            turn_bytes=decision.turn_bytes,
        )

    async def _send_commit(self, meta: _CommitMeta) -> None:
        """Send ``input_audio_buffer.commit`` and number it for in-order emission."""
        assert self._ws is not None
        self._commit_seq += 1
        meta.seq = self._commit_seq
        self._pending_commits.append(meta)
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception:
            self._pending_commits.pop()
            self._commit_seq -= 1
            raise

    async def _emit_in_order(self, seq: int | None, evt: TranscriptCompleted | None) -> None:
        """Emit completions in commit order; ``evt=None`` marks a failed turn.

        Completions can arrive out of order. One that is ahead of an
        outstanding earlier turn is held until that turn completes or fails,
        or until ``REORDER_WAIT_SECONDS`` passes — then everything held is
        released in order and the missing turn, if it ever arrives, is emitted
        on its own. Unnumbered completions (ack missed) go straight out.
        """
        if seq is None or seq < self._next_emit_seq:
            if evt is not None:
                await self._events.put(evt)
            return
        self._held[seq] = evt
        await self._release_held()
        if self._held and self._reorder_task is None:
            self._reorder_task = asyncio.create_task(self._reorder_timeout())

    async def _release_held(self) -> None:
        while self._next_emit_seq in self._held:
            evt = self._held.pop(self._next_emit_seq)
            self._next_emit_seq += 1
            if evt is not None:
                await self._events.put(evt)
        if not self._held and self._reorder_task is not None:
            self._reorder_task.cancel()
            self._reorder_task = None

    async def _reorder_timeout(self) -> None:
        try:
            await asyncio.sleep(REORDER_WAIT_SECONDS)
        except asyncio.CancelledError:
            return
        self._reorder_task = None
        if self._held:
            logger.warning(
                "OpenAIRealtimeSession: turn %d still untranscribed after %.0fs; releasing %d later turn(s)",
                self._next_emit_seq, REORDER_WAIT_SECONDS, len(self._held),
            )
            await self._flush_held()

    async def _flush_held(self) -> None:
        """Release every held completion in commit order, skipping gaps."""
        for seq in sorted(self._held):
            evt = self._held.pop(seq)
            self._next_emit_seq = max(self._next_emit_seq, seq + 1)
            if evt is not None:
                await self._events.put(evt)

    async def update_language(self, language: str | None) -> None:
        """Apply a new transcription language to the running session.

        ``language`` is an ISO code or ``None`` for auto-detect. A concrete
        code is pushed with a fresh ``session.update`` (the GA API accepts
        config updates at any time; the model is unchanged). Auto-detect
        instead reopens the socket: ``session.update`` merges, so re-sending
        the config *without* a ``language`` key would not reliably clear a
        pinned one, and sending an empty value risks an
        ``invalid_request_error`` (classified fatal). The reconnect path
        replays the uncommitted audio, so nothing is lost.

        Mid-reconnect (``_ws is None``) the new value is simply stored:
        ``_open()`` sends it with the next ``session.update``.
        """
        self.language = language or None
        if self._closed or self._ws is None:
            return
        if self.language:
            await self._send_session_update()
            return
        await self._reopen_for_config()

    async def _reopen_for_config(self) -> None:
        """Close the socket so the receive loop reconnects with the current config.

        ``_ws`` is cleared first so PCM arriving during the close + backoff is
        buffered for replay instead of failing frame by frame on a dead socket.
        """
        ws = self._ws
        if ws is None:
            return
        self._reopen_requested = True
        self._ws = None
        try:
            await ws.close()
        except Exception as exc:
            self._reopen_requested = False
            logger.warning("OpenAIRealtimeSession: reopen close failed (%s)", exc)

    async def finalize(self, timeout: float = 3.0) -> None:
        """Commit the open turn and wait for every owed transcription.

        If the uncommitted audio contains speech, the detector's open turn is
        closed and committed; pure silence is never committed (it would only
        bill, and invite a hallucinated tail). Then we wait until every commit
        has been acked and transcribed, so the still-running receive/consume
        loop can persist the trailing ``.completed`` before teardown.
        ``timeout`` is a hard ceiling — we never block longer than that even if
        the server goes quiet (failsafe against a wedged socket).
        """
        if self._closed or self._ws is None:
            return
        decision = self._turn.flush()
        if decision.commit:
            meta = self._commit_meta(decision)
            self._pending_commits.append(meta)
            try:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            except Exception as exc:
                self._pending_commits.pop()
                logger.debug("OpenAIRealtimeSession: commit raised (%s) — continuing", exc)
                return
            self._drop_uncommitted()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            all_acked = not self._pending_commits
            all_transcribed = self._committed_count <= self._completed_count
            if all_acked and all_transcribed:
                break
            await asyncio.sleep(0.05)
        # Stop must not strand a transcript behind a turn that never came back.
        await self._flush_held()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        stats = self._turn.stats
        logger.info(
            "OpenAIRealtimeSession: closing (frames=%d speech_frames=%d commits=%d clears=%d noise_floor=%.4f)",
            stats.frames, stats.speech_frames, stats.commits, stats.clears, self._turn.noise_floor,
        )
        if self._reorder_task is not None:
            self._reorder_task.cancel()
            self._reorder_task = None
        # Held completions are already transcribed; hand them over in order
        # (non-blocking, like the courtesy event below).
        for seq in sorted(self._held):
            evt = self._held.pop(seq)
            if evt is not None:
                try:
                    self._events.put_nowait(evt)
                except asyncio.QueueFull:
                    break
        # Cancel a reconnect that may be sleeping in backoff so it can't wake
        # after close and reopen a fresh, leaked, billable session.
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reconnect_task = None
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None
        # Best-effort courtesy event. Must not block: if the consumer task has
        # already died, nobody is draining the queue and a blocking put() here
        # would hang teardown forever (and leak the ffmpeg pipe the caller
        # closes only after this returns).
        try:
            self._events.put_nowait(SessionStatus(state="disconnected"))
        except asyncio.QueueFull:
            pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_realtime_session(
    *,
    model_id: str,
    prompt: str | None = None,
    language: str | None = None,
) -> OpenAIRealtimeSession:
    """Return a provider-appropriate realtime transcription session.

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
        return OpenAIRealtimeSession(
            model_id=model_id,
            prompt=prompt,
            language=language,
        )
    raise UnsupportedModelError(f"No realtime session implementation for provider {info.provider!r}")


__all__ = [
    "OpenAIRealtimeSession",
    "build_realtime_session",
    "TranscriptDelta",
    "TranscriptCompleted",
    "SessionError",
    "SessionStatus",
    "SessionActivity",
    "SessionEvent",
    "RealtimeSessionError",
    "UnsupportedModelError",
]
