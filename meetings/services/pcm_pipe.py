"""Long-running ffmpeg subprocess transcoding container audio to 24kHz PCM16.

The Realtime transcription API expects a continuous stream of 16-bit
little-endian PCM at 24kHz mono. Browser MediaRecorder output is a WebM
(Chrome/Edge), Ogg (Firefox), or MP4 (Safari) container with Opus or AAC
audio — we cannot send it straight to OpenAI.

One ``PcmPipe`` per live session spawns a single long-lived ``ffmpeg``
process with stdin/stdout pipes. Incoming container bytes are written to
stdin as they arrive; decoded PCM flows out of stdout and is read in
small slices ready to be base64-encoded and sent to the Realtime API.

Design notes
------------
* One subprocess per session — cheaper than spawning per WebSocket frame
  (fork+exec is tens-of-ms on Linux, hundreds on Windows/Heroku dynos).
* The process is driven by ``asyncio.subprocess`` so stdin/stdout/stderr
  drain in parallel with the Channels consumer's event loop.
* stderr is drained on a dedicated task and logged at WARNING so ffmpeg
  errors surface in Sentry without blocking decode.
* Backpressure: stdin writes go through ``drain()``; if ffmpeg stops
  reading, the producer blocks rather than buffering unbounded bytes in
  Python. stdout is read in fixed-size slices.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class PcmPipeError(RuntimeError):
    """Raised when the ffmpeg subprocess exits or fails to start."""


# Target format expected by OpenAI Realtime for input_audio_format="pcm16":
# 16-bit little-endian, 24 kHz, mono.
PCM_SAMPLE_RATE = 24_000
PCM_SAMPLE_BYTES = 2       # 16-bit
PCM_CHANNELS = 1

# 40ms slices at 24kHz mono PCM16 = 960 samples * 2 bytes = 1920 bytes.
# Smaller slices mean lower latency to the Realtime server (each append
# gets VAD-processed sooner) but more WebSocket frames per second.
DEFAULT_FRAME_MS = 40
DEFAULT_FRAME_BYTES = (PCM_SAMPLE_RATE * DEFAULT_FRAME_MS // 1000) * PCM_SAMPLE_BYTES * PCM_CHANNELS  # 1920


# Mime → ffmpeg input-format flag. ffmpeg's Matroska demuxer handles the
# Chrome/Edge WebM family; Safari serves MP4/AAC; Firefox occasionally
# Ogg/Opus. When the client-reported mime is unknown, fall back to
# letting ffmpeg sniff the format (omit ``-f`` entirely).
_MIME_TO_DEMUXER: dict[str, str] = {
    "audio/webm": "matroska",
    "audio/webm;codecs=opus": "matroska",
    "video/webm": "matroska",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/mp4": "mov",
    "audio/mp4;codecs=opus": "mov",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def _demuxer_for_mime(mime: str | None) -> str | None:
    if not mime:
        return None
    key = mime.strip().lower().split(";")[0]
    return _MIME_TO_DEMUXER.get(key) or _MIME_TO_DEMUXER.get(mime.strip().lower())


@dataclass
class PcmPipeStats:
    """Diagnostic counters for a running PcmPipe (exposed for observability)."""
    bytes_in: int = 0
    bytes_out: int = 0
    stdout_underruns: int = 0


class PcmPipe:
    """Wrap a single long-lived ffmpeg subprocess decoding container audio to PCM16.

    Usage::

        pipe = PcmPipe(mime="audio/webm")
        await pipe.start()
        await pipe.write(container_bytes)
        async for frame in pipe.read_frames():
            await realtime.send_pcm(frame)
        await pipe.aclose()

    The caller is responsible for running ``write`` and ``read_frames``
    on separate tasks so neither blocks the other.
    """

    def __init__(self, mime: str | None = None, *, frame_bytes: int = DEFAULT_FRAME_BYTES):
        self._mime = mime
        self._frame_bytes = frame_bytes
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = False
        self.stats = PcmPipeStats()

    async def start(self) -> None:
        if self._proc is not None:
            return
        if shutil.which("ffmpeg") is None:
            raise PcmPipeError("ffmpeg is not installed or not on PATH")

        demuxer = _demuxer_for_mime(self._mime)
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if demuxer:
            args += ["-f", demuxer]
        args += [
            "-i", "pipe:0",
            "-vn",                         # drop any video stream
            "-ac", str(PCM_CHANNELS),      # mono
            "-ar", str(PCM_SAMPLE_RATE),   # 24 kHz
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ]
        logger.info("PcmPipe: launching ffmpeg mime=%s demuxer=%s", self._mime, demuxer)
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                # ffmpeg's -loglevel error only surfaces actual decode problems,
                # so log at WARNING to make them visible without being noisy.
                logger.warning("ffmpeg: %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("PcmPipe: stderr drain task crashed")

    async def write(self, data: bytes) -> None:
        """Forward container bytes to ffmpeg stdin.

        Applies backpressure via ``drain()`` — if ffmpeg is slow to
        consume, the caller awaits here rather than the pipe buffering
        unbounded bytes in Python.
        """
        if self._closed or self._proc is None or self._proc.stdin is None:
            raise PcmPipeError("pipe is not running")
        if not data:
            return
        try:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()
            self.stats.bytes_in += len(data)
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise PcmPipeError(f"ffmpeg stdin closed: {exc}") from exc

    async def read_frames(self) -> AsyncIterator[bytes]:
        """Yield PCM frames of ``frame_bytes`` each until ffmpeg closes stdout."""
        if self._proc is None or self._proc.stdout is None:
            raise PcmPipeError("pipe is not running")
        while True:
            try:
                chunk = await self._proc.stdout.readexactly(self._frame_bytes)
            except asyncio.IncompleteReadError as exc:
                # ffmpeg closed stdout — drain whatever partial frame is left.
                if exc.partial:
                    self.stats.bytes_out += len(exc.partial)
                    yield exc.partial
                return
            self.stats.bytes_out += len(chunk)
            yield chunk

    async def aclose(self, *, grace_seconds: float = 3.0) -> None:
        """Close stdin, wait for ffmpeg to exit, force-kill on timeout."""
        if self._closed:
            return
        self._closed = True
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            logger.warning("PcmPipe: ffmpeg did not exit within %.1fs — killing", grace_seconds)
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                await self._proc.wait()
            except Exception:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stderr_task = None


__all__ = [
    "PcmPipe",
    "PcmPipeError",
    "PcmPipeStats",
    "PCM_SAMPLE_RATE",
    "PCM_SAMPLE_BYTES",
    "PCM_CHANNELS",
    "DEFAULT_FRAME_BYTES",
]
