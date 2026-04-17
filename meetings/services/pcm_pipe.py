"""Long-running ffmpeg subprocess transcoding container audio to 24kHz PCM16.

The Realtime transcription API expects a continuous stream of 16-bit
little-endian PCM at 24kHz mono. Browser MediaRecorder output is a WebM
(Chrome/Edge), Ogg (Firefox), or MP4 (Safari) container with Opus or AAC
audio — we cannot send it straight to OpenAI.

One ``PcmPipe`` per live session owns one long-lived ``ffmpeg`` process.
Container bytes are written to stdin as they arrive; decoded PCM flows
out of stdout and is read in small slices ready to be base64-encoded and
sent to the Realtime API.

Implementation note — why threads, not ``asyncio.create_subprocess_exec``
-----------------------------------------------------------------------
We use blocking ``subprocess.Popen`` with two background threads that
adapt reads/writes to asyncio via ``asyncio.to_thread`` and
``loop.call_soon_threadsafe``. This side-steps a Windows-only bug: the
default ``SelectorEventLoop`` does not implement ``subprocess_exec`` and
Daphne's Twisted asyncio reactor tends to land on that loop despite our
best efforts to force the Proactor policy. The thread-pool approach
works on any event loop (ProactorEventLoop, SelectorEventLoop, uvloop)
and on every platform. It costs us two extra OS threads per active live
meeting — acceptable.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import subprocess
import threading
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

# Bounded ring of decoded PCM frames waiting for the forwarder task. Five
# seconds at our frame size is ~125 frames — enough smoothing for transient
# forwarder delays without risking unbounded memory on a stuck session.
_STDOUT_QUEUE_MAX_FRAMES = 125

_EOF = b""  # sentinel marking stdout EOF


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
    stdout_drops: int = 0


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
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_q: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closed = False
        self._stdin_lock = threading.Lock()
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

        # Popen runs the fork/exec on the calling thread — cheap enough to do
        # synchronously. We don't need to wrap this in ``asyncio.to_thread``.
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # unbuffered — we want PCM frames immediately available
        )
        self._loop = asyncio.get_running_loop()
        self._stdout_q = asyncio.Queue(maxsize=_STDOUT_QUEUE_MAX_FRAMES)
        self._stdout_thread = threading.Thread(
            target=self._stdout_reader_thread,
            name=f"PcmPipe-stdout-{self._proc.pid}",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_thread,
            name=f"PcmPipe-stderr-{self._proc.pid}",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    # ---------------- background threads ----------------

    def _stdout_reader_thread(self) -> None:
        """Read PCM frames from ffmpeg stdout and push them onto the asyncio queue.

        Uses ``loop.call_soon_threadsafe`` to hand frames to the event loop.
        When the queue is full the oldest frame is dropped — realtime
        transcription is a soft-real-time workload, and buffering stale audio
        helps nobody. Increments a stat so operators can see drops.
        """
        assert self._proc is not None and self._proc.stdout is not None
        assert self._stdout_q is not None and self._loop is not None
        stdout = self._proc.stdout
        frame_bytes = self._frame_bytes
        try:
            while True:
                chunk = _read_exact(stdout, frame_bytes)
                if not chunk:
                    break
                self.stats.bytes_out += len(chunk)
                self._loop.call_soon_threadsafe(self._enqueue_frame, chunk)
        except Exception:
            logger.exception("PcmPipe: stdout reader crashed")
        finally:
            # Signal EOF to any waiting consumer.
            try:
                self._loop.call_soon_threadsafe(self._enqueue_frame, _EOF)
            except RuntimeError:
                # Loop is already closed — nothing to do.
                pass

    def _enqueue_frame(self, chunk: bytes) -> None:
        """Runs on the event loop thread — push to queue, drop-oldest on full."""
        if self._stdout_q is None:
            return
        try:
            self._stdout_q.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                _ = self._stdout_q.get_nowait()
                self.stats.stdout_drops += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._stdout_q.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def _stderr_reader_thread(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        stderr = self._proc.stderr
        try:
            while True:
                line = stderr.readline()
                if not line:
                    return
                logger.warning(
                    "ffmpeg: %s", line.decode("utf-8", errors="replace").rstrip()
                )
        except Exception:
            logger.exception("PcmPipe: stderr reader crashed")

    # ---------------- public async API ----------------

    async def write(self, data: bytes) -> None:
        """Forward container bytes to ffmpeg stdin.

        Offloads the blocking write to the default executor so the event
        loop stays responsive. A lock serialises writes from multiple
        concurrent tasks (shouldn't happen in normal flow, but defence).
        """
        if self._closed or self._proc is None or self._proc.stdin is None:
            raise PcmPipeError("pipe is not running")
        if not data:
            return

        def _do_write():
            with self._stdin_lock:
                try:
                    assert self._proc is not None and self._proc.stdin is not None
                    self._proc.stdin.write(data)
                    self._proc.stdin.flush()
                except (BrokenPipeError, ValueError, OSError) as exc:
                    raise PcmPipeError(f"ffmpeg stdin closed: {exc}") from exc

        await asyncio.to_thread(_do_write)
        self.stats.bytes_in += len(data)

    async def read_frames(self) -> AsyncIterator[bytes]:
        """Yield PCM frames until ffmpeg's stdout closes."""
        if self._stdout_q is None:
            raise PcmPipeError("pipe is not running")
        q = self._stdout_q
        while True:
            chunk = await q.get()
            if chunk == _EOF:
                return
            yield chunk

    async def aclose(self, *, grace_seconds: float = 3.0) -> None:
        """Close stdin, wait for ffmpeg to exit, force-kill on timeout."""
        if self._closed:
            return
        self._closed = True
        if self._proc is None:
            return

        def _shutdown():
            assert self._proc is not None
            try:
                if self._proc.stdin is not None:
                    try:
                        self._proc.stdin.close()
                    except Exception:
                        pass
                try:
                    self._proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "PcmPipe: ffmpeg did not exit within %.1fs — killing",
                        grace_seconds,
                    )
                    try:
                        self._proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        self._proc.wait(timeout=2.0)
                    except Exception:
                        pass
            finally:
                # Close stdout/stderr so the reader threads exit their read loops.
                for fh in (self._proc.stdout, self._proc.stderr):
                    try:
                        if fh is not None:
                            fh.close()
                    except Exception:
                        pass

        try:
            await asyncio.to_thread(_shutdown)
        finally:
            # Join reader threads briefly so they don't outlive the pipe.
            for t in (self._stdout_thread, self._stderr_thread):
                if t is not None:
                    t.join(timeout=1.0)
            self._stdout_thread = None
            self._stderr_thread = None


def _read_exact(fh, n: int) -> bytes:
    """Read exactly ``n`` bytes from a blocking file handle, or whatever arrives before EOF.

    Mirrors ``asyncio.StreamReader.readexactly`` semantics — returns a short
    final buffer on EOF instead of raising. The synchronous equivalent
    (``fh.read(n)``) may return fewer bytes for non-EOF reads on pipes, so
    we loop to collect a full frame before yielding.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            return bytes(buf)
        buf.extend(chunk)
    return bytes(buf)


__all__ = [
    "PcmPipe",
    "PcmPipeError",
    "PcmPipeStats",
    "PCM_SAMPLE_RATE",
    "PCM_SAMPLE_BYTES",
    "PCM_CHANNELS",
    "DEFAULT_FRAME_BYTES",
]
