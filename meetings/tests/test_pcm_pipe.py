"""End-to-end test for PcmPipe against the real ffmpeg binary.

Requires ``ffmpeg`` on PATH (the same constraint the upload path already
has). The test drives a short generated sine wave through ffmpeg twice:
once to produce a WebM/Opus container that mimics what MediaRecorder
emits in Chrome, and then a second time through PcmPipe to decode that
container back to PCM16@24kHz. We assert that the decoded PCM has the
right length (within tolerance for encoder framing) and is non-zero.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from meetings.services.pcm_pipe import (
    DEFAULT_FRAME_BYTES,
    PcmPipe,
    PCM_SAMPLE_BYTES,
    PCM_SAMPLE_RATE,
)


def _ffmpeg_on_path() -> bool:
    return shutil.which("ffmpeg") is not None


def _run_async(coro):
    """Run a coroutine on a fresh event loop.

    PcmPipe uses ``subprocess.Popen`` plus a pair of background threads
    rather than ``asyncio.create_subprocess_exec``, so the loop type
    doesn't matter anymore — works on both SelectorEventLoop and
    ProactorEventLoop. We still create a fresh loop to avoid leaking
    state between tests.
    """
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        finally:
            asyncio.set_event_loop(None)


def _generate_webm_fixture(duration_seconds: float = 1.0) -> bytes:
    """Return WebM/Opus bytes representing ``duration_seconds`` of a 440Hz sine tone."""
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp.close()
    path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi",
                "-i", f"sine=frequency=440:duration={duration_seconds}",
                "-c:a", "libopus",
                "-b:a", "64k",
                "-f", "webm",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


@unittest.skipUnless(_ffmpeg_on_path(), "ffmpeg not on PATH")
class PcmPipeTests(unittest.TestCase):
    def test_decodes_webm_opus_to_pcm16_at_24khz(self):
        fixture = _generate_webm_fixture(duration_seconds=1.0)
        # 1s of PCM16 mono @ 24kHz = 24000 * 2 = 48000 bytes expected. The
        # Opus encoder pads the tail with a few frames of silence so we
        # accept +/- 10% of the target.
        expected_bytes = PCM_SAMPLE_RATE * PCM_SAMPLE_BYTES
        tolerance = expected_bytes // 10

        async def run() -> int:
            pipe = PcmPipe(mime="audio/webm")
            await pipe.start()
            # Feed the whole container at once — ffmpeg buffers it.
            await pipe.write(fixture)

            # Drain stdout in a bounded reader task; close stdin to signal EOF.
            pcm_chunks: list[bytes] = []

            async def read_until_eof():
                async for frame in pipe.read_frames():
                    pcm_chunks.append(frame)

            read_task = asyncio.create_task(read_until_eof())
            await pipe.aclose(grace_seconds=5.0)
            # After aclose, read_frames should complete (ffmpeg stdout closed).
            try:
                await asyncio.wait_for(read_task, timeout=5.0)
            except asyncio.TimeoutError:
                read_task.cancel()
                try:
                    await read_task
                except Exception:
                    pass
            return sum(len(c) for c in pcm_chunks)

        total = _run_async(run())
        self.assertGreater(total, 0, "PcmPipe produced no PCM output")
        self.assertGreaterEqual(
            total, expected_bytes - tolerance,
            f"expected ~{expected_bytes} bytes of PCM, got {total}",
        )
        self.assertLessEqual(
            total, expected_bytes + tolerance,
            f"expected ~{expected_bytes} bytes of PCM, got {total}",
        )

    def test_frame_size_matches_40ms_slices(self):
        # Sanity check: our declared frame size matches the math we use
        # elsewhere (24kHz * 40ms * 2 bytes/sample = 1920).
        self.assertEqual(DEFAULT_FRAME_BYTES, 1920)
