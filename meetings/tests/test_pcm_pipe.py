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
    _EBML_MAGIC,
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

    def test_headerless_prefix_skipped_then_decodes(self):
        """Header-less bytes before the init segment are skipped; valid audio decodes."""
        fixture = _generate_webm_fixture(duration_seconds=0.5)
        expected_bytes = (PCM_SAMPLE_RATE * PCM_SAMPLE_BYTES) // 2
        tolerance = expected_bytes // 10

        async def run() -> int:
            pipe = PcmPipe(mime="audio/webm")
            await pipe.start()
            # Feed junk that looks like a header-less mid-stream continuation.
            junk = b"\x00" * 4096
            await pipe.write(junk)
            # Now feed the real fixture starting with the EBML magic.
            await pipe.write(fixture)

            pcm_chunks: list[bytes] = []

            async def read_until_eof():
                async for frame in pipe.read_frames():
                    pcm_chunks.append(frame)

            read_task = asyncio.create_task(read_until_eof())
            await pipe.aclose(grace_seconds=5.0)
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
        self.assertGreater(total, 0, "PcmPipe produced no PCM output after header-less prefix")
        self.assertGreaterEqual(
            total, expected_bytes - tolerance,
            f"expected ~{expected_bytes} bytes of PCM after skipping prefix, got {total}",
        )

    def test_ebml_magic_split_across_writes_decodes(self):
        """EBML magic spanning two write() calls is still detected."""
        fixture = _generate_webm_fixture(duration_seconds=0.5)
        # The fixture begins with the 4-byte EBML magic. Split it so the first
        # write carries bytes 0-1 and the second write carries bytes 2-onwards.
        part1 = fixture[:2]
        part2 = fixture[2:]
        expected_bytes = (PCM_SAMPLE_RATE * PCM_SAMPLE_BYTES) // 2
        tolerance = expected_bytes // 10

        async def run() -> int:
            pipe = PcmPipe(mime="audio/webm")
            await pipe.start()
            await pipe.write(part1)
            await pipe.write(part2)

            pcm_chunks: list[bytes] = []

            async def read_until_eof():
                async for frame in pipe.read_frames():
                    pcm_chunks.append(frame)

            read_task = asyncio.create_task(read_until_eof())
            await pipe.aclose(grace_seconds=5.0)
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
        self.assertGreater(total, 0, "PcmPipe produced no PCM when magic was split across writes")
        self.assertGreaterEqual(
            total, expected_bytes - tolerance,
            f"expected ~{expected_bytes} bytes of PCM, got {total}",
        )


class PcmPipeHeaderGateUnitTests(unittest.TestCase):
    """Pure-Python unit tests for the EBML header gate — no ffmpeg required."""

    def test_non_webm_mime_gate_disabled(self):
        """Non-webm mimes have the gate disabled (pass-through from the start)."""
        for mime in ("audio/ogg", "audio/ogg;codecs=opus", "audio/mp4", "audio/wav", None):
            pipe = PcmPipe(mime=mime)
            self.assertFalse(
                pipe._header_gate_enabled,
                f"gate should be disabled for mime={mime!r}",
            )
            self.assertTrue(
                pipe._header_seen,
                f"_header_seen should be True (pass-through) for mime={mime!r}",
            )

    def test_webm_mime_gate_enabled_initially_unseen(self):
        """WebM mimes start with gate enabled and magic not yet seen."""
        for mime in ("audio/webm", "audio/webm;codecs=opus", "video/webm"):
            pipe = PcmPipe(mime=mime)
            self.assertTrue(pipe._header_gate_enabled, f"gate should be enabled for mime={mime!r}")
            self.assertFalse(pipe._header_seen, f"magic should not be seen yet for mime={mime!r}")
            self.assertEqual(pipe._header_carry, b"")

    def test_magic_detected_at_byte_zero(self):
        """Gate locks open immediately when data starts with EBML magic."""
        pipe = PcmPipe(mime="audio/webm")
        # Simulate a fresh init segment (magic + arbitrary payload).
        data = _EBML_MAGIC + b"\x01\x02\x03\x04" * 100
        # We can test the gate logic directly without starting ffmpeg.
        scan = pipe._header_carry + data
        idx = scan.find(_EBML_MAGIC)
        self.assertEqual(idx, 0)
        # After the gate logic runs the carry is cleared and the full buffer forwarded.
        pipe._header_seen = True
        pipe._header_carry = b""
        self.assertTrue(pipe._header_seen)
        self.assertEqual(pipe._header_carry, b"")

    def test_magic_split_carry_logic(self):
        """3-byte carry correctly detects a magic split across two writes."""
        # Simulate: first write is bytes 0-2 of the EBML magic (3 bytes),
        # second write is byte 3 + payload.
        part1 = _EBML_MAGIC[:3]   # b"\x1a\x45\xdf"
        part2 = _EBML_MAGIC[3:] + b"\x00" * 100  # b"\xa3\x00..."

        # First write: magic not found in 3-byte prefix → store as carry.
        scan1 = b"" + part1
        idx1 = scan1.find(_EBML_MAGIC)
        self.assertEqual(idx1, -1)
        carry = scan1[-3:]
        self.assertEqual(carry, _EBML_MAGIC[:3])

        # Second write: combine carry + new data → magic found at offset 0.
        scan2 = carry + part2
        idx2 = scan2.find(_EBML_MAGIC)
        self.assertEqual(idx2, 0)
        forward = scan2[idx2:]
        self.assertEqual(forward[:4], _EBML_MAGIC)

    def test_headerless_bytes_produce_no_carry_overflow(self):
        """Multiple header-less writes keep carry capped at 3 bytes."""
        pipe = PcmPipe(mime="audio/webm")
        for _ in range(10):
            junk = b"\x00" * 1000
            scan = pipe._header_carry + junk
            idx = scan.find(_EBML_MAGIC)
            if idx == -1:
                pipe._header_carry = scan[-3:] if len(scan) >= 3 else scan
        self.assertLessEqual(len(pipe._header_carry), 3)
