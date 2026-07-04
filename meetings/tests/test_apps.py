"""Tests for the startup orphan-chunk sweeper (meetings.apps)."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from meetings.apps import MeetingsConfig


def _sweep():
    # apps.MeetingsConfig._sweep_orphan_chunks is a bound-ish method; call it on
    # a lightweight instance (no AppConfig machinery is exercised by the sweep).
    config = MeetingsConfig.__new__(MeetingsConfig)
    config._sweep_orphan_chunks()


class SweepOrphanChunksTests(SimpleTestCase):
    @override_settings(MEETING_CHUNK_TEMP_DIR="")
    def test_blank_setting_returns_without_walking_cwd(self):
        # Path("") == Path(".") is truthy and exists(), so a blank setting used
        # to make the sweep os.walk the process CWD. It must now no-op instead.
        sentinel = Path("meetings") / "apps.py"
        self.assertTrue(sentinel.exists())
        _sweep()
        # A real source file older than 1h in the CWD must be untouched.
        self.assertTrue(sentinel.exists())

    def test_only_chunk_pattern_files_removed(self):
        with tempfile.TemporaryDirectory(prefix="sweep_test_") as d:
            old = time.time() - 7200  # 2h ago
            chunk = Path(d) / "000003.webm"
            other = Path(d) / "important.txt"
            chunk.write_bytes(b"x")
            other.write_bytes(b"y")
            for p in (chunk, other):
                os.utime(p, (old, old))

            with override_settings(MEETING_CHUNK_TEMP_DIR=d):
                _sweep()

            # Chunk-named file swept; non-chunk file preserved even though it's old.
            self.assertFalse(chunk.exists())
            self.assertTrue(other.exists())

    def test_recent_chunk_files_kept(self):
        with tempfile.TemporaryDirectory(prefix="sweep_test_") as d:
            chunk = Path(d) / "000000.mp3"
            chunk.write_bytes(b"x")  # freshly created (mtime = now)
            with override_settings(MEETING_CHUNK_TEMP_DIR=d):
                _sweep()
            self.assertTrue(chunk.exists())
