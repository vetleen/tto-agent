from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class MeetingsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "meetings"
    verbose_name = "Meetings"

    def ready(self):
        import meetings.signals  # noqa: F401 — retention + file cleanup signals

        # Skip background side-effects when running the test runner.
        if len(sys.argv) >= 2 and sys.argv[1] == "test":
            return

        self._sweep_orphan_chunks()

    # Chunk files are written as ``{segment_index:06d}.{ext}`` (see
    # meetings.services.chunks). Only delete files matching that shape so a
    # misconfigured temp dir can't turn the sweep into a general file reaper.
    _CHUNK_NAME_RE = re.compile(r"\d{6}\.[A-Za-z0-9]{1,8}")

    def _sweep_orphan_chunks(self) -> None:
        """Delete stale temp audio files older than 1 hour at startup.

        Defensive cleanup against worker crashes that left chunks behind.
        Uses os.walk so it tolerates a missing directory and partial trees.
        """
        # Read the raw setting first: Path("") is Path(".") (truthy, and
        # exists()), so ``if not temp_dir`` would be dead and an empty/misset
        # MEETING_CHUNK_TEMP_DIR would make os.walk sweep the process CWD.
        raw = getattr(settings, "MEETING_CHUNK_TEMP_DIR", "")
        if not raw:
            return
        temp_dir = Path(raw)
        if not temp_dir.exists():
            return
        cutoff = time.time() - 3600
        deleted = 0
        for root, _dirs, files in os.walk(temp_dir):
            for name in files:
                if not self._CHUNK_NAME_RE.fullmatch(name):
                    continue
                path = Path(root) / name
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        deleted += 1
                except OSError:
                    continue
        if deleted:
            logger.info("meetings: swept %d orphan chunk file(s)", deleted)
