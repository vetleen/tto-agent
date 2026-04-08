from __future__ import annotations

import logging
import os
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
        # Register chat tools on startup so the LLM can call them.
        import meetings.tools  # noqa: F401

        # Skip background side-effects when running the test runner.
        if len(sys.argv) >= 2 and sys.argv[1] == "test":
            return

        self._sweep_orphan_chunks()

    def _sweep_orphan_chunks(self) -> None:
        """Delete stale temp audio files older than 1 hour at startup.

        Defensive cleanup against worker crashes that left chunks behind.
        Uses os.walk so it tolerates a missing directory and partial trees.
        """
        temp_dir = Path(getattr(settings, "MEETING_CHUNK_TEMP_DIR", ""))
        if not temp_dir or not temp_dir.exists():
            return
        cutoff = time.time() - 3600
        deleted = 0
        for root, _dirs, files in os.walk(temp_dir):
            for name in files:
                path = Path(root) / name
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        deleted += 1
                except OSError:
                    continue
        if deleted:
            logger.info("meetings: swept %d orphan chunk file(s)", deleted)
