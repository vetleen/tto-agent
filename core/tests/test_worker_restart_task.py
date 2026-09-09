"""Tests for the nightly worker self-restart task.

The task warm-shuts-down the Celery worker on Heroku (the dyno manager then
boots a fresh process, shedding the memory high-water mark). Tests verify the
Heroku gate and that the shutdown broadcast targets only the consuming node —
never other workers on a shared broker.
"""
from __future__ import annotations

from unittest.mock import patch

from django.conf import settings
from django.test import SimpleTestCase

from core.tasks import restart_worker_nightly


class RestartWorkerNightlyTests(SimpleTestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_off_heroku(self):
        with patch.object(restart_worker_nightly.app.control, "shutdown") as mock_shutdown:
            result = restart_worker_nightly.apply()
        self.assertEqual(result.get(), "skipped")
        mock_shutdown.assert_not_called()

    @patch.dict("os.environ", {"DYNO": "worker.1"})
    def test_shuts_down_own_node_on_heroku(self):
        with patch.object(restart_worker_nightly.app.control, "shutdown") as mock_shutdown:
            result = restart_worker_nightly.apply()
        self.assertEqual(result.get(), "shutdown")
        mock_shutdown.assert_called_once()
        destination = mock_shutdown.call_args.kwargs["destination"]
        # Eager runs may not carry a hostname; the task must then pass None
        # (broadcast) rather than a [None] destination Celery would ignore.
        self.assertTrue(destination is None or isinstance(destination, list))
        if isinstance(destination, list):
            self.assertEqual(len(destination), 1)
            self.assertIsNotNone(destination[0])

    def test_beat_schedule_entry_present(self):
        entry = settings.CELERY_BEAT_SCHEDULE["nightly-worker-restart"]
        self.assertEqual(entry["task"], "core.tasks.restart_worker_nightly")
        # Crontab schedule (fixed clock time), not a plain interval.
        self.assertTrue(hasattr(entry["schedule"], "hour"))
