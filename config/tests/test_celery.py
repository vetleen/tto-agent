"""Tests guarding the Celery worker invocation in the Procfile."""

from pathlib import Path

from django.test import SimpleTestCase


class ProcfileWorkerCommandTests(SimpleTestCase):
    """Guard the worker invocation in the Procfile.

    Regression tests for two staging crash-loops:

    1. Celery 5 removed ``-A`` as an option of the ``worker`` subcommand, so it
       must be passed as a global option *before* ``worker``.
    2. The gevent/eventlet pools reject embedded beat (``-B``). The worker runs
       the ``threads`` pool, which supports ``-B`` — so a single worker dyno can
       still embed the scheduler. These assertions fail loudly if someone
       switches the pool back to gevent/eventlet while keeping ``-B``.
    """

    def _worker_line(self) -> str:
        procfile = Path(__file__).resolve().parents[2] / "Procfile"
        return next(
            line
            for line in procfile.read_text().splitlines()
            if line.startswith("worker:")
        )

    def test_worker_uses_threads_pool(self):
        line = self._worker_line()
        self.assertIn("--pool=threads", line)
        # gevent/eventlet can't embed beat; threads can. Guard against regressing.
        self.assertNotIn("gevent", line)
        self.assertNotIn("eventlet", line)

    def test_app_option_precedes_worker_subcommand(self):
        tokens = self._worker_line().split()
        self.assertIn("-A", tokens)
        self.assertIn("worker", tokens)
        self.assertLess(
            tokens.index("-A"),
            tokens.index("worker"),
            "-A must precede the 'worker' subcommand (Celery 5 removed -A as a "
            "worker-subcommand option)",
        )

    def test_embedded_beat_enabled(self):
        # -B embeds the scheduler in the single worker dyno (see RUNBOOK).
        self.assertIn("-B", self._worker_line().split())
