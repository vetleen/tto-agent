"""Tests for the gevent worker launcher (``config/celery_gevent.py``).

These deliberately mock out the real monkeypatching — calling
``gevent.monkey.patch_all()`` inside the test process would patch the test
runner's sockets/threads and corrupt the rest of the suite. We only assert that
the launcher wires the calls together in the right order.
"""

import sys
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

import celery_gevent as cg


class CeleryGeventLauncherTests(SimpleTestCase):
    def test_patch_applies_gevent_then_psycopg(self):
        manager = mock.Mock()
        with mock.patch("gevent.monkey.patch_all") as m_patch_all, mock.patch(
            "psycogreen.gevent.patch_psycopg"
        ) as m_patch_pg:
            manager.attach_mock(m_patch_all, "patch_all")
            manager.attach_mock(m_patch_pg, "patch_psycopg")
            cg._patch()

        m_patch_all.assert_called_once_with()
        m_patch_pg.assert_called_once_with()
        # gevent must be patched before psycopg2's wait callback is installed.
        self.assertEqual(
            [name for name, _, _ in manager.mock_calls],
            ["patch_all", "patch_psycopg"],
        )

    def test_main_patches_before_handing_off_to_celery(self):
        original_argv = list(sys.argv)
        try:
            with mock.patch.object(cg, "_patch") as m_patch, mock.patch(
                "celery.__main__.main"
            ) as m_celery_main:
                m_celery_main.side_effect = lambda: self.assertTrue(
                    m_patch.called,
                    "monkeypatching must happen before the celery CLI starts",
                )
                cg.main()

            m_patch.assert_called_once_with()
            m_celery_main.assert_called_once_with()
            # celery's CLI inspects argv[0]; the launcher must normalize it.
            self.assertEqual(sys.argv[0], "celery")
        finally:
            sys.argv[:] = original_argv


class ProcfileWorkerCommandTests(SimpleTestCase):
    """Guard the worker invocation in the Procfile.

    Regression test for a staging crash-loop: Celery 5 removed ``-A`` as an
    option of the ``worker`` subcommand, so it must be passed as a global
    option *before* ``worker``. The launcher must also be the root-level
    ``celery_gevent`` module (not ``config.celery_gevent``, which imports the
    ``config`` package — and thus ssl — before monkeypatching).
    """

    def _worker_line(self) -> str:
        procfile = Path(__file__).resolve().parents[2] / "Procfile"
        return next(
            line
            for line in procfile.read_text().splitlines()
            if line.startswith("worker:")
        )

    def test_worker_launches_root_gevent_module(self):
        line = self._worker_line()
        self.assertIn("celery_gevent", line)
        self.assertNotIn("config.celery_gevent", line)
        self.assertIn("--pool=gevent", line)

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
