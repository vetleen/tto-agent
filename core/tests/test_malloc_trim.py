"""Tests for the periodic glibc malloc_trim daemon (``core.malloc_trim``).

Covers the process-gating decision (Linux web process only; never Celery/test/
management, honours MALLOC_TRIM_INTERVAL=0) and that ``trim_malloc`` is a safe
no-op off glibc. The daemon thread itself is never started here. The gating tests
pin ``sys.platform`` to ``linux`` so the argv/interval logic is exercised on any
host (the real Linux short-circuit is covered separately).
"""

import sys
from unittest import mock

from django.test import SimpleTestCase

from core import malloc_trim


class ShouldStartGatingTests(SimpleTestCase):
    def setUp(self):
        # Force the Linux branch so the argv/interval gating runs on any dev host.
        p = mock.patch.object(malloc_trim.sys, "platform", "linux")
        p.start()
        self.addCleanup(p.stop)

    def test_starts_on_web_process_by_default(self):
        self.assertTrue(malloc_trim._should_start({}, ["/app/.heroku/python/bin/daphne"]))

    def test_disabled_when_interval_zero(self):
        self.assertFalse(
            malloc_trim._should_start({"MALLOC_TRIM_INTERVAL": "0"}, ["daphne"])
        )

    def test_declines_on_celery_worker(self):
        self.assertFalse(
            malloc_trim._should_start({}, ["/app/.heroku/python/bin/celery", "-A", "config"])
        )

    def test_declines_for_management_and_test_commands(self):
        for cmd in ("test", "migrate", "collectstatic", "shell"):
            with self.subTest(cmd=cmd):
                self.assertFalse(malloc_trim._should_start({}, ["manage.py", cmd]))


class PlatformGateTests(SimpleTestCase):
    def test_declines_off_linux(self):
        with mock.patch.object(malloc_trim.sys, "platform", "win32"):
            self.assertFalse(malloc_trim._should_start({}, ["daphne"]))


class TrimMallocTests(SimpleTestCase):
    def test_trim_malloc_never_raises(self):
        # Returns True on glibc/Linux, False elsewhere — must never raise either way.
        result = malloc_trim.trim_malloc()
        self.assertIn(result, (True, False))
        if not sys.platform.startswith("linux"):
            self.assertFalse(result)

    def test_maybe_start_noop_when_disabled(self):
        self.assertFalse(
            malloc_trim.maybe_start(env={"MALLOC_TRIM_INTERVAL": "0"}, argv=["daphne"])
        )
        self.assertFalse(malloc_trim._started)
