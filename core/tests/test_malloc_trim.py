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

    def test_unparseable_interval_still_starts(self):
        # A typo (e.g. "60s") must not silently disable the R14 mitigation —
        # only a valid non-positive value is a deliberate off switch. maybe_start
        # then clamps the bad value back to the default interval.
        self.assertTrue(
            malloc_trim._should_start({"MALLOC_TRIM_INTERVAL": "60s"}, ["daphne"])
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


class MaybeTrimTests(SimpleTestCase):
    """The task-triggered worker trim (``maybe_trim``): threshold- and
    rate-limited, and never raises into the calling Celery task."""

    def setUp(self):
        # Put the last-trim clock well in the past so the rate limit doesn't
        # block the first call on a freshly-booted host (monotonic() may be small).
        malloc_trim._last_maybe_trim = malloc_trim.time.monotonic() - 10_000

    def test_skips_when_rss_below_threshold(self):
        with mock.patch.object(malloc_trim, "_read_rss_kb", return_value=100 * 1024), \
             mock.patch.object(malloc_trim, "trim_malloc") as trim:
            self.assertFalse(malloc_trim.maybe_trim(threshold_mb=700, min_interval_s=0))
            trim.assert_not_called()

    def test_trims_when_rss_elevated(self):
        # First read (threshold check) is high; second (post-trim log) is lower.
        with mock.patch.object(malloc_trim, "_read_rss_kb", side_effect=[800 * 1024, 600 * 1024]), \
             mock.patch.object(malloc_trim, "trim_malloc", return_value=True) as trim:
            self.assertTrue(malloc_trim.maybe_trim(threshold_mb=700, min_interval_s=0))
            trim.assert_called_once()

    def test_rate_limited_within_interval(self):
        with mock.patch.object(malloc_trim, "_read_rss_kb", return_value=800 * 1024), \
             mock.patch.object(malloc_trim, "trim_malloc", return_value=True) as trim:
            self.assertTrue(malloc_trim.maybe_trim(threshold_mb=700, min_interval_s=1000))
            # A second call inside the interval must not trim again.
            self.assertFalse(malloc_trim.maybe_trim(threshold_mb=700, min_interval_s=1000))
            trim.assert_called_once()

    def test_never_raises_into_task(self):
        with mock.patch.object(malloc_trim, "_read_rss_kb", side_effect=RuntimeError("boom")):
            self.assertFalse(malloc_trim.maybe_trim(min_interval_s=0))
