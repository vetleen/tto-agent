"""Tests for ``chat.tasks._notify_consumer``'s retry-once behavior.

The sub-agent completion broadcast rides the shared Redis Mini, which drops
connections under load (WILFRED-6P). From a sync Celery thread every
``async_to_sync`` call builds a fresh event loop and pubsub pool, so one retry
after a short sleep gets a brand-new connection — the notification must survive
a single blip silently and only WARN when both attempts fail.
"""

import logging
from unittest.mock import patch

from django.test import SimpleTestCase
from redis.exceptions import ConnectionError as RedisConnectionError

from chat.tasks import _notify_consumer


class _FlakyLayer:
    """group_send raises the queued exceptions in order, then succeeds."""

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    async def group_send(self, group, message):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)


class NotifyConsumerRetryTests(SimpleTestCase):
    def _run(self, layer):
        with patch("channels.layers.get_channel_layer", return_value=layer):
            with patch("chat.tasks.time.sleep") as mock_sleep:
                _notify_consumer("run-1", "thread-1")
        return mock_sleep

    def test_success_first_try_no_retry(self):
        layer = _FlakyLayer([])
        sleep = self._run(layer)
        self.assertEqual(layer.calls, 1)
        sleep.assert_not_called()

    def test_transient_blip_recovers_silently(self):
        layer = _FlakyLayer([RedisConnectionError("boom")])
        with self.assertNoLogs("chat.tasks", level="WARNING"):
            sleep = self._run(layer)
        self.assertEqual(layer.calls, 2)
        sleep.assert_called_once()

    def test_both_attempts_fail_warns_once(self):
        layer = _FlakyLayer(
            [RedisConnectionError("boom"), RedisConnectionError("boom")]
        )
        with self.assertLogs("chat.tasks", level="WARNING") as cm:
            self._run(layer)
        self.assertEqual(layer.calls, 2)
        warnings = [r for r in cm.records if r.levelno >= logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("Could not notify consumer", warnings[0].getMessage())

    def test_non_redis_failure_stays_debug_and_never_raises(self):
        layer = _FlakyLayer([ValueError("bad"), ValueError("bad")])
        with self.assertLogs("chat.tasks", level="DEBUG") as cm:
            self._run(layer)  # must not raise
        self.assertTrue(all(r.levelno < logging.WARNING for r in cm.records))
