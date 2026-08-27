"""Tests for ``core.redis_errors`` — level selection on best-effort broadcasts.

The point of the helper is observability: a Redis failure on a swallowed
``group_send`` must reach Sentry (WARNING, per the LoggingIntegration's
``event_level``), while the ordinary "nobody was listening" case stays at DEBUG so
it costs nothing. Getting the split wrong in either direction is a real problem —
too quiet hides an outage (WILFRED-6P), too loud floods the issue stream.
"""

from __future__ import annotations

import logging

from django.test import SimpleTestCase
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    MaxConnectionsError,
    TimeoutError as RedisTimeoutError,
)

from core.redis_errors import is_redis_blip, log_broadcast_failure


logger = logging.getLogger("core.tests.redis_errors_probe")


class IsRedisBlipTests(SimpleTestCase):
    def test_connection_error_is_a_blip(self):
        self.assertTrue(is_redis_blip(RedisConnectionError("boom")))

    def test_timeout_is_a_blip(self):
        self.assertTrue(is_redis_blip(RedisTimeoutError("boom")))

    def test_pool_exhaustion_is_a_blip(self):
        """A bounded pool at its cap raises MaxConnectionsError.

        It subclasses ConnectionError, which is exactly why bounding the pools is
        safe — see CHANNEL_LAYERS in config/settings.py.
        """
        self.assertTrue(is_redis_blip(MaxConnectionsError("Too many connections")))

    def test_ssl_handshake_reset_is_a_blip(self):
        """Heroku drops over-limit TLS handshakes as a builtin ConnectionResetError."""
        self.assertTrue(is_redis_blip(ConnectionResetError()))

    def test_ordinary_error_is_not_a_blip(self):
        self.assertFalse(is_redis_blip(ValueError("bad payload")))


class LogBroadcastFailureTests(SimpleTestCase):
    def test_redis_blip_logs_warning(self):
        with self.assertLogs(logger, level="DEBUG") as cm:
            log_broadcast_failure(
                logger, RedisConnectionError("boom"), "could not publish to %s", "g1"
            )
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].levelno, logging.WARNING)
        self.assertIn("g1", cm.records[0].getMessage())

    def test_redis_blip_attaches_the_traceback(self):
        """Sentry groups on the exception, so exc_info has to survive."""
        with self.assertLogs(logger, level="DEBUG") as cm:
            log_broadcast_failure(logger, RedisConnectionError("boom"), "x")
        self.assertIsNotNone(cm.records[0].exc_info)

    def test_non_redis_error_stays_debug(self):
        with self.assertLogs(logger, level="DEBUG") as cm:
            log_broadcast_failure(logger, ValueError("nope"), "could not publish")
        self.assertEqual(cm.records[0].levelno, logging.DEBUG)

    def test_redis_level_override_suppresses_a_repeat(self):
        """Hot paths pass DEBUG to stay quiet during a known outage."""
        with self.assertLogs(logger, level="DEBUG") as cm:
            log_broadcast_failure(
                logger,
                RedisConnectionError("boom"),
                "could not publish",
                redis_level=logging.DEBUG,
            )
        self.assertEqual(cm.records[0].levelno, logging.DEBUG)

    def test_override_does_not_promote_a_non_redis_error(self):
        """redis_level only ever applies to blips."""
        with self.assertLogs(logger, level="DEBUG") as cm:
            log_broadcast_failure(
                logger, ValueError("nope"), "x", redis_level=logging.ERROR
            )
        self.assertEqual(cm.records[0].levelno, logging.DEBUG)
