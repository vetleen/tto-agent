"""Tests for the shared-Redis connection settings in config/settings.py.

Production runs a 20-connection Heroku Redis Mini shared by the Celery broker, the
Django cache, and the channels layer. Exceeding that cap makes Heroku drop the
over-limit TLS handshakes for *every* consumer on the instance, which is what
produced the WebSocket reconnect storm in WILFRED-6P. These tests pin the two
properties that keep the app inside the cap, plus the code invariant the pub/sub
channel layer depends on.
"""

from __future__ import annotations

import ast
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


REPO_ROOT = Path(__file__).resolve().parents[2]

# Apps whose source may touch the channel layer. Tests are excluded: they run on
# InMemoryChannelLayer, which does support cross-process-style sends.
_CHANNEL_LAYER_APPS = ("chat", "meetings", "documents", "core", "llm", "agent_skills")


class ChannelLayerSettingsTest(SimpleTestCase):
    """The channels layer must be the pub/sub backend, with a bounded pool."""

    def test_uses_pubsub_channel_layer(self):
        """RedisPubSubChannelLayer keeps a constant per-process connection count.

        channels_redis.core.RedisChannelLayer takes a pool connection per concurrent
        operation, so a 16-thread worker fan-out could exhaust the 20-connection cap
        by itself. The pub/sub layer holds one publisher + one subscriber per process
        and multiplexes everything over them.
        """
        self.assertEqual(
            settings.CHANNEL_LAYERS["default"]["BACKEND"],
            "channels_redis.pubsub.RedisPubSubChannelLayer",
        )

    def test_channel_layer_pool_is_bounded(self):
        """redis-py defaults max_connections to 2**31 — effectively unbounded."""
        hosts = settings.CHANNEL_LAYERS["default"]["CONFIG"]["hosts"]
        self.assertEqual(len(hosts), 1)
        host = hosts[0]
        self.assertIsInstance(
            host, dict, "host must be a dict so max_connections reaches the pool"
        )
        max_connections = host.get("max_connections")
        self.assertIsNotNone(max_connections, "channel layer pool must be bounded")
        # Two connections are the working minimum (publisher + subscriber); the cap
        # has to stay well under the 20-connection instance limit it shares.
        self.assertGreaterEqual(max_connections, 2)
        self.assertLess(max_connections, 20)

    def test_channel_layer_connection_is_health_checked(self):
        """A silently-dropped pub/sub connection must be detected before use.

        Without health_check_interval / socket_keepalive the long-lived subscriber
        connection can go stale and the next SUBSCRIBE at WebSocket-connect fails the
        handshake (WILFRED-7B). Both keys are passed straight to ConnectionPool.from_url.
        """
        host = settings.CHANNEL_LAYERS["default"]["CONFIG"]["hosts"][0]
        self.assertGreaterEqual(host.get("health_check_interval", 0), 1)
        self.assertIs(host.get("socket_keepalive"), True)

    def test_channel_layer_connection_retries(self):
        """health_check_interval only detects a dead pub/sub socket; a retry is what
        lets it reconnect instead of failing the WS-connect SUBSCRIBE (WILFRED-7B)."""
        host = settings.CHANNEL_LAYERS["default"]["CONFIG"]["hosts"][0]
        retry = host.get("retry")
        self.assertIsNotNone(retry, "channel layer connection must configure a retry")
        # redis Retry keeps its attempt budget in _retries; >=1 means it reconnects.
        self.assertGreaterEqual(getattr(retry, "_retries", 0), 1)

    def test_cache_connection_is_health_checked(self):
        """The cache shares the instance and the same stale-connection risk."""
        options = settings.CACHES["default"].get("OPTIONS", {})
        self.assertGreaterEqual(options.get("health_check_interval", 0), 1)
        self.assertIs(options.get("socket_keepalive"), True)

    def test_cache_pool_is_bounded(self):
        """The Django cache shares the same instance and needs the same bound."""
        options = settings.CACHES["default"].get("OPTIONS", {})
        max_connections = options.get("max_connections")
        self.assertIsNotNone(max_connections, "cache pool must be bounded")
        self.assertGreaterEqual(max_connections, 2)
        self.assertLess(max_connections, 20)

    def test_cache_keeps_tls_option_alongside_pool_bound(self):
        """Adding max_connections must not have displaced ssl_cert_reqs.

        Both live in the same OPTIONS dict, and Heroku Redis serves a self-signed
        chain that fails default verification — dropping ssl_cert_reqs would break
        every cache call in production while passing every local test.
        """
        import inspect

        import config.settings as config_mod

        source = inspect.getsource(config_mod)
        self.assertIn('_cache_config["OPTIONS"]["ssl_cert_reqs"] = ssl.CERT_NONE', source)


class ChannelLayerUsageTest(SimpleTestCase):
    """Pub/sub has no cross-process delivery to a *specific* channel.

    ``group_send`` is fine — it publishes to a group channel every interested process
    is subscribed to. ``channel_layer.send(some_other_process_channel, ...)`` is not:
    under the pub/sub layer it silently goes nowhere. Nothing uses it today; this test
    keeps it that way, because the failure mode is a silent drop rather than an error.
    """

    def _channel_layer_sends(self, path: Path) -> list[int]:
        """Return line numbers of ``<something>.send(...)`` calls on a channel layer."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "send":
                continue
            # Match `channel_layer.send(...)` / `self.channel_layer.send(...)` /
            # `layer.send(...)` — the names the call sites actually bind.
            target = func.value
            name = None
            if isinstance(target, ast.Name):
                name = target.id
            elif isinstance(target, ast.Attribute):
                name = target.attr
            if name in {"channel_layer", "layer", "_channel_layer"}:
                hits.append(node.lineno)
        return hits

    def test_no_direct_channel_layer_send(self):
        offenders = []
        for app in _CHANNEL_LAYER_APPS:
            app_dir = REPO_ROOT / app
            if not app_dir.is_dir():
                continue
            for path in app_dir.rglob("*.py"):
                if "tests" in path.parts or path.name.startswith("test_"):
                    continue
                for lineno in self._channel_layer_sends(path):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        self.assertEqual(
            offenders,
            [],
            "Direct channel_layer.send() is unsupported by the pub/sub channel layer "
            "and drops the message silently. Use group_send instead. Offenders: "
            + ", ".join(offenders),
        )
