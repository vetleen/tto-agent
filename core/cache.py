"""Redis cache backend that fails open on transient connection blips.

The app runs on a small (20-connection) Heroku Redis shared by the Django cache,
the Celery broker, and the channels layer. Under a batch fan-out the connection
cap is briefly exceeded and Heroku drops the over-limit TLS handshakes, surfacing
to redis-py as a ``ConnectionError`` (``SSL: UNEXPECTED_EOF``). A cache read on a
user-facing request path must never turn that into a 500 — a blip should degrade
to a cache miss, not break the request (e.g. the navbar budget bar in
``core.spend.get_cached_budget_status`` must not fail a document upload).

This is the built-in ``RedisCache`` with a thin fail-open veneer, chosen over
switching to ``django-redis`` (IGNORE_EXCEPTIONS) so the already-verified Heroku
TLS config (``ssl_cert_reqs=CERT_NONE``), client, and serializer are untouched.

Per-method fallbacks are chosen so degradation is SAFE:

  - reads (``get`` / ``get_many`` / ``has_key``) -> miss (the caller recomputes);
  - ``add`` -> ``False`` — a lock "not acquired". The meetings live-presence lock
    (``cache.add``) therefore fails CLOSED under a blip (it refuses the second
    LIVE meeting) rather than handing out a duplicate lock;
  - ``set`` / ``touch`` / ``delete`` / ``*_many`` -> no-op.

``incr`` / ``decr`` / ``clear`` are intentionally left to the parent so genuine
errors on those still surface.
"""

from __future__ import annotations

import logging

from django.core.cache.backends.base import DEFAULT_TIMEOUT
from django.core.cache.backends.redis import RedisCache

try:
    from redis.exceptions import (
        ConnectionError as _RedisConnectionError,
        TimeoutError as _RedisTimeoutError,
    )

    _BLIP: tuple[type[BaseException], ...] = (_RedisConnectionError, _RedisTimeoutError)
except Exception:  # pragma: no cover - redis is always installed in this app
    _BLIP = ()

logger = logging.getLogger(__name__)


class ResilientRedisCache(RedisCache):
    """``RedisCache`` that swallows transient connection/timeout errors (fail-open)."""

    def _degraded(self, op: str, key=None) -> None:
        # INFO, not WARNING: this is a tolerated, expected condition under
        # connection pressure. Logging it at WARNING would re-flood Sentry with
        # the very events this backend exists to absorb (WARNING+ becomes a
        # Sentry event; INFO is only a breadcrumb). A genuine Redis outage still
        # surfaces via the broker/channels layers, which this does not touch.
        logger.info("cache %s degraded (redis unavailable) key=%s", op, key)

    def get(self, key, default=None, version=None):
        try:
            return super().get(key, default, version)
        except _BLIP:
            self._degraded("get", key)
            return default

    def get_many(self, keys, version=None):
        try:
            return super().get_many(keys, version)
        except _BLIP:
            self._degraded("get_many")
            return {}

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        try:
            return super().add(key, value, timeout, version)
        except _BLIP:
            self._degraded("add", key)
            return False  # "not acquired" — safe for lock-style callers

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        try:
            return super().set(key, value, timeout, version)
        except _BLIP:
            self._degraded("set", key)
            return None

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        try:
            return super().touch(key, timeout, version)
        except _BLIP:
            self._degraded("touch", key)
            return False

    def delete(self, key, version=None):
        try:
            return super().delete(key, version)
        except _BLIP:
            self._degraded("delete", key)
            return False

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        try:
            return super().set_many(data, timeout, version)
        except _BLIP:
            self._degraded("set_many")
            return list(data)  # all keys "failed to insert" (BaseCache contract)

    def delete_many(self, keys, version=None):
        try:
            return super().delete_many(keys, version)
        except _BLIP:
            self._degraded("delete_many")
            return None

    def has_key(self, key, version=None):
        try:
            return super().has_key(key, version)
        except _BLIP:
            self._degraded("has_key", key)
            return False

    def incr(self, key, delta=1, version=None):
        # ValueError (missing key) is NOT caught — django-ratelimit relies on it.
        # On a connection blip, fail open to a minimal count so rate limiting reads
        # "under limit" and allows the request (this is what makes the existing
        # RATELIMIT_FAIL_OPEN actually effective under a Redis outage).
        try:
            return super().incr(key, delta, version)
        except _BLIP:
            self._degraded("incr", key)
            return delta

    def decr(self, key, delta=1, version=None):
        try:
            return super().decr(key, delta, version)
        except _BLIP:
            self._degraded("decr", key)
            return -delta
