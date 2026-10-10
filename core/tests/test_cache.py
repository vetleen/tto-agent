"""Tests for the fail-open Redis cache backend (core.cache.ResilientRedisCache).

Regression cover for a production incident: a batch document upload saturated the
shared 20-connection Mini Redis, Heroku dropped the over-limit TLS handshakes
(``SSL: UNEXPECTED_EOF``), and the per-request budget-bar ``cache.get`` in
``core.spend.get_cached_budget_status`` raised ``ConnectionError`` — 500-ing ~11
uploads. The backend now degrades a blip to a cache miss instead.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from redis.exceptions import ConnectionError as RedisConnectionError

from core.cache import ResilientRedisCache

User = get_user_model()

_LOC = "redis://127.0.0.1:6379/1"
_PARENT = "django.core.cache.backends.redis.RedisCache"


class ResilientRedisCacheUnitTests(SimpleTestCase):
    """Each method fails open on a connection blip, with safe per-method defaults."""

    def setUp(self):
        self.cache = ResilientRedisCache(_LOC, {})

    def test_get_returns_default_on_blip(self):
        with patch(f"{_PARENT}.get", side_effect=RedisConnectionError("boom")):
            self.assertEqual(self.cache.get("k", "fallback"), "fallback")

    def test_get_many_empty_on_blip(self):
        with patch(f"{_PARENT}.get_many", side_effect=RedisConnectionError("boom")):
            self.assertEqual(self.cache.get_many(["a", "b"]), {})

    def test_add_fails_closed_on_blip(self):
        # A lock-style add must report "not acquired" (False), never a spurious True.
        with patch(f"{_PARENT}.add", side_effect=RedisConnectionError("boom")):
            self.assertIs(self.cache.add("lock", "owner"), False)

    def test_set_is_noop_on_blip(self):
        with patch(f"{_PARENT}.set", side_effect=RedisConnectionError("boom")):
            self.assertIsNone(self.cache.set("k", "v"))

    def test_delete_false_on_blip(self):
        with patch(f"{_PARENT}.delete", side_effect=RedisConnectionError("boom")):
            self.assertIs(self.cache.delete("k"), False)

    def test_has_key_false_on_blip(self):
        with patch(f"{_PARENT}.has_key", side_effect=RedisConnectionError("boom")):
            self.assertIs(self.cache.has_key("k"), False)

    def test_incr_fails_open_on_blip(self):
        # django-ratelimit's cache.incr must not 500 the request on a Redis blip;
        # it fails open to a minimal count so the rate-limit check reads under-limit.
        with patch(f"{_PARENT}.incr", side_effect=RedisConnectionError("boom")):
            self.assertEqual(self.cache.incr("rl:key"), 1)
            self.assertEqual(self.cache.incr("rl:key", 5), 5)

    def test_decr_fails_open_on_blip(self):
        with patch(f"{_PARENT}.decr", side_effect=RedisConnectionError("boom")):
            self.assertEqual(self.cache.decr("rl:key"), -1)

    def test_incr_value_error_still_propagates(self):
        # Missing-key ValueError is part of the incr contract django-ratelimit
        # depends on — it must NOT be swallowed as if it were a blip.
        with patch(f"{_PARENT}.incr", side_effect=ValueError("missing key")):
            with self.assertRaises(ValueError):
                self.cache.incr("rl:key")

    def test_non_blip_error_propagates(self):
        # A real bug (not a transient connection error) must NOT be swallowed.
        with patch(f"{_PARENT}.get", side_effect=ValueError("bug")):
            with self.assertRaises(ValueError):
                self.cache.get("k")


class SharedClientTests(SimpleTestCase):
    """Every thread's backend instance shares one client (one bounded pool) per
    process, so max_connections caps the process rather than each thread."""

    def test_instances_with_same_config_share_a_client(self):
        a = ResilientRedisCache(_LOC, {"OPTIONS": {"max_connections": 3}})
        b = ResilientRedisCache(_LOC, {"OPTIONS": {"max_connections": 3}})
        self.assertIs(a._cache, b._cache)

    def test_threads_share_the_client(self):
        import threading

        seen = []
        backends = [ResilientRedisCache(_LOC, {"OPTIONS": {"max_connections": 4}}) for _ in range(4)]
        threads = [threading.Thread(target=lambda c=c: seen.append(c._cache)) for c in backends]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len({id(c) for c in seen}), 1)

    def test_different_config_gets_its_own_client(self):
        a = ResilientRedisCache(_LOC, {"OPTIONS": {"max_connections": 3}})
        b = ResilientRedisCache("redis://127.0.0.1:6379/2", {"OPTIONS": {"max_connections": 3}})
        c = ResilientRedisCache(_LOC, {"OPTIONS": {"max_connections": 5}})
        self.assertIsNot(a._cache, b._cache)
        self.assertIsNot(a._cache, c._cache)


@override_settings(
    CACHES={"default": {"BACKEND": "core.cache.ResilientRedisCache", "LOCATION": _LOC}},
    BUDGET_STATUS_CACHE_SECONDS=60,  # force the cached path (0 would bypass the cache)
)
class BudgetStatusCacheBlipTests(TestCase):
    """The navbar budget lookup must survive a cache blip instead of 500-ing a request."""

    def test_get_cached_budget_status_survives_blip(self):
        from core.spend import get_cached_budget_status

        user = User.objects.create_user(email="blip@example.com", password="pw")
        with patch(f"{_PARENT}.get", side_effect=RedisConnectionError("boom")), patch(
            f"{_PARENT}.set", side_effect=RedisConnectionError("boom")
        ):
            # Must not raise: the cache read degrades to a miss and the value is
            # recomputed directly.
            result = get_cached_budget_status(user)
        self.assertTrue(result is None or isinstance(result, dict))
