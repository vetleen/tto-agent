"""Unit tests for the per-thread turn lease (chat/thread_turn_lease.py)."""

import asyncio
import time
from unittest.mock import patch

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from chat import thread_turn_lease as ttl

LOCMEM = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache",
                      "LOCATION": "thread-turn-lease-tests"}}


@override_settings(CACHES=LOCMEM)
class ThreadTurnLeaseTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        patcher = patch.object(ttl, "POLL_INTERVAL_S", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_acquire_then_busy_then_release(self):
        first = ttl.try_acquire("t1", kind="chat", user_message_id="m1")
        self.assertIsNotNone(first)
        self.assertFalse(first.degraded)
        self.assertIsNone(ttl.try_acquire("t1", kind="chat"))
        self.assertEqual(ttl.current("t1")["token"], first.token)
        self.assertEqual(ttl.current("t1")["user_message_id"], "m1")
        # Another thread is independent.
        self.assertIsNotNone(ttl.try_acquire("t2", kind="chat"))

        ttl.release(first)
        self.assertIsNone(ttl.current("t1"))
        self.assertIsNotNone(ttl.try_acquire("t1", kind="loop"))

    def test_release_never_drops_a_successors_lease(self):
        first = ttl.try_acquire("t1", kind="chat")
        cache.delete(ttl._key("t1"))  # expired / evicted
        second = ttl.try_acquire("t1", kind="chat")
        ttl.release(first)
        self.assertEqual(ttl.current("t1")["token"], second.token)
        ttl.release(first)  # idempotent

    def test_refresh_extends_own_and_re_adds_a_vanished_lease(self):
        lease = ttl.try_acquire("t1", kind="chat")
        cache.delete(ttl._key("t1"))
        ttl.refresh(lease)
        self.assertEqual(ttl.current("t1")["token"], lease.token)

        other = ttl.ThreadTurnLease(thread_id="t1", token="foreign", value={"token": "foreign"})
        ttl.refresh(other)  # never overwrites the holder
        self.assertEqual(ttl.current("t1")["token"], lease.token)

    @override_settings(CHAT_THREAD_TURN_LEASE_SECONDS=1)
    def test_lease_expires_after_its_ttl(self):
        ttl.try_acquire("t1", kind="chat")
        time.sleep(1.2)
        self.assertIsNone(ttl.current("t1"))
        self.assertIsNotNone(ttl.try_acquire("t1", kind="chat"))

    def test_cache_failure_fails_open(self):
        from redis.exceptions import ConnectionError as RedisConnectionError

        with patch.object(cache, "add", side_effect=RedisConnectionError("blip")):
            lease = ttl.try_acquire("t1", kind="chat")
        self.assertTrue(lease.degraded)
        ttl.release(lease)  # no-op, no error

        # ResilientRedisCache style: add() -> False and get() -> None on a blip.
        with patch.object(cache, "add", return_value=False), \
                patch.object(cache, "get", return_value=None):
            lease = ttl.try_acquire("t1", kind="chat")
        self.assertTrue(lease.degraded)

    def test_current_ignores_junk(self):
        cache.set(ttl._key("t1"), "not-a-lease")
        self.assertIsNone(ttl.current("t1"))


@override_settings(CACHES=LOCMEM)
class WaitUntilFreeTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)
        patcher = patch.object(ttl, "POLL_INTERVAL_S", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_free_thread_is_acquired_at_once(self):
        lease = await ttl.wait_until_free("t1", kind="chat", timeout=1)
        self.assertIsNotNone(lease)
        self.assertFalse(lease.degraded)

    async def test_waits_for_the_holder_then_acquires(self):
        holder = ttl.try_acquire("t1", kind="chat")
        waiting = []

        async def on_waiting():
            waiting.append(True)

        task = asyncio.create_task(
            ttl.wait_until_free("t1", kind="chat", on_waiting=on_waiting, timeout=2)
        )
        await asyncio.sleep(0.1)
        self.assertFalse(task.done())
        self.assertEqual(waiting, [True])  # notified once
        await ttl.aio_release(holder)
        lease = await asyncio.wait_for(task, timeout=2)
        self.assertEqual(ttl.current("t1")["token"], lease.token)
        self.assertEqual(waiting, [True])

    async def test_should_stop_returns_none(self):
        ttl.try_acquire("t1", kind="chat")
        stop = {"now": False}
        task = asyncio.create_task(
            ttl.wait_until_free("t1", kind="chat", should_stop=lambda: stop["now"], timeout=2)
        )
        await asyncio.sleep(0.05)
        stop["now"] = True
        self.assertIsNone(await asyncio.wait_for(task, timeout=2))

    async def test_times_out(self):
        ttl.try_acquire("t1", kind="chat")
        with self.assertRaises(ttl.ThreadBusyTimeout):
            await ttl.wait_until_free("t1", kind="chat", timeout=0.05)

    async def test_keep_alive_refreshes_until_cancelled(self):
        lease = ttl.try_acquire("t1", kind="chat")
        with override_settings(CHAT_THREAD_TURN_REFRESH_SECONDS=0.01):
            task = asyncio.create_task(ttl.keep_alive(lease))
            cache.delete(ttl._key("t1"))
            await asyncio.sleep(0.1)
            self.assertEqual(ttl.current("t1")["token"], lease.token)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
