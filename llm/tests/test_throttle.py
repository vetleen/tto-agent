"""Tests for shared tool throttling helpers (llm/tools/_throttle.py)."""

import time
import uuid
from unittest import TestCase, skipUnless
from unittest.mock import MagicMock

from llm.tools._throttle import SlidingWindowScheduler, TokenBucketRateLimiter, deadline_capped_wait
from llm.types.context import RunContext


class SetRateTests(TestCase):
    def test_set_rate_changes_the_refill_rate_in_place(self):
        limiter = TokenBucketRateLimiter(requests_per_second=0.001, burst=1)
        limiter.acquire()  # spends the only token; the next one is ~1000 s away
        limiter.set_rate(1000.0)
        start = time.monotonic()
        limiter.acquire()
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(limiter._rps, 1000.0)

    def test_set_rate_never_zero(self):
        limiter = TokenBucketRateLimiter(requests_per_second=1.0, burst=1)
        limiter.set_rate(0)
        self.assertGreater(limiter._rps, 0)


class _Ctx:
    """Minimal duck-typed context exposing remaining_seconds()."""

    def __init__(self, remaining):
        self._remaining = remaining

    def remaining_seconds(self):
        return self._remaining


class DeadlineCappedWaitTests(TestCase):
    def test_no_context_sleeps_full_and_may_retry(self):
        self.assertEqual(deadline_capped_wait(5.0, None), (5.0, True))

    def test_context_without_deadline_sleeps_full(self):
        # A RunContext with no deadline_seconds -> remaining_seconds() is None.
        self.assertEqual(deadline_capped_wait(5.0, RunContext.create()), (5.0, True))

    def test_deadline_well_beyond_wait(self):
        self.assertEqual(deadline_capped_wait(5.0, _Ctx(100.0)), (5.0, True))

    def test_wait_exceeds_deadline_caps_and_stops(self):
        # A 30s backoff with only 10s of run budget: sleep out the 10s and stop.
        self.assertEqual(deadline_capped_wait(30.0, _Ctx(10.0)), (10.0, False))

    def test_deadline_already_passed_no_sleep_and_stops(self):
        self.assertEqual(deadline_capped_wait(30.0, _Ctx(-1.0)), (0.0, False))


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class SlidingWindowSchedulerTests(TestCase):
    """In-process path (no Redis client): the algorithm itself."""

    def _sched(self, window=60.0):
        clock = _Clock()
        return SlidingWindowScheduler("t", window=window, clock=clock), clock

    def test_a_burst_up_to_the_limit_is_immediate(self):
        s, _ = self._sched()
        self.assertEqual([s.reserve("search", 5) for _ in range(5)], [0.0] * 5)

    def test_the_next_waits_until_the_oldest_leaves_the_window(self):
        s, clock = self._sched()
        for _ in range(5):
            s.reserve("search", 5)
        clock.t += 10
        self.assertAlmostEqual(s.reserve("search", 5), 50.0)  # oldest (t0) + 60 - now (t0+10)

    def test_queued_callers_are_served_in_arrival_order(self):
        s, _ = self._sched()
        waits = [s.reserve("search", 2) for _ in range(6)]
        # Two per window: 0, 0, 60, 60, 120, 120 — never earlier than the caller before.
        self.assertEqual(waits, [0.0, 0.0, 60.0, 60.0, 120.0, 120.0])
        self.assertEqual(waits, sorted(waits))

    def test_window_frees_up_after_it_passes(self):
        s, clock = self._sched()
        for _ in range(3):
            s.reserve("search", 3)
        clock.t += 61
        self.assertEqual(s.reserve("search", 3), 0.0)

    def test_a_lower_limit_waits_for_the_window_to_drain(self):
        s, clock = self._sched()
        for _ in range(10):
            s.reserve("search", 15)  # busy: 10 starts at t0
        clock.t += 5
        # Overloaded now (5/min): 10 starts are in the window, so wait until only 4 remain.
        self.assertAlmostEqual(s.reserve("search", 5), 55.0)

    def test_services_have_separate_windows(self):
        s, _ = self._sched()
        s.reserve("search", 1)
        self.assertEqual(s.reserve("retrieval", 1), 0.0)
        self.assertEqual(s.reserve("search", 1), 60.0)

    def test_concurrent_callers_never_share_a_slot_beyond_the_limit(self):
        import threading

        s = SlidingWindowScheduler("t", window=60.0)
        waits = []
        lock = threading.Lock()

        def go():
            w = s.reserve("search", 4)
            with lock:
                waits.append(w)

        threads = [threading.Thread(target=go) for _ in range(12)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(sum(1 for w in waits if w < 1), 4)
        self.assertEqual(sum(1 for w in waits if 59 < w < 61), 4)
        self.assertEqual(sum(1 for w in waits if 119 < w < 121), 4)


class SlidingWindowRedisTests(TestCase):
    def test_redis_error_falls_back_to_the_local_window(self):
        def broken():
            raise ConnectionError("down")

        s = SlidingWindowScheduler("t", window=60.0, redis_client=broken)
        with self.assertLogs("llm.tools._throttle", level="INFO") as cm:
            self.assertEqual(s.reserve("search", 1), 0.0)
            self.assertGreater(s.reserve("search", 1), 59.0)  # local window still enforces it
        self.assertEqual(len(cm.records), 1)  # logged once per blip, not per call

    def test_no_client_uses_the_local_window(self):
        s = SlidingWindowScheduler("t", window=60.0, redis_client=lambda: None)
        self.assertEqual(s.reserve("search", 1), 0.0)
        self.assertGreater(s.reserve("search", 1), 59.0)

    def test_script_result_is_the_wait(self):
        client = MagicMock()
        client.register_script.return_value = MagicMock(return_value=b"12.5")
        s = SlidingWindowScheduler("p", window=61.0, redis_client=lambda: client)
        self.assertEqual(s.reserve("search", 5), 12.5)
        script = client.register_script.return_value
        kwargs = script.call_args.kwargs
        self.assertEqual(kwargs["keys"], ["p:search"])
        self.assertEqual(kwargs["args"][:2], [5, 61.0])
        s.reserve("search", 5)
        client.register_script.assert_called_once()  # registered once per client


def _local_redis():
    try:
        import redis

        client = redis.Redis.from_url("redis://127.0.0.1:6379/15", socket_connect_timeout=0.5)
        client.ping()
        return client
    except Exception:
        return None


_REDIS = _local_redis()


@skipUnless(_REDIS is not None, "local Redis not reachable")
class SlidingWindowRealRedisTests(TestCase):
    """The Lua script against a real Redis (DB 15, own key)."""

    def setUp(self):
        self.prefix = f"wf_test_slots:{uuid.uuid4().hex}"
        self.addCleanup(lambda: [_REDIS.delete(k) for k in _REDIS.keys(self.prefix + ":*")])

    def test_burst_then_queue_in_order(self):
        s = SlidingWindowScheduler(self.prefix, window=60.0, redis_client=lambda: _REDIS)
        waits = [s.reserve("search", 3) for _ in range(5)]
        self.assertEqual([w < 0.5 for w in waits], [True, True, True, False, False])
        self.assertTrue(59 < waits[3] < 60.5)
        self.assertGreaterEqual(waits[4], waits[3])

    def test_two_schedulers_share_the_window(self):
        # Two processes (web + worker) = two scheduler objects on one key.
        a = SlidingWindowScheduler(self.prefix, window=60.0, redis_client=lambda: _REDIS)
        b = SlidingWindowScheduler(self.prefix, window=60.0, redis_client=lambda: _REDIS)
        self.assertLess(a.reserve("search", 2), 0.5)
        self.assertLess(b.reserve("search", 2), 0.5)
        self.assertGreater(a.reserve("search", 2), 59.0)

    def test_key_expires(self):
        s = SlidingWindowScheduler(self.prefix, window=60.0, redis_client=lambda: _REDIS)
        s.reserve("search", 3)
        ttl = _REDIS.pttl(f"{self.prefix}:search")
        self.assertTrue(0 < ttl <= 120_000)
