"""Shared client-side throttling utilities for outbound-API tools.

``brave_search`` and ``epo_ops`` each call a third-party API under its own rate
limit and retry/backoff policy. The *mechanism* (token bucket, sliding window,
backoff schedule, env parsing) lives here; each tool keeps its own module-level
limiter instance so throttling stays independent per API.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Callable

logger = logging.getLogger(__name__)

# Retry/backoff knobs shared by the outbound-API tools.
MAX_RETRIES = 3
BACKOFF_BASE = 0.5
RATE_LIMIT_BACKOFF_SCHEDULE: list[float] = [5.0, 15.0, 30.0, 60.0]


def deadline_capped_wait(seconds: float, context=None) -> tuple[float, bool]:
    """Compute a retry-backoff wait that never runs past the run deadline.

    Tool retry loops execute on worker threads with no cancel signal, so the run
    deadline (``RunContext.remaining_seconds()``) is the only way to interrupt a
    long rate-limit backoff. Returns ``(sleep_seconds, may_retry)``: how long to
    sleep (never past the deadline) and whether the caller may retry afterward
    (False once the deadline is reached — the caller should surface a terminal
    "temporarily unavailable" result). The caller performs the actual
    ``time.sleep`` so it stays observable/patchable in the tool module.
    ``context`` is duck-typed: any object exposing ``remaining_seconds()`` works;
    None means no deadline.
    """
    remaining = None
    fn = getattr(context, "remaining_seconds", None)
    if callable(fn):
        remaining = fn()
    if remaining is None:
        return seconds, True
    if remaining <= 0:
        return 0.0, False
    if seconds >= remaining:
        # Sleeping the full backoff would reach the deadline: wait out the
        # remaining budget, then stop.
        return remaining, False
    return seconds, True


def parse_rpm(env_value: str | None, default: int) -> int:
    """Parse a requests-per-minute value, falling back to *default*.

    Guards module import (a malformed value must not abort registration of the
    whole llm app) and the token-bucket divisor (0/negative would raise
    ZeroDivisionError on the second acquire).
    """
    try:
        rpm = int(env_value) if env_value is not None else default
    except (TypeError, ValueError):
        return default
    return rpm if rpm > 0 else default


class TokenBucketRateLimiter:
    """Process-wide token bucket that gates outgoing requests."""

    def __init__(self, requests_per_second: float, burst: int = 1):
        self._rps = requests_per_second
        self._max_tokens = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def set_rate(self, requests_per_second: float) -> None:
        """Change the refill rate in place (for advisory server-side throttling).

        Tokens accrued at the old rate are kept; the new rate applies from now.
        """
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._max_tokens,
                self._tokens + (now - self._last_refill) * self._rps,
            )
            self._last_refill = now
            self._rps = max(float(requests_per_second), 1e-6)

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + (now - self._last_refill) * self._rps,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rps
            time.sleep(wait)


# Reserve the next start slot in a sliding window, atomically, on Redis's clock
# (so the web and worker dynos share one window and one clock). KEYS[1] = ZSET of
# reserved start times; ARGV = limit, window seconds, unique member. Returns the
# wait in seconds as a string (Lua numbers would be truncated to integers).
_RESERVE_LUA = """
if redis.replicate_commands then pcall(redis.replicate_commands) end
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local n = redis.call('ZCARD', key)
local slot = now
if n >= limit then
  local kth = redis.call('ZRANGE', key, n - limit, n - limit, 'WITHSCORES')
  slot = math.max(slot, tonumber(kth[2]) + window)
end
if n > 0 then
  local last = redis.call('ZRANGE', key, -1, -1, 'WITHSCORES')
  slot = math.max(slot, tonumber(last[2]))
end
redis.call('ZADD', key, slot, ARGV[3])
redis.call('PEXPIRE', key, math.ceil((slot - now + 2 * window) * 1000))
return tostring(slot - now)
"""


class SlidingWindowScheduler:
    """At most ``limit`` request starts per ``window`` seconds, served FIFO.

    For APIs whose quota is "N requests per minute" (EPO OPS): a burst up to the
    quota goes out at once, the next request waits until the oldest start in the
    window has aged out. ``reserve`` books the caller's start slot atomically and
    returns how long to wait for it, so waiters never race (a later caller never
    gets an earlier slot) and each sleeps exactly once.

    Slots live in Redis when ``redis_client`` yields a client, shared by every
    process using the same key; on any Redis error (or without Redis) an
    in-process window with the same algorithm takes over.
    """

    def __init__(
        self,
        key_prefix: str,
        window: float = 61.0,
        *,
        redis_client: Callable[[], object | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.key_prefix = key_prefix
        self.window = float(window)
        self._redis_client = redis_client
        self._clock = clock
        self._lock = threading.Lock()
        self._slots: dict[str, deque[float]] = {}
        self._scripts: dict[int, object] = {}
        self._degraded = False

    def reserve(self, key: str, limit: int) -> float:
        """Book the next start slot for ``key``; return seconds to wait (>= 0)."""
        limit = max(1, int(limit))
        wait = self._reserve_redis(key, limit)
        if wait is None:
            wait = self._reserve_local(key, limit)
        return max(0.0, wait)

    def _reserve_redis(self, key: str, limit: int) -> float | None:
        if self._redis_client is None:
            return None
        try:
            client = self._redis_client()
            if client is None:
                return None
            script = self._scripts.get(id(client))
            if script is None:
                script = client.register_script(_RESERVE_LUA)
                self._scripts[id(client)] = script
            raw = script(keys=[f"{self.key_prefix}:{key}"], args=[limit, self.window, uuid.uuid4().hex])
            wait = float(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as e:
            if not self._degraded:
                # INFO: a Redis blip is tolerated (the local window takes over).
                logger.info("%s: shared rate window unavailable (%s); using the in-process one",
                            self.key_prefix, type(e).__name__)
                self._degraded = True
            return None
        if self._degraded:
            logger.info("%s: shared rate window back", self.key_prefix)
            self._degraded = False
        return wait

    def _reserve_local(self, key: str, limit: int) -> float:
        with self._lock:
            now = self._clock()
            slots = self._slots.setdefault(key, deque())
            while slots and slots[0] <= now - self.window:
                slots.popleft()
            slot = now
            if len(slots) >= limit:
                slot = max(slot, slots[len(slots) - limit] + self.window)
            if slots:
                slot = max(slot, slots[-1])
            slots.append(slot)
            return slot - now
