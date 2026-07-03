"""Shared client-side throttling utilities for outbound-API tools.

``brave_search`` and ``epo_ops`` each call a third-party API under its own rate
limit and retry/backoff policy. The *mechanism* (token bucket, backoff schedule,
env parsing) is identical, so it lives here; each tool keeps its own module-level
limiter instance so throttling stays independent per API.
"""

from __future__ import annotations

import threading
import time

# Retry/backoff knobs shared by the outbound-API tools.
MAX_RETRIES = 3
BACKOFF_BASE = 0.5
RATE_LIMIT_BACKOFF_SCHEDULE: list[float] = [5.0, 15.0, 30.0, 60.0]


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
