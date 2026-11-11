"""One chat turn at a time per thread, across processes.

The composer lock is per browser tab. After a reload, in a second tab or on a
second device, the page does not know a turn is still running on the server (a
turn whose socket closed finishes *detached*), so the user could start a second
turn on the same thread: it ran concurrently, without the first turn's reply in
its history, and the two replies landed out of order. Headless loop turns (on
the Celery worker) could overlap a user's turn the same way.

Every turn now takes a short lease on its thread before it assembles its inputs
and holds it until its reply is saved; a turn that finds the lease taken waits
(``wait_until_free``) and then runs with the finished reply in its history.

The lease lives in the Django cache — the dedicated cache Redis in production
(``core.cache.ResilientRedisCache``), so no channel-layer connections — keyed
per thread, with an owner token, a TTL and a keep-alive refresh, like the
meetings live-presence lock (``meetings/consumers.py``). A crashed holder frees
the thread when the TTL runs out.

**Fail-open.** Unlike the meetings lock, a cache problem must never block a user:
if the cache cannot answer, the caller gets a ``degraded`` lease and proceeds as
before this module existed (turns may overlap — today's behaviour).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from asgiref.sync import sync_to_async

logger = logging.getLogger(__name__)

KEY_PREFIX = "chatturn:v1:"

# Seconds between lease checks while a turn waits (patched to ~0 in tests).
POLL_INTERVAL_S = 1.0


class ThreadBusyTimeout(Exception):
    """Another turn held the thread for longer than CHAT_THREAD_TURN_WAIT_SECONDS."""


@dataclass
class ThreadTurnLease:
    thread_id: str
    token: str
    value: dict = field(default_factory=dict)
    # The cache could not answer: the turn proceeds unserialized (fail-open).
    degraded: bool = False
    released: bool = False
    # Acquired only after waiting for another turn (wait_until_free).
    waited: bool = False


def _key(thread_id) -> str:
    return f"{KEY_PREFIX}{thread_id}"


def _setting(name: str, default: int) -> float:
    from django.conf import settings

    try:
        return float(getattr(settings, name, default))
    except (TypeError, ValueError):
        return float(default)


def lease_ttl() -> float:
    return _setting("CHAT_THREAD_TURN_LEASE_SECONDS", 120)


def refresh_interval() -> float:
    return _setting("CHAT_THREAD_TURN_REFRESH_SECONDS", 30)


def wait_timeout() -> float:
    return _setting("CHAT_THREAD_TURN_WAIT_SECONDS", 300)


def _cache():
    from django.core.cache import cache

    return cache


def _holder_token(holder) -> str | None:
    return holder.get("token") if isinstance(holder, dict) else None


def try_acquire(thread_id, *, kind: str, user_message_id=None) -> ThreadTurnLease | None:
    """Take the thread's lease. ``None`` = another turn holds it (busy).

    Never raises: a cache that cannot answer yields a ``degraded`` lease.
    """
    thread_id = str(thread_id)
    token = uuid.uuid4().hex
    value = {
        "token": token,
        "kind": kind,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "user_message_id": str(user_message_id) if user_message_id else None,
        "pid": os.getpid(),
    }
    lease = ThreadTurnLease(thread_id=thread_id, token=token, value=value)
    cache = _cache()
    key = _key(thread_id)
    ttl = lease_ttl()
    try:
        # Two attempts: a holder can expire between our add() and get().
        for _ in range(2):
            if cache.add(key, value, ttl):
                return lease
            holder = cache.get(key)
            if _holder_token(holder):
                return None
        # add() refused, yet nobody holds it: the cache is not answering
        # (ResilientRedisCache turns a blip into add=False / get=None).
    except Exception:
        logger.info("thread turn lease: cache unavailable thread=%s", thread_id, exc_info=True)
    lease.degraded = True
    return lease


def current(thread_id) -> dict | None:
    """The lease value of the turn running on ``thread_id``, or ``None``."""
    try:
        holder = _cache().get(_key(thread_id))
    except Exception:
        return None
    return holder if _holder_token(holder) else None


def refresh(lease: ThreadTurnLease) -> None:
    """Extend our own lease (re-add it if it vanished); never touch a foreign one."""
    if lease is None or lease.degraded or lease.released:
        return
    cache = _cache()
    key = _key(lease.thread_id)
    try:
        holder = cache.get(key)
        if _holder_token(holder) == lease.token:
            cache.set(key, lease.value, lease_ttl())
        elif holder is None:
            # Evicted or lost in a blip while our turn still runs: take it back
            # (atomic — a turn that grabbed it in between keeps it).
            cache.add(key, lease.value, lease_ttl())
    except Exception:
        logger.info("thread turn lease: refresh failed thread=%s", lease.thread_id, exc_info=True)


def release(lease: ThreadTurnLease) -> None:
    """Drop our own lease. Idempotent; never deletes a successor's lease."""
    if lease is None or lease.released:
        return
    lease.released = True
    if lease.degraded:
        return
    cache = _cache()
    key = _key(lease.thread_id)
    try:
        if _holder_token(cache.get(key)) == lease.token:
            cache.delete(key)
    except Exception:
        logger.info("thread turn lease: release failed thread=%s", lease.thread_id, exc_info=True)


# Cache calls go to a worker thread, never the thread-sensitive executor the
# consumer's DB work shares: the cache client is thread-safe and process-shared.
_aio_try_acquire = sync_to_async(try_acquire, thread_sensitive=False)
_aio_refresh = sync_to_async(refresh, thread_sensitive=False)
aio_release = sync_to_async(release, thread_sensitive=False)
aio_current = sync_to_async(current, thread_sensitive=False)


async def keep_alive(lease: ThreadTurnLease) -> None:
    """Refresh ``lease`` until cancelled (run as a task while the turn holds it)."""
    if lease is None or lease.degraded:
        return
    try:
        while not lease.released:
            await asyncio.sleep(refresh_interval())
            await _aio_refresh(lease)
    except asyncio.CancelledError:
        pass


def _release_if_taken(future: asyncio.Future) -> None:
    """Done-callback: give back a lease a cancelled waiter's cache call still took."""
    if future.cancelled() or future.exception() is not None:
        return
    lease = future.result()
    if lease is not None:
        asyncio.get_running_loop().run_in_executor(None, release, lease)


async def _try_acquire_cancel_safe(thread_id, *, kind, user_message_id=None):
    """``try_acquire`` in a worker thread, safe against the caller's cancellation.

    The cache call can't be interrupted: a Stop that cancels the waiting turn
    mid-``cache.add`` would leave a lease in the cache that no one holds, keeping
    the thread locked for the full TTL (staging, 2026-11-11). On cancellation the
    call runs to the end and whatever it took is released.
    """
    future = asyncio.ensure_future(
        _aio_try_acquire(thread_id, kind=kind, user_message_id=user_message_id)
    )
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        future.add_done_callback(_release_if_taken)
        raise


async def wait_until_free(
    thread_id,
    *,
    kind: str,
    user_message_id=None,
    should_stop=None,
    on_waiting=None,
    timeout: float | None = None,
) -> ThreadTurnLease | None:
    """Acquire the thread's lease, waiting while another turn holds it.

    Returns the lease, or ``None`` when ``should_stop()`` turns true while
    waiting (the turn was stopped or superseded). ``on_waiting`` is awaited once,
    the first time the lease is found busy. Raises ``ThreadBusyTimeout`` after
    ``timeout`` seconds (default CHAT_THREAD_TURN_WAIT_SECONDS).
    """
    timeout = wait_timeout() if timeout is None else timeout
    deadline = time.monotonic() + timeout
    notified = False
    while True:
        if should_stop is not None and should_stop():
            return None
        lease = await _try_acquire_cancel_safe(thread_id, kind=kind, user_message_id=user_message_id)
        if lease is not None:
            if notified:
                lease.waited = True
                logger.info("thread turn lease: acquired after waiting thread=%s kind=%s", thread_id, kind)
            return lease
        if not notified:
            notified = True
            logger.info("thread turn lease: busy, waiting thread=%s kind=%s", thread_id, kind)
            if on_waiting is not None:
                await on_waiting()
        if time.monotonic() >= deadline:
            raise ThreadBusyTimeout(str(thread_id))
        await asyncio.sleep(POLL_INTERVAL_S)
