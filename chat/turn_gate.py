"""In-process admission control for interactive chat turns.

This is the web dyno's chat-turn queue. A turn holds a slot for exactly as long
as its streaming task is alive: the slot is taken inside
``ChatConsumer._stream_and_finalize`` and released in that task's ``finally``, so
a stop, a disconnect, an exception, or a dyno restart can never leave a phantom
slot behind. That is the whole reason the tally lives in this process rather than
in Redis — there is no state here that can outlive the task owning it, so the
gate cannot wrongly tell a user they are over the limit.

A turn over a cap is **queued, never refused**. It waits in line and starts by
itself when a slot frees.

Distinct from two neighbouring limits:

* ``chat.subagent_limits`` — worker-side, DB-backed, hard deny at admission.
* ``LLM_MAX_CONCURRENT_STREAMS`` (``llm/service/llm_service.py``) — a
  provider-connection cap one level down. This gate is always acquired *outside*
  it and nothing inside it ever waits on this gate, so the two cannot deadlock.

Only the interactive WebSocket path is gated. Headless loop turns reach the turn
machinery through ``run_turn_to_completion`` on the Celery worker and never call
``_stream_and_finalize``; belt and braces, ``async_to_sync`` gives them a fresh
event loop and therefore a fresh, empty gate (see ``_gate``).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to *default* on empty or invalid input.

    Heroku's ``config:set`` has been observed writing empty strings (see
    RUNBOOK.md), and a bare ``int("")`` would crash the process at import.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        logger.warning(
            "turn_gate: %s=%r is not an integer; using %d", name, raw, default,
        )
        return default


# Concurrent streaming turns allowed to one user. Because a new message on a
# socket cancels that socket's previous turn, this is effectively a cap on how
# many of one user's TABS may stream at once. Over the cap the turn queues.
# 0 disables.
CHAT_TURN_MAX_PER_USER = _env_int("CHAT_TURN_MAX_PER_USER", 2)
# Concurrent streaming turns allowed in this process. This is the web dyno's
# memory and API-burst lever. Over the cap the turn queues. 0 disables.
CHAT_TURN_MAX_CONCURRENT = _env_int("CHAT_TURN_MAX_CONCURRENT", 8)
# How long a turn may wait for a slot before giving up, in seconds. 0 waits
# forever. Keep this at or under SUBAGENT_REPORT_LEASE_MINUTES * 60 (300s, see
# ChatConsumer) so a timed-out sub-agent continuation is still re-claimed by the
# watchdog rather than being dropped.
CHAT_TURN_QUEUE_TIMEOUT_SECONDS = _env_int("CHAT_TURN_QUEUE_TIMEOUT_SECONDS", 300)


class TurnQueueTimeout(Exception):
    """A turn waited longer than the queue timeout for a slot."""


# Ticket lifecycle. Ownership of a slot lives HERE and never in the return value
# of an await — see the cancellation argument on _pump/_finish.
_QUEUED = "queued"
_HELD = "held"
_DONE = "done"


class _Ticket:
    """One turn's place in the gate."""

    __slots__ = ("seq", "user_key", "future", "state", "enqueued_at", "position", "reason")

    def __init__(self, seq: int, user_key: str) -> None:
        self.seq = seq
        self.user_key = user_key
        self.future: asyncio.Future | None = None
        self.state = _QUEUED
        self.enqueued_at = 0.0
        self.position = 0
        self.reason: str | None = None


class TurnSlot:
    """Yielded by :func:`slot` — what happened, for the caller's log line."""

    __slots__ = ("queued", "position", "reason", "wait_seconds")

    def __init__(self, queued: bool, position: int, reason: str | None) -> None:
        self.queued = queued
        self.position = position
        self.reason = reason
        self.wait_seconds = 0.0


class _TurnGate:
    """Tallies and the waiting line for one event loop.

    Every method that mutates ``_active_total``, ``_active_user``, ``_waiters``
    or a ticket's ``state`` is synchronous and contains no ``await``, so they are
    atomic with respect to each other on the single event-loop thread. That is
    what makes an ``asyncio.Lock`` unnecessary here — and adding one would be
    actively harmful, since it would introduce suspension points into the
    critical section and reintroduce the interleaving this design removes.
    """

    def __init__(self) -> None:
        self._active_total = 0
        self._active_user: dict[str, int] = {}
        # Insertion-ordered, so iteration order is arrival order (FIFO).
        self._waiters: dict[int, _Ticket] = {}
        self._seq = itertools.count()

    # -- capacity ---------------------------------------------------------

    def _system_full(self) -> bool:
        cap = CHAT_TURN_MAX_CONCURRENT  # module global, read on every call
        return cap > 0 and self._active_total >= cap

    def _user_full(self, user_key: str) -> bool:
        cap = CHAT_TURN_MAX_PER_USER  # module global, read on every call
        return cap > 0 and self._active_user.get(user_key, 0) >= cap

    def _blocked_reason(self, user_key: str) -> str | None:
        """Why this user cannot start right now, or None if they can.

        Per-user is checked first: when both caps are hit it is the more
        actionable thing to tell the user about.
        """
        if self._user_full(user_key):
            return "user"
        if self._system_full():
            return "system"
        return None

    # -- tallies ----------------------------------------------------------

    def _take(self, user_key: str) -> None:
        self._active_total += 1
        self._active_user[user_key] = self._active_user.get(user_key, 0) + 1

    def _give_back(self, user_key: str) -> None:
        self._active_total -= 1
        remaining = self._active_user.get(user_key, 0) - 1
        if remaining > 0:
            self._active_user[user_key] = remaining
        else:
            # Drop the key so _active_user cannot grow without bound.
            self._active_user.pop(user_key, None)
        if self._active_total < 0:
            # An invariant break must be loud, never silent.
            logger.error("turn_gate: active_total went negative; clamping to 0")
            self._active_total = 0

    # -- admission --------------------------------------------------------

    def _enqueue(self, user_key: str) -> _Ticket:
        """Take a slot outright, or join the line. Never awaits."""
        ticket = _Ticket(next(self._seq), user_key)
        reason = self._blocked_reason(user_key)
        if reason is None:
            # Uncontended fast path: no future is created at all.
            ticket.state = _HELD
            self._take(user_key)
            return ticket
        ticket.state = _QUEUED
        ticket.reason = reason
        ticket.future = asyncio.get_running_loop().create_future()
        ticket.enqueued_at = time.monotonic()
        self._waiters[ticket.seq] = ticket
        ticket.position = len(self._waiters)  # 1-based, counting self
        return ticket

    def _pump(self) -> None:
        """Admit every waiter capacity allows, in arrival order. Never awaits."""
        for seq, ticket in list(self._waiters.items()):
            if self._system_full():
                break  # no capacity for anyone
            if ticket.future is not None and ticket.future.done():
                # Cancelled or expired while queued. Drop it BEFORE touching any
                # tally, so a dead ticket can never consume a slot.
                self._waiters.pop(seq, None)
                continue
            if self._user_full(ticket.user_key):
                # The no-head-of-line rule: skip this one and keep scanning, so
                # a user sitting at their own cap never blocks anybody else.
                # The skipped ticket keeps its place for the next pump.
                continue
            self._waiters.pop(seq, None)
            # Order matters. Ownership is transferred (state), then the tally is
            # updated, and only then is the waiter woken. A cancellation landing
            # between any of these is handled by _finish reading `state`.
            ticket.state = _HELD
            self._take(ticket.user_key)
            ticket.future.set_result(True)

    def _expire(self, ticket: _Ticket) -> None:
        """Deadline callback. Only ever fires a ticket that is STILL in line.

        Runs as a ``call_later`` callback on the same thread as ``_pump``, so one
        of the two always wins cleanly: if the slot was already granted the state
        is no longer ``_QUEUED`` and this is a no-op. That is what makes a false
        "busy" impossible.
        """
        if ticket.state != _QUEUED:
            return
        ticket.state = _DONE
        self._waiters.pop(ticket.seq, None)
        if ticket.future is not None and not ticket.future.done():
            # A result, not an exception: if the task is cancelled before it
            # resumes there is no "exception was never retrieved" noise.
            ticket.future.set_result(False)

    def _finish(self, ticket: _Ticket) -> None:
        """Release exactly once, whatever happened. Never awaits.

        Reads ``ticket.state`` rather than the outcome of the caller's await,
        which is what closes the classic semaphore leak: a waiter admitted by
        ``_pump`` and then cancelled before it resumes is still ``_HELD`` here,
        so its slot is handed on instead of being lost.
        """
        if ticket.state == _HELD:
            ticket.state = _DONE
            self._give_back(ticket.user_key)
        elif ticket.state == _QUEUED:
            # Never counted, so there is nothing to give back.
            ticket.state = _DONE
            self._waiters.pop(ticket.seq, None)
        else:
            return  # already done; idempotent
        self._pump()

    async def _wait(self, ticket: _Ticket, timeout: float) -> None:
        """Block until admitted, or raise :class:`TurnQueueTimeout`.

        Deliberately not ``asyncio.wait_for``: that cancels the *task*, so a
        deadline landing in the same loop iteration as ``_pump``'s
        ``set_result`` would raise ``TimeoutError`` on a turn that had in fact
        just been granted a slot. A ``call_later`` timer cannot do that.
        """
        handle = None
        if timeout and timeout > 0:
            handle = asyncio.get_running_loop().call_later(
                timeout, self._expire, ticket,
            )
        try:
            admitted = await ticket.future
        finally:
            if handle is not None:
                handle.cancel()
        if not admitted:
            raise TurnQueueTimeout("Timed out waiting for a chat turn slot")

    def _drain_for_reset(self) -> None:
        """Resolve every waiter as timed-out and zero the tallies (tests only)."""
        for ticket in list(self._waiters.values()):
            ticket.state = _DONE
            if ticket.future is not None and not ticket.future.done():
                try:
                    ticket.future.set_result(False)
                except RuntimeError:
                    # Loop already closed; the waiting task died with it.
                    pass
        self._waiters.clear()
        self._active_total = 0
        self._active_user.clear()


# One gate per event loop, never a module-level singleton.
#
# NOTE (mirrors llm_service._get_stream_semaphore): asyncio primitives bind to a
# loop. Daphne runs one loop per web process, so in production this resolves to a
# single gate. Keying by loop means a loop that goes away takes its tallies with
# it, so a leaked count can never reach a later turn — in tests, where each async
# test method gets a fresh loop, or under async_to_sync on the worker, which gets
# a fresh loop per call and is therefore ungated by construction.
_gates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _gate() -> _TurnGate:
    """The gate for the running loop, created on first use."""
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _TurnGate()
        _gates[loop] = gate
    return gate


@asynccontextmanager
async def slot(user_id, *, on_queued=None, timeout=None):
    """Hold one chat-turn slot for the duration of the block.

    ``on_queued(reason, position)`` is awaited once and only if the turn actually
    has to wait; ``reason`` is ``"user"`` or ``"system"``. ``timeout`` defaults to
    ``CHAT_TURN_QUEUE_TIMEOUT_SECONDS`` (0 waits forever) and raises
    :class:`TurnQueueTimeout` when it expires.

    The slot is released on every exit path, including cancellation, because the
    release lives in this function's ``finally``.
    """
    gate = _gate()
    user_key = str(user_id)
    if timeout is None:
        timeout = CHAT_TURN_QUEUE_TIMEOUT_SECONDS

    ticket = gate._enqueue(user_key)
    info = TurnSlot(
        queued=(ticket.state == _QUEUED),
        position=ticket.position,
        reason=ticket.reason,
    )
    try:
        if ticket.state == _QUEUED:
            logger.info(
                "turn_gate: queued user=%s reason=%s position=%d active=%d waiting=%d",
                user_key, ticket.reason, ticket.position,
                gate._active_total, len(gate._waiters),
            )
            if on_queued is not None:
                try:
                    await on_queued(ticket.reason, ticket.position)
                except Exception:
                    # Never let a notification failure affect admission.
                    logger.warning(
                        "turn_gate: on_queued callback failed", exc_info=True,
                    )
            await gate._wait(ticket, timeout)
            info.wait_seconds = time.monotonic() - ticket.enqueued_at
            logger.info(
                "turn_gate: admitted user=%s after %.1fs", user_key, info.wait_seconds,
            )
        yield info
    finally:
        gate._finish(ticket)


def stats() -> dict:
    """Current occupancy for the running loop. Diagnostics and tests."""
    try:
        gate = _gate()
    except RuntimeError:  # no running loop
        return {"active": 0, "waiting": 0, "per_user": {}}
    return {
        "active": gate._active_total,
        "waiting": len(gate._waiters),
        "per_user": dict(gate._active_user),
    }


def pump() -> None:
    """Re-run admission for the running loop.

    Only needed by tests that raise a cap while turns are already queued: in
    production caps change at restart, and every release pumps.
    """
    try:
        gate = _gate()
    except RuntimeError:
        return
    gate._pump()


def reset() -> None:
    """Drop all gate state for every loop. Tests only.

    Outstanding waiters are resolved as timed-out first so a leftover task
    unwinds instead of hanging. Safe to call from a synchronous ``setUp`` where
    there is no running loop.
    """
    for gate in list(_gates.values()):
        gate._drain_for_reset()
    _gates.clear()
