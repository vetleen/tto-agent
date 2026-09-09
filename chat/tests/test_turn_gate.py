"""Unit tests for the in-process chat-turn concurrency gate.

Pure asyncio, no DB and no sockets, so these run in about a second. The caps are
module-level ``os.environ`` reads, so they are set here with
``@patch("chat.turn_gate.<CONST>", n)`` method decorators — the same convention
``chat/tests/test_subagent_dispatch.py`` uses for ``SUBAGENT_WORKER_SLOTS``.

Several tests reach into ``turn_gate._gate()`` internals on purpose: the
interesting failure modes (a waiter cancelled in the window between being
admitted and resuming, a deadline racing an admission) are exactly the ones that
cannot be provoked deterministically through the public API alone.
"""

import asyncio
import time
from unittest.mock import patch

from django.test import SimpleTestCase

from chat import turn_gate


async def _until(predicate, timeout=2.0):
    """Poll a Python-side condition rather than sleeping a fixed time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met within %.1fs" % timeout)


class TurnGateTests(SimpleTestCase):
    def setUp(self):
        # Each async test method runs on its own event loop, so the gate is
        # already fresh; reset anyway so a failure can never leak sideways.
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)

    # -- fast path --------------------------------------------------------

    async def test_uncontended_acquires_immediately(self):
        queued_calls = []

        async def on_queued(reason, position):
            queued_calls.append((reason, position))

        async with turn_gate.slot("u1", on_queued=on_queued) as info:
            self.assertFalse(info.queued)
            self.assertIsNone(info.reason)
            self.assertEqual(turn_gate.stats()["active"], 1)
            # No future is created at all on this path.
            gate = turn_gate._gate()
            self.assertEqual(gate._waiters, {})

        self.assertEqual(queued_calls, [])
        self.assertEqual(turn_gate.stats(), {"active": 0, "waiting": 0, "per_user": {}})

    # -- the two caps -----------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_system_cap_queues_the_extra_turn(self):
        seen = []
        release = asyncio.Event()

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def waiter():
            async def on_queued(reason, position):
                seen.append((reason, position))

            async with turn_gate.slot("u2", on_queued=on_queued) as info:
                seen.append(("admitted", info.reason))

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        self.assertEqual(seen, [("system", 1)])
        release.set()
        await asyncio.gather(h, w)
        self.assertEqual(seen[-1], ("admitted", "system"))
        self.assertEqual(turn_gate.stats()["active"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 8)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    async def test_per_user_cap_queues_the_same_user(self):
        seen = []
        release = asyncio.Event()

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def waiter():
            async def on_queued(reason, position):
                seen.append((reason, position))

            async with turn_gate.slot("u1", on_queued=on_queued):
                seen.append(("admitted", None))

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        self.assertEqual(seen, [("user", 1)])
        release.set()
        await asyncio.gather(h, w)
        self.assertEqual(turn_gate.stats()["active"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 4)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    async def test_per_user_block_does_not_block_other_users(self):
        """The no-head-of-line rule.

        A waiter parked on its own per-user cap must not hold up a different
        user who has capacity.
        """
        release = asyncio.Event()
        started = []

        async def turn(user, label):
            async with turn_gate.slot(user):
                started.append(label)
                await release.wait()

        a1 = asyncio.create_task(turn("A", "a1"))
        await _until(lambda: started == ["a1"])
        a2 = asyncio.create_task(turn("A", "a2"))  # blocked by A's own cap
        await _until(lambda: turn_gate.stats()["waiting"] == 1)
        b1 = asyncio.create_task(turn("B", "b1"))  # must jump the parked A

        await _until(lambda: "b1" in started)
        self.assertEqual(started, ["a1", "b1"])
        self.assertEqual(turn_gate.stats()["waiting"], 1)

        release.set()
        await asyncio.gather(a1, a2, b1)
        self.assertEqual(turn_gate.stats()["active"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_release_admits_the_oldest_eligible_waiter(self):
        started = []
        gate_events = {}

        async def turn(label):
            ev = asyncio.Event()
            gate_events[label] = ev
            async with turn_gate.slot(label):
                started.append(label)
                await ev.wait()

        first = asyncio.create_task(turn("one"))
        await _until(lambda: started == ["one"])
        second = asyncio.create_task(turn("two"))
        await _until(lambda: turn_gate.stats()["waiting"] == 1)
        third = asyncio.create_task(turn("three"))
        await _until(lambda: turn_gate.stats()["waiting"] == 2)

        gate_events["one"].set()
        await _until(lambda: "two" in started)
        self.assertEqual(started, ["one", "two"])  # FIFO, not LIFO

        gate_events["two"].set()
        await _until(lambda: "three" in started)
        gate_events["three"].set()
        await asyncio.gather(first, second, third)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 2)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    async def test_skipped_waiter_keeps_its_place(self):
        """A waiter passed over for its per-user cap is not sent to the back."""
        started = []
        events = {}

        async def turn(user, label):
            ev = asyncio.Event()
            events[label] = ev
            async with turn_gate.slot(user):
                started.append(label)
                await ev.wait()

        a1 = asyncio.create_task(turn("A", "a1"))
        await _until(lambda: started == ["a1"])
        a2 = asyncio.create_task(turn("A", "a2"))  # queued, skipped every pump
        await _until(lambda: turn_gate.stats()["waiting"] == 1)
        c1 = asyncio.create_task(turn("C", "c1"))  # arrives after a2
        await _until(lambda: "c1" in started)

        # Freeing A makes a2 eligible; it must beat a later arrival.
        events["a1"].set()
        await _until(lambda: "a2" in started)
        self.assertEqual(started, ["a1", "c1", "a2"])

        events["a2"].set()
        events["c1"].set()
        await asyncio.gather(a1, a2, c1)

    # -- cancellation -----------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_cancel_while_queued_leaves_no_residue(self):
        release = asyncio.Event()
        entered = []

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def waiter():
            async with turn_gate.slot("u2"):
                entered.append(True)

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        w.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await w

        # Nothing was ever counted for the cancelled waiter.
        self.assertEqual(turn_gate.stats()["active"], 1)
        self.assertEqual(turn_gate.stats()["waiting"], 0)
        self.assertEqual(entered, [])

        release.set()
        await h
        self.assertEqual(turn_gate.stats()["active"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_cancel_after_admission_releases_the_slot(self):
        """The classic asyncio.Semaphore leak, which this gate must not have.

        A waiter is handed ownership by the pump and then cancelled before it
        can resume. Ownership lives in the ticket state, not in the return value
        of the await, so the slot is still given back.
        """
        entered = []

        async def waiter():
            async with turn_gate.slot("u2"):
                entered.append(True)
                await asyncio.sleep(3600)

        gate = turn_gate._gate()
        # Hold the only slot with a bare ticket, so releasing it is synchronous
        # and the waiter cannot slip in between statements.
        holder = gate._enqueue("u1")
        self.assertEqual(holder.state, turn_gate._HELD)

        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        gate._finish(holder)  # synchronous: pumps the waiter into ownership
        self.assertEqual(turn_gate.stats()["active"], 1)
        self.assertEqual(entered, [])  # ...but it has not resumed yet

        w.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await w

        self.assertEqual(turn_gate.stats()["active"], 0)
        self.assertEqual(turn_gate.stats()["waiting"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_exception_in_body_releases_the_slot(self):
        class Boom(Exception):
            pass

        with self.assertRaises(Boom):
            async with turn_gate.slot("u1"):
                raise Boom()

        self.assertEqual(turn_gate.stats()["active"], 0)

    # -- timeout ----------------------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_timeout_raises_and_frees_the_line(self):
        release = asyncio.Event()
        result = {}

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def waiter():
            try:
                async with turn_gate.slot("u2", timeout=0.05):
                    result["admitted"] = True
            except turn_gate.TurnQueueTimeout:
                result["timed_out"] = True

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        await waiter()

        self.assertEqual(result, {"timed_out": True})
        self.assertEqual(turn_gate.stats()["waiting"], 0)
        self.assertEqual(turn_gate.stats()["active"], 1)

        release.set()
        await h

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_timeout_never_fires_on_an_admitted_turn(self):
        """A deadline landing after admission must be a no-op, never a false busy."""
        entered = []

        async def waiter():
            async with turn_gate.slot("u2", timeout=60):
                entered.append(True)

        gate = turn_gate._gate()
        holder = gate._enqueue("u1")
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)
        ticket = next(iter(gate._waiters.values()))

        gate._finish(holder)  # admits the waiter; ticket is now _HELD
        self.assertEqual(ticket.state, turn_gate._HELD)

        # The deadline fires late, as it would if it raced the pump.
        gate._expire(ticket)

        await w  # must NOT raise TurnQueueTimeout
        self.assertEqual(entered, [True])
        self.assertEqual(turn_gate.stats()["active"], 0)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_zero_timeout_waits_indefinitely(self):
        release = asyncio.Event()
        entered = []

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def waiter():
            async with turn_gate.slot("u2", timeout=0):
                entered.append(True)

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        await asyncio.sleep(0.1)  # well past any short deadline
        self.assertEqual(turn_gate.stats()["waiting"], 1)
        self.assertEqual(entered, [])

        release.set()
        await asyncio.gather(h, w)
        self.assertEqual(entered, [True])

    # -- disabling and configuration --------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 0)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_zero_disables_both_caps(self):
        release = asyncio.Event()
        started = []

        async def turn(label):
            async with turn_gate.slot("same-user"):
                started.append(label)
                await release.wait()

        tasks = [asyncio.create_task(turn(i)) for i in range(5)]
        await _until(lambda: len(started) == 5)
        self.assertEqual(turn_gate.stats()["waiting"], 0)
        release.set()
        await asyncio.gather(*tasks)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 0)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1)
    async def test_zero_system_cap_still_honours_the_per_user_cap(self):
        release = asyncio.Event()
        started = []

        async def turn(user, label):
            async with turn_gate.slot(user):
                started.append(label)
                await release.wait()

        a = asyncio.create_task(turn("A", "a"))
        await _until(lambda: started == ["a"])
        b = asyncio.create_task(turn("A", "a2"))
        await _until(lambda: turn_gate.stats()["waiting"] == 1)
        c = asyncio.create_task(turn("B", "b"))
        await _until(lambda: "b" in started)

        release.set()
        await asyncio.gather(a, b, c)

    async def test_caps_are_read_at_admission_time(self):
        """Constants are module globals, so patching them takes effect at once."""
        release = asyncio.Event()
        started = []

        async def turn(label):
            async with turn_gate.slot("u1"):
                started.append(label)
                await release.wait()

        with patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 4):
            first = asyncio.create_task(turn("first"))
            await _until(lambda: started == ["first"])

        with patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 1):
            second = asyncio.create_task(turn("second"))
            await _until(lambda: turn_gate.stats()["waiting"] == 1)
            self.assertEqual(started, ["first"])

        release.set()
        await asyncio.gather(first, second)

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_pump_applies_a_cap_raised_while_a_turn_waits(self):
        release = asyncio.Event()
        started = []

        async def turn(label):
            async with turn_gate.slot(label):
                started.append(label)
                await release.wait()

        first = asyncio.create_task(turn("one"))
        await _until(lambda: started == ["one"])
        second = asyncio.create_task(turn("two"))
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        with patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 4):
            turn_gate.pump()
            await _until(lambda: "two" in started)

        release.set()
        await asyncio.gather(first, second)

    # -- callback, stats, reset -------------------------------------------

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_on_queued_failure_does_not_break_admission(self):
        release = asyncio.Event()
        entered = []

        async def holder():
            async with turn_gate.slot("u1"):
                await release.wait()

        async def on_queued(reason, position):
            raise RuntimeError("socket went away")

        async def waiter():
            async with turn_gate.slot("u2", on_queued=on_queued):
                entered.append(True)

        h = asyncio.create_task(holder())
        await _until(lambda: turn_gate.stats()["active"] == 1)
        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        release.set()
        await asyncio.gather(h, w)
        self.assertEqual(entered, [True])

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 4)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_stats_reports_active_waiting_and_per_user(self):
        release = asyncio.Event()

        async def turn(user):
            async with turn_gate.slot(user):
                await release.wait()

        tasks = [
            asyncio.create_task(turn("A")),
            asyncio.create_task(turn("A")),
            asyncio.create_task(turn("B")),
        ]
        await _until(lambda: turn_gate.stats()["active"] == 3)

        snapshot = turn_gate.stats()
        self.assertEqual(snapshot["waiting"], 0)
        self.assertEqual(snapshot["per_user"], {"A": 2, "B": 1})

        release.set()
        await asyncio.gather(*tasks)
        # The per-user key is dropped at zero so the map cannot grow unbounded.
        self.assertEqual(turn_gate.stats()["per_user"], {})

    @patch("chat.turn_gate.CHAT_TURN_MAX_CONCURRENT", 1)
    @patch("chat.turn_gate.CHAT_TURN_MAX_PER_USER", 0)
    async def test_reset_clears_state_and_unblocks_waiters(self):
        outcome = {}

        async def waiter():
            try:
                async with turn_gate.slot("u2", timeout=0):
                    outcome["admitted"] = True
            except turn_gate.TurnQueueTimeout:
                outcome["released"] = True

        gate = turn_gate._gate()
        # Occupy the slot with a bare ticket that is never finished, so nothing
        # decrements the tally after reset() has zeroed it. A holder *task* here
        # would trip the negative-tally guard on its way out, which is a genuine
        # invariant alarm and should stay reserved for real bugs.
        gate._enqueue("u1")

        w = asyncio.create_task(waiter())
        await _until(lambda: turn_gate.stats()["waiting"] == 1)

        turn_gate.reset()
        await w
        self.assertEqual(outcome, {"released": True})
        self.assertEqual(turn_gate.stats(), {"active": 0, "waiting": 0, "per_user": {}})


class EnvIntTests(SimpleTestCase):
    """The env reader must survive Heroku writing an empty config var."""

    def test_missing_returns_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(turn_gate._env_int("NOPE", 7), 7)

    def test_empty_string_returns_default(self):
        with patch.dict("os.environ", {"CAP": "   "}):
            self.assertEqual(turn_gate._env_int("CAP", 7), 7)

    def test_garbage_returns_default(self):
        with patch.dict("os.environ", {"CAP": "eight"}):
            self.assertEqual(turn_gate._env_int("CAP", 7), 7)

    def test_valid_value_is_parsed(self):
        with patch.dict("os.environ", {"CAP": " 3 "}):
            self.assertEqual(turn_gate._env_int("CAP", 7), 3)
