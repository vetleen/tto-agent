"""Tests for shared tool throttling helpers (llm/tools/_throttle.py)."""

from unittest import TestCase

from llm.tools._throttle import deadline_capped_wait
from llm.types.context import RunContext


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
