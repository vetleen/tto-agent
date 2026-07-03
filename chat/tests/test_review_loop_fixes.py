"""Tests for the Group B loop / sub-agent fixes from the chat-app review.

Covers:
- B1: an invalid timezone is rejected at loop-create time (no poison row).
- B2: a headless loop turn skipped for a suspended / over-budget owner advances
      the schedule but doesn't count as a run.
- B4: a Celery retry after COMPLETED does not resurrect and re-run the job.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from chat.loop_service import HeadlessTurnRunner, create_loop, execute_loop_run
from chat.models import ChatThread, Loop, SubAgentRun
from chat.subagent_service import run_subagent

User = get_user_model()


def _loop_body(**overrides):
    body = {
        "prompt": "Summarize new docs.",
        "history_mode": "fresh",
        "cadence_kind": "interval",
        "interval_value": 6, "interval_unit": "hours",
        "first_run_mode": "now",
    }
    body.update(overrides)
    return body


class InvalidTimezoneTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="tz@x.io", password="p")

    def test_create_loop_rejects_invalid_tz(self):
        loop, errors = create_loop(
            user=self.user, body=_loop_body(), now=timezone.now(), tz_name="Oslo",
        )
        self.assertIsNone(loop)
        self.assertTrue(any("time zone" in e for e in errors))
        self.assertFalse(Loop.objects.exists())

    def test_create_loop_accepts_valid_tz(self):
        loop, errors = create_loop(
            user=self.user, body=_loop_body(), now=timezone.now(),
            tz_name="Europe/Oslo",
        )
        self.assertEqual(errors, [])
        self.assertIsNotNone(loop)


class LoopSuspensionSkipTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="skip@x.io", password="p")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.loop = Loop.objects.create(
            thread=self.thread, created_by=self.user, prompt="p",
            history_mode="fresh", cadence_kind="interval", interval_seconds=3600,
            next_run=timezone.now() - timedelta(minutes=1), running=True,
            status=Loop.Status.ACTIVE,
        )

    def test_skipped_run_advances_schedule_without_counting(self):
        with patch.object(HeadlessTurnRunner, "run_loop_turn", new=AsyncMock(return_value=False)):
            execute_loop_run(self.loop.id)
        self.loop.refresh_from_db()
        self.assertEqual(self.loop.runs_completed, 0)          # not counted
        self.assertGreater(self.loop.next_run, timezone.now())  # rescheduled
        self.assertEqual(self.loop.status, Loop.Status.ACTIVE)  # still active
        self.assertFalse(self.loop.running)                     # lock released

    def test_ran_turn_counts(self):
        with patch.object(HeadlessTurnRunner, "run_loop_turn", new=AsyncMock(return_value=True)):
            execute_loop_run(self.loop.id)
        self.loop.refresh_from_db()
        self.assertEqual(self.loop.runs_completed, 1)


class SubagentRetryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="retry@x.io", password="p")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_completed_run_is_not_resurrected(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="done",
            status=SubAgentRun.Status.COMPLETED, result="already delivered",
        )
        # A retry re-entering run_subagent on a COMPLETED row must no-op, not flip
        # it back to RUNNING and re-run the LLM job. If the guard failed, get_llm
        # would be reached; the early return means it never is.
        run_subagent(run.id, deadline_seconds=1)
        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(run.result, "already delivered")
