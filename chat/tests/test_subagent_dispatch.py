"""Tests for the sub-agent execution queue: slot-gated, arrival-order dispatch."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.db.utils import OperationalError
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatMessage, ChatThread, SubAgentRun
from chat.subagent_limits import (
    STALE_WAITING_MINUTES,
    _expire_stale_runs,
    dispatch_pending_subagents,
    get_queue_depth,
)
from chat.subagent_tool import CreateSubagentTool
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id):
    return RunContext.create(
        user_id=user_id, conversation_id=str(thread_id), data_room_ids=[],
    )


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


def _backdate(run, **minutes_ago):
    """Rewrite timestamp columns by minutes (auto_now_add can't be set on create)."""
    SubAgentRun.objects.filter(pk=run.pk).update(**{
        field: timezone.now() - timedelta(minutes=minutes)
        for field, minutes in minutes_ago.items()
    })
    run.refresh_from_db()
    return run


class _QueueTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="queue@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _run(
        self, status=SubAgentRun.Status.PENDING, *,
        minutes_ago=0, dispatched_minutes_ago=None, **kw,
    ):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", status=status, **kw,
        )
        deltas = {}
        if minutes_ago:
            deltas["created_at"] = minutes_ago
        if dispatched_minutes_ago is not None:
            deltas["dispatched_at"] = dispatched_minutes_ago
        return _backdate(run, **deltas) if deltas else run


# ---------------------------------------------------------------------------
# dispatch_pending_subagents
# ---------------------------------------------------------------------------

class DispatchPendingSubagentsTests(_QueueTestCase):
    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 2)
    @patch("chat.tasks.run_subagent_task")
    def test_dispatches_oldest_waiting_first_up_to_free_slots(self, mock_task):
        mock_task.delay.side_effect = lambda run_id: MagicMock(id=f"celery-{run_id[:8]}")
        oldest = self._run(minutes_ago=3)
        middle = self._run(minutes_ago=2)
        newest = self._run(minutes_ago=1)

        dispatched = dispatch_pending_subagents()

        self.assertEqual(dispatched, [oldest.id, middle.id])
        for run in (oldest, middle):
            run.refresh_from_db()
            self.assertEqual(run.status, SubAgentRun.Status.PENDING)
            self.assertIsNotNone(run.dispatched_at)
            self.assertEqual(run.celery_task_id, f"celery-{str(run.id)[:8]}")
        newest.refresh_from_db()
        self.assertIsNone(newest.dispatched_at)
        self.assertEqual(newest.celery_task_id, "")
        self.assertEqual(mock_task.delay.call_count, 2)
        mock_task.delay.assert_any_call(str(oldest.id))

    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 2)
    @patch("chat.tasks.run_subagent_task")
    def test_running_and_dispatched_pending_hold_slots(self, mock_task):
        self._run(SubAgentRun.Status.RUNNING)
        self._run(dispatched_minutes_ago=0, celery_task_id="celery-x")
        waiting = self._run(minutes_ago=1)

        self.assertEqual(dispatch_pending_subagents(), [])

        mock_task.delay.assert_not_called()
        waiting.refresh_from_db()
        self.assertIsNone(waiting.dispatched_at)

    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 1)
    @patch("chat.tasks.run_subagent_task")
    def test_terminal_rows_do_not_hold_slots(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-1")
        self._run(SubAgentRun.Status.COMPLETED, dispatched_minutes_ago=5)
        self._run(SubAgentRun.Status.FAILED, dispatched_minutes_ago=5)
        waiting = self._run(minutes_ago=1)

        self.assertEqual(dispatch_pending_subagents(), [waiting.id])

    @patch("chat.tasks.run_subagent_task")
    def test_publish_failure_reverts_claim(self, mock_task):
        mock_task.delay.side_effect = ConnectionError("broker down")
        waiting = self._run(minutes_ago=1)

        self.assertEqual(dispatch_pending_subagents(), [])

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, SubAgentRun.Status.PENDING)
        self.assertIsNone(waiting.dispatched_at)
        self.assertEqual(waiting.celery_task_id, "")

    @patch("chat.tasks.run_subagent_task")
    def test_publish_failure_does_not_resurrect_cancelled_row(self, mock_task):
        waiting = self._run(minutes_ago=1)

        def cancel_then_fail(run_id):
            SubAgentRun.objects.filter(pk=run_id).update(
                status=SubAgentRun.Status.FAILED, error="Cancelled by user.",
            )
            raise ConnectionError("broker down")

        mock_task.delay.side_effect = cancel_then_fail

        self.assertEqual(dispatch_pending_subagents(), [])

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, SubAgentRun.Status.FAILED)
        self.assertEqual(waiting.error, "Cancelled by user.")

    @patch("chat.tasks.run_subagent_task")
    def test_noop_when_nothing_waiting(self, mock_task):
        self._run(SubAgentRun.Status.RUNNING)

        self.assertEqual(dispatch_pending_subagents(), [])

        mock_task.delay.assert_not_called()

    def test_get_queue_depth_counts_slots_and_waiting(self):
        self._run(SubAgentRun.Status.RUNNING)
        self._run(dispatched_minutes_ago=0)
        self._run(minutes_ago=1)
        self._run(SubAgentRun.Status.COMPLETED)

        depth = get_queue_depth()

        self.assertEqual(depth["running"], 2)
        self.assertEqual(depth["waiting"], 1)


# ---------------------------------------------------------------------------
# Task hooks: after_return frees a slot; the sweeper is the backstop
# ---------------------------------------------------------------------------

class SubagentTaskAfterReturnTests(_QueueTestCase):
    @patch("chat.subagent_limits.dispatch_pending_subagents")
    def test_after_return_calls_dispatcher(self, mock_dispatch):
        from chat.tasks import run_subagent_task

        run_subagent_task.after_return(
            "SUCCESS", None, "tid", [str(self._run().id)], {}, None,
        )

        mock_dispatch.assert_called_once_with()

    @patch(
        "chat.subagent_limits.dispatch_pending_subagents",
        side_effect=RuntimeError("boom"),
    )
    def test_after_return_swallows_dispatcher_error(self, mock_dispatch):
        from chat.tasks import run_subagent_task

        run_subagent_task.after_return(
            "SUCCESS", None, "tid", [str(self._run().id)], {}, None,
        )

        mock_dispatch.assert_called_once()

    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 1)
    @patch("chat.tasks.run_subagent_task.delay")
    def test_failure_then_after_return_frees_slot(self, mock_delay):
        from chat.tasks import run_subagent_task

        mock_delay.return_value = MagicMock(id="celery-next")
        failing = self._run(SubAgentRun.Status.RUNNING)
        waiting = self._run(minutes_ago=1)

        # The RUNNING row holds the only slot.
        self.assertEqual(dispatch_pending_subagents(), [])

        run_subagent_task.on_failure(
            RuntimeError("boom"), "tid", [str(failing.id)], {}, None,
        )
        run_subagent_task.after_return(
            "FAILURE", None, "tid", [str(failing.id)], {}, None,
        )

        waiting.refresh_from_db()
        self.assertIsNotNone(waiting.dispatched_at)
        self.assertEqual(waiting.celery_task_id, "celery-next")
        mock_delay.assert_called_once_with(str(waiting.id))


class SweeperDispatchTests(_QueueTestCase):
    @patch("chat.subagent_limits.dispatch_pending_subagents")
    def test_expire_task_calls_dispatcher(self, mock_dispatch):
        from chat.tasks import expire_stale_subagent_runs

        self.assertEqual(expire_stale_subagent_runs(), 0)

        mock_dispatch.assert_called_once_with()

    @patch(
        "chat.subagent_limits.dispatch_pending_subagents",
        side_effect=OperationalError("db gone"),
    )
    def test_expire_task_swallows_dispatcher_operational_error(self, mock_dispatch):
        from chat.tasks import expire_stale_subagent_runs

        self.assertEqual(expire_stale_subagent_runs(), 0)

    def test_waiting_run_not_expired_at_pending_threshold(self):
        """Time spent waiting in line is legitimate — the 7-minute pending rule
        applies only to runs already handed to Celery."""
        waiting = self._run(minutes_ago=11)

        self.assertEqual(_expire_stale_runs(), 0)

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, SubAgentRun.Status.PENDING)

    def test_waiting_run_expired_after_max_wait(self):
        waiting = self._run(minutes_ago=STALE_WAITING_MINUTES + 1)

        self.assertEqual(_expire_stale_runs(), 1)

        waiting.refresh_from_db()
        self.assertEqual(waiting.status, SubAgentRun.Status.FAILED)
        self.assertTrue(waiting.error.startswith("Expired:"))
        self.assertIn("queue", waiting.error)
        self.assertTrue(
            ChatMessage.objects.filter(
                thread=self.thread, is_hidden_from_user=True,
                content__contains="Expired:",
            ).exists()
        )

    def test_dispatched_pending_clocked_from_dispatched_at(self):
        run = self._run(minutes_ago=20, dispatched_minutes_ago=2)

        self.assertEqual(_expire_stale_runs(), 0)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)

    def test_dispatched_pending_expired_after_pending_threshold(self):
        run = self._run(minutes_ago=20, dispatched_minutes_ago=8)

        self.assertEqual(_expire_stale_runs(), 1)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("pending", run.error)


# ---------------------------------------------------------------------------
# The create tool reports the real queue state
# ---------------------------------------------------------------------------

class CreateSubagentToolQueueStatusTests(_QueueTestCase):
    def setUp(self):
        super().setUp()
        self.other_user = User.objects.create_user(email="other@test.com", password="pass")
        self.other_thread = ChatThread.objects.create(created_by=self.other_user)

    def _other_run(self, status, **kw):
        return SubAgentRun.objects.create(
            thread=self.other_thread, user=self.other_user, prompt="theirs",
            status=status, **kw,
        )

    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 1)
    @patch("chat.tasks.run_subagent_task")
    def test_reports_queued_with_position_when_slots_full(self, mock_task):
        self._other_run(SubAgentRun.Status.RUNNING)
        _backdate(self._other_run(SubAgentRun.Status.PENDING), created_at=1)

        result = _invoke(
            CreateSubagentTool, {"prompt": "new task"}, _ctx(self.user.pk, self.thread.id),
        )

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["queue_position"], 2)
        self.assertEqual(result["ahead_in_queue"], 1)
        self.assertEqual(result["currently_running"], 1)
        self.assertEqual(result["worker_slots"], 1)
        self.assertIn("starts automatically", result["message"])
        mock_task.delay.assert_not_called()
        run = SubAgentRun.objects.get(pk=result["run_id"])
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)
        self.assertIsNone(run.dispatched_at)
        self.assertEqual(run.celery_task_id, "")

    @patch("chat.tasks.run_subagent_task")
    def test_reports_started_when_dispatched(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-1")

        result = _invoke(
            CreateSubagentTool, {"prompt": "task"}, _ctx(self.user.pk, self.thread.id),
        )

        self.assertEqual(result["status"], "started")
        run = SubAgentRun.objects.get(pk=result["run_id"])
        self.assertIsNotNone(run.dispatched_at)
        self.assertEqual(run.celery_task_id, "celery-1")

    @patch(
        "chat.subagent_limits.dispatch_pending_subagents",
        side_effect=RuntimeError("boom"),
    )
    def test_reports_queued_when_dispatcher_raises(self, mock_dispatch):
        result = _invoke(
            CreateSubagentTool, {"prompt": "task"}, _ctx(self.user.pk, self.thread.id),
        )

        self.assertEqual(result["status"], "queued")
        self.assertTrue(
            SubAgentRun.objects.filter(
                pk=result["run_id"], status=SubAgentRun.Status.PENDING,
            ).exists()
        )

    @patch("chat.subagent_limits.dispatch_pending_subagents")
    def test_reports_started_when_a_sibling_call_dispatched_it(self, mock_dispatch):
        """Parallel chat_subagent_create calls in one turn: a sibling thread's
        dispatcher can hand out this run first, so our own call returns nothing."""
        def sibling_dispatched_everything():
            SubAgentRun.objects.filter(
                status=SubAgentRun.Status.PENDING, dispatched_at__isnull=True,
            ).update(dispatched_at=timezone.now(), celery_task_id="celery-sibling")
            return []

        mock_dispatch.side_effect = sibling_dispatched_everything

        result = _invoke(
            CreateSubagentTool, {"prompt": "task"}, _ctx(self.user.pk, self.thread.id),
        )

        self.assertEqual(result["status"], "started")
        run = SubAgentRun.objects.get(pk=result["run_id"])
        self.assertEqual(run.celery_task_id, "celery-sibling")

    @patch("chat.subagent_limits.SUBAGENT_WORKER_SLOTS", 1)
    @patch("chat.tasks.run_subagent_task")
    def test_deadline_branch_reports_queued_when_still_waiting(self, mock_task):
        self._other_run(SubAgentRun.Status.RUNNING)

        with patch("chat.subagent_tool.time") as mock_time:
            mock_time.monotonic.side_effect = [0, 2, 32]
            mock_time.sleep.return_value = None
            result = _invoke(
                CreateSubagentTool, {"prompt": "task", "timeout": 30},
                _ctx(self.user.pk, self.thread.id),
            )

        self.assertEqual(result["status"], "queued")
        self.assertIn("waiting for a free execution slot", result["message"])
        mock_task.delay.assert_not_called()

    @patch("chat.subagent_limits.SUBAGENT_MAX_PER_USER", 1)
    def test_per_user_cap_counts_waiting_runs(self):
        self._run()

        result = _invoke(
            CreateSubagentTool, {"prompt": "task"}, _ctx(self.user.pk, self.thread.id),
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("too many", result["message"])


class CancelDispatchTests(_QueueTestCase):
    @patch("celery.result.AsyncResult")
    @patch("chat.subagent_limits.dispatch_pending_subagents")
    def test_cancel_triggers_dispatcher(self, mock_dispatch, mock_async_result):
        self._run(dispatched_minutes_ago=0, celery_task_id="celery-1")
        from chat.consumers import ChatConsumer

        consumer = ChatConsumer()
        consumer.user = self.user
        ChatConsumer._cancel_active_subagents.__wrapped__(consumer, self.thread.id)

        mock_dispatch.assert_called_once_with()


# ---------------------------------------------------------------------------
# Orchestrator prompt wording
# ---------------------------------------------------------------------------

class SubagentStatusPromptWordingTests(TestCase):
    def _prompt_for(self, **run):
        from chat.prompts import build_system_prompt

        base = {
            "id": uuid.uuid4(), "status": "pending", "prompt": "Research",
            "model_tier": "mid", "result": "", "error": "",
        }
        base.update(run)
        return build_system_prompt(has_subagent_tool=True, subagent_runs=[base])

    def test_waiting_run_says_queued(self):
        prompt = self._prompt_for(dispatched_at=None)
        self.assertIn("Still in progress", prompt)
        self.assertIn("waiting for a free execution slot", prompt)

    def test_dispatched_run_says_starting(self):
        prompt = self._prompt_for(dispatched_at=timezone.now())
        self.assertIn("(starting)", prompt)

    def test_limit_line_counts_waiting_runs(self):
        from chat.prompts import build_system_prompt

        prompt = build_system_prompt(has_subagent_tool=True)
        self.assertIn("up to 4 sub-agents concurrently (waiting or running)", prompt)
