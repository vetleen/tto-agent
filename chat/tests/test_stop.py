"""Tests for the stop / cancellation feature."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatThread, SubAgentRun
from chat.subagent_tool import CreateSubagentTool
from core.preferences import ResolvedPreferences
from llm.types.context import RunContext

User = get_user_model()


def _prefs(**overrides):
    """Create a ResolvedPreferences with sensible defaults."""
    defaults = dict(
        top_model="openai/gpt-5",
        mid_model="openai/gpt-5-mini",
        cheap_model="openai/gpt-5-nano",
        allowed_models=["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano"],
        allowed_tools=[
            "search_documents", "read_document", "web_fetch", "brave_search",
            "write_canvas", "edit_canvas", "create_subagent",
        ],
        allowed_skills=[],
        theme="light",
        parallel_subagents=True,
    )
    defaults.update(overrides)
    return ResolvedPreferences(**defaults)


def _ctx(user_id, thread_id, data_room_ids=None):
    return RunContext.create(
        user_id=user_id,
        conversation_id=str(thread_id),
        data_room_ids=data_room_ids or [],
    )


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


# ---------------------------------------------------------------------------
# _cancel_active_subagents tests
# ---------------------------------------------------------------------------

class CancelActiveSubagentsPendingTests(TestCase):
    """_cancel_active_subagents marks PENDING runs as FAILED."""

    def setUp(self):
        self.user = User.objects.create_user(email="cancel_pending@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("celery.result.AsyncResult")
    def test_pending_runs_marked_failed(self, mock_async_result):
        run1 = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="pending task 1", status=SubAgentRun.Status.PENDING,
            celery_task_id="celery-1",
        )
        run2 = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="pending task 2", status=SubAgentRun.Status.PENDING,
        )

        # Call the sync inner function directly
        from chat.consumers import ChatConsumer
        consumer = ChatConsumer()
        consumer.user = self.user
        # Access the wrapped sync function via __wrapped__
        cancel_sync = ChatConsumer._cancel_active_subagents.__wrapped__
        cancel_sync(consumer, self.thread.id)

        run1.refresh_from_db()
        run2.refresh_from_db()

        self.assertEqual(run1.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run1.error, "Cancelled by user.")
        self.assertIsNotNone(run1.completed_at)

        self.assertEqual(run2.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run2.error, "Cancelled by user.")
        self.assertIsNotNone(run2.completed_at)

        # Celery task revoked for run1 (which had a celery_task_id)
        mock_async_result.assert_called_once_with("celery-1")
        mock_async_result.return_value.revoke.assert_called_once_with(terminate=True)


class CancelActiveSubagentsRunningTests(TestCase):
    """_cancel_active_subagents marks RUNNING runs as FAILED."""

    def setUp(self):
        self.user = User.objects.create_user(email="cancel_running@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("celery.result.AsyncResult")
    def test_running_runs_marked_failed(self, mock_async_result):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="running task", status=SubAgentRun.Status.RUNNING,
            celery_task_id="celery-running-1",
        )

        from chat.consumers import ChatConsumer
        cancel_sync = ChatConsumer._cancel_active_subagents.__wrapped__
        consumer = ChatConsumer()
        consumer.user = self.user
        cancel_sync(consumer, self.thread.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run.error, "Cancelled by user.")
        self.assertIsNotNone(run.completed_at)

        mock_async_result.assert_called_once_with("celery-running-1")
        mock_async_result.return_value.revoke.assert_called_once_with(terminate=True)


class CancelActiveSubagentsIgnoresTerminalTests(TestCase):
    """_cancel_active_subagents does not affect COMPLETED or FAILED runs."""

    def setUp(self):
        self.user = User.objects.create_user(email="cancel_terminal@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_completed_and_failed_runs_untouched(self):
        completed_run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="done task", status=SubAgentRun.Status.COMPLETED,
            result="All good.", completed_at=timezone.now(),
        )
        failed_run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="bad task", status=SubAgentRun.Status.FAILED,
            error="LLM error", completed_at=timezone.now(),
        )

        from chat.consumers import ChatConsumer
        cancel_sync = ChatConsumer._cancel_active_subagents.__wrapped__
        consumer = ChatConsumer()
        consumer.user = self.user
        cancel_sync(consumer, self.thread.id)

        completed_run.refresh_from_db()
        failed_run.refresh_from_db()

        self.assertEqual(completed_run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(completed_run.result, "All good.")

        self.assertEqual(failed_run.status, SubAgentRun.Status.FAILED)
        self.assertEqual(failed_run.error, "LLM error")


class CancelActiveSubagentsOtherThreadTests(TestCase):
    """_cancel_active_subagents does not affect runs on other threads."""

    def setUp(self):
        self.user = User.objects.create_user(email="cancel_other@test.com", password="pass")
        self.thread_a = ChatThread.objects.create(created_by=self.user)
        self.thread_b = ChatThread.objects.create(created_by=self.user)

    def test_other_thread_runs_untouched(self):
        # Active run on thread B — should NOT be cancelled
        run_b = SubAgentRun.objects.create(
            thread=self.thread_b, user=self.user,
            prompt="other thread task", status=SubAgentRun.Status.RUNNING,
        )
        # Active run on thread A — should be cancelled
        run_a = SubAgentRun.objects.create(
            thread=self.thread_a, user=self.user,
            prompt="this thread task", status=SubAgentRun.Status.PENDING,
        )

        from chat.consumers import ChatConsumer
        cancel_sync = ChatConsumer._cancel_active_subagents.__wrapped__
        consumer = ChatConsumer()
        consumer.user = self.user
        cancel_sync(consumer, self.thread_a.id)

        run_a.refresh_from_db()
        run_b.refresh_from_db()

        self.assertEqual(run_a.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run_a.error, "Cancelled by user.")

        # Thread B run is untouched
        self.assertEqual(run_b.status, SubAgentRun.Status.RUNNING)
        self.assertEqual(run_b.error, "")


class CancelActiveSubagentsCeleryFailureTests(TestCase):
    """Runs are still marked FAILED even when Celery revocation raises."""

    def setUp(self):
        self.user = User.objects.create_user(email="cancel_celery_fail@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("celery.result.AsyncResult")
    def test_runs_marked_failed_despite_revoke_error(self, mock_async_result):
        mock_async_result.return_value.revoke.side_effect = ConnectionError("broker down")

        run1 = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task 1", status=SubAgentRun.Status.RUNNING,
            celery_task_id="celery-fail-1",
        )
        run2 = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task 2", status=SubAgentRun.Status.PENDING,
            celery_task_id="celery-fail-2",
        )

        from chat.consumers import ChatConsumer

        cancel_sync = ChatConsumer._cancel_active_subagents.__wrapped__
        consumer = ChatConsumer()
        consumer.user = self.user
        cancel_sync(consumer, self.thread.id)

        run1.refresh_from_db()
        run2.refresh_from_db()

        # Both runs should be FAILED even though revoke raised
        self.assertEqual(run1.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run1.error, "Cancelled by user.")
        self.assertIsNotNone(run1.completed_at)

        self.assertEqual(run2.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run2.error, "Cancelled by user.")
        self.assertIsNotNone(run2.completed_at)


# ---------------------------------------------------------------------------
# Sequential enforcement tests
# ---------------------------------------------------------------------------

class SequentialSubagentEnforcementTests(TestCase):
    """When parallel_subagents is False, the tool blocks a second subagent."""

    def setUp(self):
        self.user = User.objects.create_user(email="sequential@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("core.preferences.get_preferences")
    def test_blocks_second_subagent_when_parallel_disabled(self, mock_get_prefs):
        mock_get_prefs.return_value = _prefs(parallel_subagents=False)

        # An existing active run on this thread
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="first task", status=SubAgentRun.Status.PENDING,
        )

        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {"prompt": "second task", "timeout": 0}, ctx)

        self.assertEqual(result["status"], "error")
        self.assertIn("one at a time", result["message"])

    @patch("chat.tasks.run_subagent_task")
    @patch("core.preferences.get_preferences")
    def test_allows_when_parallel_enabled(self, mock_get_prefs, mock_task):
        mock_get_prefs.return_value = _prefs(parallel_subagents=True)
        mock_task.delay.return_value = MagicMock(id="celery-parallel")

        # An existing active run on this thread
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="first task", status=SubAgentRun.Status.PENDING,
        )

        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {"prompt": "second task", "timeout": 0}, ctx)

        self.assertEqual(result["status"], "started")
