"""Tests for the UpdateTasksTool."""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase

from chat.models import ChatThread, ThreadTask
from chat.task_tools import UpdateTasksTool
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(args, ctx):
    tool = UpdateTasksTool()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


class UpdateTasksToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="tasks@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)

    def test_create_tasks_from_empty(self):
        result = _invoke({"tasks": [
            {"title": "Research patents", "status": "pending"},
            {"title": "Draft report", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["tasks"]), 2)
        self.assertEqual(result["tasks"][0]["title"], "Research patents")
        self.assertEqual(result["tasks"][1]["title"], "Draft report")
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 2)

    def test_update_existing_tasks(self):
        # Create initial tasks
        result = _invoke({"tasks": [
            {"title": "Step 1", "status": "pending"},
            {"title": "Step 2", "status": "pending"},
        ]}, self.ctx)
        task_id = result["tasks"][0]["id"]

        # Update first task to in_progress, change title of second
        result2 = _invoke({"tasks": [
            {"id": task_id, "title": "Step 1", "status": "in_progress"},
            {"id": result["tasks"][1]["id"], "title": "Step 2 updated", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(result2["status"], "ok")
        self.assertEqual(result2["tasks"][0]["status"], "in_progress")
        self.assertEqual(result2["tasks"][1]["title"], "Step 2 updated")

    def test_delete_omitted_tasks(self):
        # Create 3 tasks
        result = _invoke({"tasks": [
            {"title": "A", "status": "pending"},
            {"title": "B", "status": "pending"},
            {"title": "C", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 3)

        # Send back only 2 — the omitted one should be deleted
        result2 = _invoke({"tasks": [
            {"id": result["tasks"][0]["id"], "title": "A", "status": "completed"},
            {"id": result["tasks"][2]["id"], "title": "C", "status": "in_progress"},
        ]}, self.ctx)
        self.assertEqual(len(result2["tasks"]), 2)
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 2)

    def test_reorder_tasks(self):
        result = _invoke({"tasks": [
            {"title": "First", "status": "pending"},
            {"title": "Second", "status": "pending"},
        ]}, self.ctx)

        # Swap order
        result2 = _invoke({"tasks": [
            {"id": result["tasks"][1]["id"], "title": "Second", "status": "pending"},
            {"id": result["tasks"][0]["id"], "title": "First", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(result2["tasks"][0]["title"], "Second")
        self.assertEqual(result2["tasks"][0]["order"], 0)
        self.assertEqual(result2["tasks"][1]["title"], "First")
        self.assertEqual(result2["tasks"][1]["order"], 1)

    def test_no_context_returns_error(self):
        tool = UpdateTasksTool()
        result = json.loads(tool.invoke({"tasks": [{"title": "X", "status": "pending"}]}))
        self.assertEqual(result["status"], "error")

    def test_idempotent_full_replace(self):
        """Sending the same state twice produces no duplicates."""
        result = _invoke({"tasks": [
            {"title": "Task A", "status": "in_progress"},
            {"title": "Task B", "status": "pending"},
        ]}, self.ctx)
        # Send exact same state again
        result2 = _invoke({"tasks": [
            {"id": result["tasks"][0]["id"], "title": "Task A", "status": "in_progress"},
            {"id": result["tasks"][1]["id"], "title": "Task B", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(len(result2["tasks"]), 2)
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 2)
        # IDs should be the same
        self.assertEqual(result["tasks"][0]["id"], result2["tasks"][0]["id"])
        self.assertEqual(result["tasks"][1]["id"], result2["tasks"][1]["id"])

    def test_invalid_status_defaults_to_pending(self):
        result = _invoke({"tasks": [
            {"title": "Bad status", "status": "nonexistent"},
        ]}, self.ctx)
        self.assertEqual(result["tasks"][0]["status"], "pending")

    def test_summary_format(self):
        result = _invoke({"tasks": [
            {"title": "A", "status": "completed"},
            {"title": "B", "status": "in_progress"},
            {"title": "C", "status": "pending"},
        ]}, self.ctx)
        self.assertIn("3 tasks", result["summary"])
        self.assertIn("1 done", result["summary"])
        self.assertIn("1 active", result["summary"])
        self.assertIn("1 pending", result["summary"])

    def test_clear_all_tasks(self):
        """Sending empty list deletes all tasks."""
        _invoke({"tasks": [
            {"title": "X", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 1)

        result = _invoke({"tasks": []}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["tasks"]), 0)
        self.assertEqual(ThreadTask.objects.filter(thread=self.thread).count(), 0)

    def test_bulk_create_does_not_scale_queries(self):
        """Creating N tasks must be a fixed number of queries, not one INSERT each
        (WILFRED-7V — the per-task ThreadTask.objects.create() N+1)."""
        from django.db import connection
        from django.test.utils import CaptureQueriesContext

        def query_count_for(n):
            thread = ChatThread.objects.create(created_by=self.user)
            ctx = _ctx(self.user.pk, thread.id)
            tasks = [{"title": f"T{i}", "status": "pending"} for i in range(n)]
            with CaptureQueriesContext(connection) as cap:
                _invoke({"tasks": tasks}, ctx)
            return len(cap)

        self.assertEqual(query_count_for(2), query_count_for(12))

    def test_update_persists_to_db(self):
        """bulk_update path must actually write (auto_now updated_at stamped too)."""
        result = _invoke({"tasks": [
            {"title": "Original", "status": "pending"},
        ]}, self.ctx)
        task_id = result["tasks"][0]["id"]
        _invoke({"tasks": [
            {"id": task_id, "title": "Renamed", "status": "completed"},
        ]}, self.ctx)
        row = ThreadTask.objects.get(id=task_id)
        self.assertEqual(row.title, "Renamed")
        self.assertEqual(row.status, "completed")

    def test_mix_new_and_existing(self):
        """Can update existing tasks and add new ones in same call."""
        result = _invoke({"tasks": [
            {"title": "Existing", "status": "pending"},
        ]}, self.ctx)
        existing_id = result["tasks"][0]["id"]

        result2 = _invoke({"tasks": [
            {"id": existing_id, "title": "Existing", "status": "completed"},
            {"title": "Brand new", "status": "pending"},
        ]}, self.ctx)
        self.assertEqual(len(result2["tasks"]), 2)
        self.assertEqual(result2["tasks"][0]["id"], existing_id)
        self.assertEqual(result2["tasks"][0]["status"], "completed")
        self.assertNotEqual(result2["tasks"][1]["id"], existing_id)
        self.assertEqual(result2["tasks"][1]["title"], "Brand new")


class SubagentTaskUpdateToolTests(TestCase):
    """A sub-agent's plan lives on its run, never on the thread (that plan is
    the orchestrator's, and the main tool replaces it wholesale)."""

    def setUp(self):
        from chat.models import SubAgentRun

        self.user = User.objects.create_user(email="subtasks@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        ThreadTask.objects.create(thread=self.thread, title="Orchestrator step", status="pending")
        self.run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="t")
        self.ctx = _ctx(self.user.pk, self.thread.id)
        self.ctx.run_id = str(self.run.id)
        self.ctx.agent_kind = "subagent"

    def _invoke(self, args, ctx=None):
        from chat.task_tools import SubagentTaskUpdateTool

        tool = SubagentTaskUpdateTool()
        tool.set_context(ctx or self.ctx)
        return json.loads(tool.invoke(args))

    def test_writes_run_plan_and_leaves_thread_plan_alone(self):
        result = self._invoke({"tasks": [
            {"title": "Search", "status": "in_progress"},
            {"title": "Write canvas"},
        ]})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"], "2 tasks (1 active, 1 pending)")
        self.run.refresh_from_db()
        self.assertEqual([t["title"] for t in self.run.task_plan], ["Search", "Write canvas"])
        self.assertEqual(
            list(ThreadTask.objects.filter(thread=self.thread).values_list("title", flat=True)),
            ["Orchestrator step"],
        )

    def test_keeps_known_ids_drops_omitted_and_normalizes_status(self):
        first = self._invoke({"tasks": [{"title": "A"}, {"title": "B"}]})
        a_id = first["tasks"][0]["id"]
        foreign = "00000000-0000-0000-0000-000000000001"
        second = self._invoke({"tasks": [
            {"id": a_id, "title": "A", "status": "completed"},
            {"id": foreign, "title": "C", "status": "bogus"},
        ]})
        self.assertEqual(second["tasks"][0]["id"], a_id)
        self.assertNotEqual(second["tasks"][1]["id"], foreign)
        self.assertEqual(second["tasks"][1]["status"], "pending")
        self.run.refresh_from_db()
        self.assertEqual([t["title"] for t in self.run.task_plan], ["A", "C"])

    def test_errors_outside_a_subagent_run(self):
        result = self._invoke({"tasks": [{"title": "A"}]}, ctx=_ctx(self.user.pk, self.thread.id))
        self.assertEqual(result["status"], "error")

    def test_audiences(self):
        from chat.task_tools import SubagentTaskUpdateTool

        self.assertEqual(UpdateTasksTool().audience, "main")
        self.assertEqual(SubagentTaskUpdateTool().audience, "subagent")
