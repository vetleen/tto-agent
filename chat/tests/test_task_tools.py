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
