"""Tests for the loop-management agent tools (chat_loop_*).

These exercise the tool handlers in isolation (no consumer / websocket / e2e
machinery), mirroring chat/tests/test_task_tools.py.
"""

from __future__ import annotations

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatThread, Loop
from chat.tool_loops import LoopCreateTool, LoopEditTool, LoopListTool, LoopStopTool
from documents.models import DataRoom
from llm.types.context import RunContext

User = get_user_model()


def _ctx(user_id, thread_id=None):
    return RunContext.create(
        user_id=user_id,
        conversation_id=str(thread_id) if thread_id else None,
    )


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


def _make_loop(user, *, status=Loop.Status.ACTIVE, interval_seconds=6 * 3600,
               runs_completed=0, max_runs=10, prompt="Original."):
    thread = ChatThread.objects.create(created_by=user)
    return Loop.objects.create(
        thread=thread, created_by=user, prompt=prompt,
        cadence_kind="interval", interval_seconds=interval_seconds,
        next_run=timezone.now(), status=status,
        runs_completed=runs_completed, max_runs=max_runs,
    )


class CreateLoopToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="create@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)

    def test_create_interval_happy_path(self):
        result = _invoke(LoopCreateTool, {
            "prompt": "Summarize new docs.",
            "cadence_kind": "interval", "interval_value": 6, "interval_unit": "hours",
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["schedule_label"], "Every 6 hours")
        self.assertEqual(result["loop_status"], "active")
        # A loop lives in its OWN new thread, never the current conversation.
        self.assertNotEqual(result["thread_id"], str(self.thread.id))
        self.assertEqual(Loop.objects.count(), 1)
        loop = Loop.objects.get(id=result["loop_id"])
        self.assertEqual(loop.created_by, self.user)
        self.assertEqual(loop.interval_seconds, 6 * 3600)

    def test_create_clock_daily(self):
        result = _invoke(LoopCreateTool, {
            "prompt": "Morning digest.",
            "cadence_kind": "clock", "clock_frequency": "daily", "clock_time": "09:00",
            "first_run_mode": "scheduled",
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["schedule_label"], "Daily at 09:00")
        loop = Loop.objects.get(id=result["loop_id"])
        self.assertEqual(loop.cadence_kind, "clock")
        self.assertEqual(loop.clock_time.strftime("%H:%M"), "09:00")

    def test_create_attaches_explicit_data_rooms(self):
        dr = DataRoom.objects.create(created_by=self.user, name="Room A")
        result = _invoke(LoopCreateTool, {
            "prompt": "Watch the room.", "data_room_ids": [dr.pk],
        }, self.ctx)
        loop = Loop.objects.get(id=result["loop_id"])
        self.assertEqual(
            list(loop.thread.data_rooms.values_list("pk", flat=True)), [dr.pk],
        )

    def test_create_without_resources_runs_bare(self):
        result = _invoke(LoopCreateTool, {"prompt": "Nothing attached."}, self.ctx)
        loop = Loop.objects.get(id=result["loop_id"])
        self.assertEqual(loop.thread.data_rooms.count(), 0)
        self.assertEqual(loop.thread.skills.count(), 0)

    def test_create_attaches_explicit_skills(self):
        from agent_skills.models import AgentSkill
        from chat.models import ChatThreadSkill

        s1 = AgentSkill.objects.create(
            slug="loop-s1", name="S1", instructions="x", level="user", created_by=self.user,
        )
        s2 = AgentSkill.objects.create(
            slug="loop-s2", name="S2", instructions="x", level="user", created_by=self.user,
        )
        result = _invoke(LoopCreateTool, {
            "prompt": "Use skills.", "skill_ids": [str(s1.id), str(s2.id)],
        }, self.ctx)
        loop = Loop.objects.get(id=result["loop_id"])
        attached = [
            str(sid) for sid in ChatThreadSkill.objects.filter(
                thread=loop.thread
            ).values_list("skill_id", flat=True)
        ]
        self.assertEqual(attached, [str(s1.id), str(s2.id)])

    def test_create_requires_prompt(self):
        result = _invoke(LoopCreateTool, {"prompt": "   "}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertEqual(Loop.objects.count(), 0)

    def test_create_defaults_to_unlimited(self):
        # No run cap unless the agent asks for one.
        result = _invoke(LoopCreateTool, {"prompt": "Run forever."}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["max_runs"])
        self.assertIsNone(Loop.objects.get(id=result["loop_id"]).max_runs)

    def test_create_with_max_runs(self):
        result = _invoke(LoopCreateTool, {
            "prompt": "Run a few times.", "max_runs": 7,
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["max_runs"], 7)
        self.assertEqual(Loop.objects.get(id=result["loop_id"]).max_runs, 7)

    def test_create_without_user_context(self):
        result = _invoke(LoopCreateTool, {"prompt": "No user."}, _ctx(None))
        self.assertEqual(result["status"], "error")
        self.assertEqual(Loop.objects.count(), 0)


class ListLoopsToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="list@test.com", password="pass")
        self.ctx = _ctx(self.user.pk)

    def test_list_only_callers_loops(self):
        _make_loop(self.user, prompt="Mine.")
        other = User.objects.create_user(email="other@test.com", password="pass")
        _make_loop(other, prompt="Theirs.")

        result = _invoke(LoopListTool, {}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["loops"][0]["prompt"], "Mine.")

    def test_list_item_shape(self):
        loop = _make_loop(self.user)
        result = _invoke(LoopListTool, {}, self.ctx)
        item = result["loops"][0]
        for key in (
            "loop_id", "prompt", "schedule_label", "loop_status", "history_mode",
            "next_run", "runs_completed", "max_runs", "thread_id", "is_unread",
        ):
            self.assertIn(key, item)
        self.assertEqual(item["loop_id"], str(loop.id))
        self.assertEqual(item["thread_id"], str(loop.thread_id))

    def test_list_empty(self):
        result = _invoke(LoopListTool, {}, self.ctx)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["loops"], [])

    def test_list_includes_paused(self):
        _make_loop(self.user, status=Loop.Status.PAUSED)
        result = _invoke(LoopListTool, {}, self.ctx)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["loops"][0]["loop_status"], "paused")


class StopLoopToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="stop@test.com", password="pass")
        self.ctx = _ctx(self.user.pk)

    def test_stop_pauses_loop(self):
        loop = _make_loop(self.user)
        result = _invoke(LoopStopTool, {"loop_id": str(loop.id)}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["loop_status"], "paused")
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.PAUSED)

    def test_stop_other_users_loop_denied(self):
        other = User.objects.create_user(email="o2@test.com", password="pass")
        loop = _make_loop(other)
        result = _invoke(LoopStopTool, {"loop_id": str(loop.id)}, self.ctx)
        self.assertEqual(result["status"], "error")
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.ACTIVE)  # untouched

    def test_stop_bad_uuid(self):
        result = _invoke(LoopStopTool, {"loop_id": "not-a-uuid"}, self.ctx)
        self.assertEqual(result["status"], "error")

    def test_stop_missing_loop(self):
        result = _invoke(LoopStopTool, {
            "loop_id": "00000000-0000-0000-0000-000000000000",
        }, self.ctx)
        self.assertEqual(result["status"], "error")


class EditLoopToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="edit@test.com", password="pass")
        self.ctx = _ctx(self.user.pk)

    def test_partial_edit_preserves_cadence(self):
        loop = _make_loop(self.user, interval_seconds=6 * 3600, prompt="Old.")
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id), "prompt": "New prompt only.",
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertEqual(loop.prompt, "New prompt only.")
        self.assertEqual(loop.interval_seconds, 6 * 3600)  # untouched

    def test_partial_edit_preserves_max_runs(self):
        # Omitting max_runs must not wipe an existing cap.
        loop = _make_loop(self.user, max_runs=10)
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id), "prompt": "New prompt.",
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertEqual(loop.max_runs, 10)

    def test_edit_sets_and_clears_max_runs(self):
        loop = _make_loop(self.user, max_runs=10)
        # Set a new cap.
        _invoke(LoopEditTool, {"loop_id": str(loop.id), "max_runs": 3}, self.ctx)
        loop.refresh_from_db()
        self.assertEqual(loop.max_runs, 3)
        # max_runs=0 clears it back to unlimited.
        result = _invoke(LoopEditTool, {"loop_id": str(loop.id), "max_runs": 0}, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertIsNone(loop.max_runs)

    def test_edit_interval_to_clock(self):
        loop = _make_loop(self.user)
        original_next = loop.next_run
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id),
            "cadence_kind": "clock", "clock_frequency": "daily", "clock_time": "07:30",
            "first_run_mode": "scheduled",
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertEqual(loop.cadence_kind, "clock")
        self.assertEqual(loop.clock_time.strftime("%H:%M"), "07:30")
        self.assertNotEqual(loop.next_run, original_next)  # recomputed

    def test_edit_restart_resets_run_count(self):
        loop = _make_loop(
            self.user, status=Loop.Status.PAUSED, runs_completed=5, max_runs=5,
        )
        before = timezone.now()
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id), "restart": True,
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.ACTIVE)
        self.assertEqual(loop.runs_completed, 0)
        self.assertGreaterEqual(loop.next_run, before)
        self.assertFalse(loop.running)

    def test_edit_resume_keeps_run_count(self):
        loop = _make_loop(
            self.user, status=Loop.Status.PAUSED, runs_completed=3, max_runs=10,
        )
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id), "resume": True,
        }, self.ctx)
        self.assertEqual(result["status"], "ok")
        loop.refresh_from_db()
        self.assertEqual(loop.status, Loop.Status.ACTIVE)
        self.assertEqual(loop.runs_completed, 3)  # resumed where it left off

    def test_edit_other_users_loop_denied(self):
        other = User.objects.create_user(email="o3@test.com", password="pass")
        loop = _make_loop(other, prompt="Theirs.")
        result = _invoke(LoopEditTool, {
            "loop_id": str(loop.id), "prompt": "hijack",
        }, self.ctx)
        self.assertEqual(result["status"], "error")
        loop.refresh_from_db()
        self.assertEqual(loop.prompt, "Theirs.")

    def test_edit_bad_uuid(self):
        result = _invoke(LoopEditTool, {"loop_id": "nope", "prompt": "x"}, self.ctx)
        self.assertEqual(result["status"], "error")


class LoopToolExposureTests(TestCase):
    """Loop tools moved behind the assistant_loop_tools skill (section='skills'),
    so they are no longer in the default always-on tool set — they surface only
    when that skill is active."""

    def test_loop_tools_not_in_default_allowed_tools(self):
        from core.preferences import get_preferences

        user = User.objects.create_user(email="exposure@test.com", password="pass")
        allowed = get_preferences(user).allowed_tools
        for name in (
            "chat_loop_create", "chat_loop_list", "chat_loop_stop", "chat_loop_edit",
        ):
            self.assertNotIn(name, allowed)

    def test_loop_tools_are_valid_skill_tools(self):
        from agent_skills.services import filter_to_skill_tools

        # All four survive the skills-section + main-audience allow-list, so the
        # assistant_loop_tools skill can carry them.
        names = ["chat_loop_create", "chat_loop_list", "chat_loop_stop", "chat_loop_edit"]
        self.assertEqual(filter_to_skill_tools(names, skill_audience="main"), names)
