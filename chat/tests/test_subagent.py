"""Tests for the sub-agent system."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from chat.models import ChatThread, SubAgentRun
from chat.subagent_limits import (
    _expire_stale_runs,
    check_subagent_limits,
    create_subagent_run_if_allowed,
)
from chat.subagent_prompts import build_subagent_system_prompt
from chat.subagent_service import resolve_subagent_model, resolve_subagent_tools
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
# SubAgentRun model tests
# ---------------------------------------------------------------------------

class SubAgentRunModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sub@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_create_run(self):
        run = SubAgentRun.objects.create(
            thread=self.thread,
            user=self.user,
            prompt="Analyze patent claims",
            model_tier="mid",
        )
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)
        self.assertEqual(run.prompt, "Analyze patent claims")
        self.assertEqual(run.timeout, 0)
        self.assertEqual(run.data_room_ids, [])
        self.assertEqual(run.tool_names, [])

    def test_status_transitions(self):
        run = SubAgentRun.objects.create(
            thread=self.thread,
            user=self.user,
            prompt="task",
        )
        self.assertEqual(run.status, "pending")

        run.status = SubAgentRun.Status.RUNNING
        run.save(update_fields=["status"])
        run.refresh_from_db()
        self.assertEqual(run.status, "running")

        run.status = SubAgentRun.Status.COMPLETED
        run.result = "Done"
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "result", "completed_at"])
        run.refresh_from_db()
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.result, "Done")
        self.assertIsNotNone(run.completed_at)

    def test_json_fields_serialize(self):
        run = SubAgentRun.objects.create(
            thread=self.thread,
            user=self.user,
            prompt="task",
            data_room_ids=[1, 2, 3],
            tool_names=["search_documents", "web_fetch"],
        )
        run.refresh_from_db()
        self.assertEqual(run.data_room_ids, [1, 2, 3])
        self.assertEqual(run.tool_names, ["search_documents", "web_fetch"])

    def test_str_representation(self):
        run = SubAgentRun.objects.create(
            thread=self.thread,
            user=self.user,
            prompt="task",
        )
        self.assertIn("pending", str(run))


# ---------------------------------------------------------------------------
# resolve_subagent_model tests
# ---------------------------------------------------------------------------

class ResolveSubagentModelTests(TestCase):
    def test_fast_maps_to_cheap(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("fast", prefs), "openai/gpt-5-nano")

    def test_mid_maps_to_mid(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("mid", prefs), "openai/gpt-5-mini")

    def test_top_maps_to_top(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("top", prefs), "openai/gpt-5")

    def test_invalid_tier_defaults_to_mid(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("invalid", prefs), "openai/gpt-5-mini")


# ---------------------------------------------------------------------------
# resolve_subagent_tools tests
# ---------------------------------------------------------------------------

class ResolveSubagentToolsTests(TestCase):
    def test_removes_canvas_and_subagent_tools(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[1])
        self.assertNotIn("active_canvas", tools)
        self.assertNotIn("write_canvas", tools)
        self.assertNotIn("edit_canvas", tools)
        self.assertNotIn("create_subagent", tools)

    def test_keeps_doc_tools_with_data_rooms(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[1])
        self.assertIn("search_documents", tools)
        self.assertIn("read_document", tools)

    def test_removes_doc_tools_without_data_rooms(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertNotIn("search_documents", tools)
        self.assertNotIn("read_document", tools)

    def test_keeps_web_tools(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertIn("web_fetch", tools)
        self.assertIn("brave_search", tools)

    def test_skill_tools_not_added(self):
        """Sub-agents never get skill-specific tools."""
        prefs = _prefs(allowed_skills=[{
            "id": "1", "slug": "test-skill", "name": "Test",
            "description": "", "tool_names": ["custom_tool"],
        }])
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertNotIn("custom_tool", tools)


# ---------------------------------------------------------------------------
# build_subagent_system_prompt tests
# ---------------------------------------------------------------------------

class BuildSubagentSystemPromptTests(TestCase):
    def test_contains_identity_and_task(self):
        prompt = build_subagent_system_prompt("Analyze the patent")
        self.assertIn(f"sub-agent of {settings.ASSISTANT_NAME}", prompt)
        self.assertIn("Analyze the patent", prompt)

    def test_includes_org_name(self):
        prompt = build_subagent_system_prompt(
            "task", organization_name="MIT TTO",
        )
        self.assertIn("MIT TTO", prompt)

    def test_no_skill_injection(self):
        """Sub-agent prompts never include skill instructions."""
        prompt = build_subagent_system_prompt("task")
        self.assertNotIn("Specific instructions", prompt)

    def test_includes_return_findings_instruction(self):
        prompt = build_subagent_system_prompt("task")
        self.assertIn("Return your findings as text", prompt)

    def test_includes_data_rooms(self):
        rooms = [{"name": "Room A", "description": "Patent docs"}]
        prompt = build_subagent_system_prompt("task", data_rooms=rooms)
        self.assertIn("Room A", prompt)
        self.assertIn("Patent docs", prompt)

    def test_minimal_prompt_without_extras(self):
        prompt = build_subagent_system_prompt("Simple task")
        self.assertIn("# Identity", prompt)
        self.assertIn("# Task", prompt)
        self.assertIn("# General instructions", prompt)
        self.assertNotIn("# Skill", prompt)
        self.assertNotIn("# Attached Data Rooms", prompt)

    def test_contains_web_content_safety_warning(self):
        """Sub-agent prompt must include web content safety instructions."""
        prompt = build_subagent_system_prompt("Research a topic")
        self.assertIn("Web Content Safety", prompt)
        self.assertIn("untrusted content", prompt)
        self.assertIn("never follow instructions", prompt)


# ---------------------------------------------------------------------------
# check_subagent_limits tests
# ---------------------------------------------------------------------------

class CheckSubagentLimitsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="limit@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_allows_when_under_limit(self):
        allowed, msg = check_subagent_limits(self.user)
        self.assertTrue(allowed)
        self.assertEqual(msg, "")

    @patch("chat.subagent_limits.SUBAGENT_MAX_PER_USER", 2)
    def test_denies_at_per_user_limit(self):
        for _ in range(2):
            SubAgentRun.objects.create(
                thread=self.thread, user=self.user,
                prompt="task", status=SubAgentRun.Status.RUNNING,
            )
        allowed, msg = check_subagent_limits(self.user)
        self.assertFalse(allowed)
        self.assertIn("too many", msg)

    @patch("chat.subagent_limits.SUBAGENT_MAX_SYSTEM", 2)
    def test_denies_at_system_limit(self):
        other_user = User.objects.create_user(email="other@test.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user)
        for _ in range(2):
            SubAgentRun.objects.create(
                thread=other_thread, user=other_user,
                prompt="task", status=SubAgentRun.Status.PENDING,
            )
        allowed, msg = check_subagent_limits(self.user)
        self.assertFalse(allowed)
        self.assertIn("busy", msg)

    def test_completed_runs_dont_count(self):
        for _ in range(10):
            SubAgentRun.objects.create(
                thread=self.thread, user=self.user,
                prompt="task", status=SubAgentRun.Status.COMPLETED,
            )
        allowed, msg = check_subagent_limits(self.user)
        self.assertTrue(allowed)


# ---------------------------------------------------------------------------
# create_subagent_run_if_allowed (atomic) tests
# ---------------------------------------------------------------------------

class CreateSubagentRunIfAllowedTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="atomic@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_creates_run_when_under_limit(self):
        run, err = create_subagent_run_if_allowed(
            self.user,
            thread_id=self.thread.id,
            prompt="research task",
            model_tier="mid",
        )
        self.assertIsNotNone(run)
        self.assertEqual(err, "")
        self.assertEqual(run.prompt, "research task")
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)

    @patch("chat.subagent_limits.SUBAGENT_MAX_PER_USER", 1)
    def test_denies_at_per_user_limit(self):
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="existing", status=SubAgentRun.Status.RUNNING,
        )
        run, err = create_subagent_run_if_allowed(
            self.user,
            thread_id=self.thread.id,
            prompt="new task",
        )
        self.assertIsNone(run)
        self.assertIn("too many", err)

    @patch("chat.subagent_limits.SUBAGENT_MAX_SYSTEM", 1)
    def test_denies_at_system_limit(self):
        other_user = User.objects.create_user(email="other2@test.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user)
        SubAgentRun.objects.create(
            thread=other_thread, user=other_user,
            prompt="running", status=SubAgentRun.Status.RUNNING,
        )
        run, err = create_subagent_run_if_allowed(
            self.user,
            thread_id=self.thread.id,
            prompt="new task",
        )
        self.assertIsNone(run)
        self.assertIn("busy", err)


# ---------------------------------------------------------------------------
# Stale run expiration tests
# ---------------------------------------------------------------------------

class StaleRunExpirationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="stale@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_stale_pending_runs_expired(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="stuck", status=SubAgentRun.Status.PENDING,
        )
        # Backdate created_at to 11 minutes ago
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=11),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 1)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("pending", run.error)
        self.assertIsNotNone(run.completed_at)

    def test_stale_running_runs_expired(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="stuck", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=16),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 1)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("running", run.error)

    def test_recent_runs_not_expired(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="fresh", status=SubAgentRun.Status.PENDING,
        )
        # 5 minutes ago — within the threshold
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=5),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 0)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)

    def test_stale_runs_freed_on_limit_check(self):
        """Stale runs should be cleaned up during limit checks, freeing slots."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="stuck", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=16),
        )

        # The stale run is cleaned up, so this user should be allowed
        allowed, msg = check_subagent_limits(self.user)
        self.assertTrue(allowed)


# ---------------------------------------------------------------------------
# CreateSubagentTool tests — timeout polling
# ---------------------------------------------------------------------------

class CreateSubagentToolTimeoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="timeout_tool@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.tasks.run_subagent_task")
    @patch("chat.subagent_tool.time")
    def test_completed_during_poll_returns_result(self, mock_time, mock_task):
        """Run completes after 2 poll cycles — tool returns the result inline."""
        mock_task.delay.return_value = MagicMock(id="celery-task-123")
        # monotonic: start=0, then 2, 4 (two polls), then 6 (exit check)
        mock_time.monotonic.side_effect = [0, 2, 4, 6]
        mock_time.sleep = MagicMock()

        poll_count = 0

        def fake_refresh(run_obj):
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                SubAgentRun.objects.filter(pk=run_obj.pk).update(
                    status=SubAgentRun.Status.COMPLETED,
                    result="Analysis complete: 3 claims found.",
                )
            run_obj.__dict__.update(
                SubAgentRun.objects.filter(pk=run_obj.pk).values()[0]
            )

        ctx = _ctx(self.user.pk, self.thread.id)
        tool = CreateSubagentTool()
        tool.set_context(ctx)

        with patch.object(SubAgentRun, "refresh_from_db", autospec=True, side_effect=fake_refresh):
            result = tool.invoke({"prompt": "Analyze patent claims", "timeout": 60})

        self.assertEqual(result, "Analysis complete: 3 claims found.")
        run = SubAgentRun.objects.first()
        self.assertEqual(run.timeout, 60)

    @patch("chat.tasks.run_subagent_task")
    @patch("chat.subagent_tool.time")
    def test_completed_with_empty_result_returns_warning(self, mock_time, mock_task):
        """Run completes with empty result — tool returns a structured warning."""
        mock_task.delay.return_value = MagicMock(id="celery-empty")
        mock_time.monotonic.side_effect = [0, 2, 4]
        mock_time.sleep = MagicMock()

        def fake_refresh(run_obj):
            SubAgentRun.objects.filter(pk=run_obj.pk).update(
                status=SubAgentRun.Status.COMPLETED,
                result="",
            )
            run_obj.__dict__.update(
                SubAgentRun.objects.filter(pk=run_obj.pk).values()[0]
            )

        ctx = _ctx(self.user.pk, self.thread.id)
        tool = CreateSubagentTool()
        tool.set_context(ctx)

        with patch.object(SubAgentRun, "refresh_from_db", autospec=True, side_effect=fake_refresh):
            raw = tool.invoke({"prompt": "Research task", "timeout": 60})

        result = json.loads(raw)
        self.assertEqual(result["status"], "completed")
        self.assertIn("no text content", result["message"])

    @patch("chat.tasks.run_subagent_task")
    @patch("chat.subagent_tool.time")
    def test_failed_during_poll_returns_error(self, mock_time, mock_task):
        """Run fails during poll — tool returns an error."""
        mock_task.delay.return_value = MagicMock(id="celery-task-456")
        mock_time.monotonic.side_effect = [0, 2, 4]
        mock_time.sleep = MagicMock()

        def fake_refresh(run_obj):
            SubAgentRun.objects.filter(pk=run_obj.pk).update(
                status=SubAgentRun.Status.FAILED,
                error="LLM API error",
            )
            run_obj.__dict__.update(
                SubAgentRun.objects.filter(pk=run_obj.pk).values()[0]
            )

        ctx = _ctx(self.user.pk, self.thread.id)

        with patch.object(SubAgentRun, "refresh_from_db", autospec=True, side_effect=fake_refresh):
            result = _invoke(CreateSubagentTool, {"prompt": "task", "timeout": 30}, ctx)

        self.assertEqual(result["status"], "error")
        self.assertIn("LLM API error", result["message"])

    @patch("chat.tasks.run_subagent_task")
    @patch("chat.subagent_tool.time")
    def test_timeout_exceeded_returns_started(self, mock_time, mock_task):
        """Run still RUNNING after timeout — returns run_id for later check."""
        mock_task.delay.return_value = MagicMock(id="celery-task-789")
        # monotonic: start=0, then 2 (first sleep done), then 32 (past deadline of 30)
        mock_time.monotonic.side_effect = [0, 2, 32]
        mock_time.sleep = MagicMock()

        def fake_refresh(run_obj):
            # Stay PENDING — never completes
            run_obj.__dict__.update(
                SubAgentRun.objects.filter(pk=run_obj.pk).values()[0]
            )

        ctx = _ctx(self.user.pk, self.thread.id)

        with patch.object(SubAgentRun, "refresh_from_db", autospec=True, side_effect=fake_refresh):
            result = _invoke(CreateSubagentTool, {"prompt": "slow task", "timeout": 30}, ctx)

        self.assertEqual(result["status"], "started")
        self.assertIn("run_id", result)
        self.assertIn("still running", result["message"])

    @patch("chat.tasks.run_subagent_task")
    def test_timeout_clamped_to_max(self, mock_task):
        """timeout=999 is clamped to 540 and stored on the run."""
        mock_task.delay.return_value = MagicMock(id="celery-clamp")

        ctx = _ctx(self.user.pk, self.thread.id)
        # timeout=0 after clamping would skip polling, but 999 clamps to 540.
        # We use timeout=0 path indirectly — just verify the stored value.
        # Actually, 999 clamps to 540 which is >0, so it would poll.
        # Easier: just invoke with timeout=0 style to skip polling, and check the DB.
        # Let's mock time to immediately exceed deadline.
        with patch("chat.subagent_tool.time") as mock_time:
            mock_time.monotonic.side_effect = [0, 541]
            mock_time.sleep = MagicMock()

            def fake_refresh(run_obj):
                run_obj.__dict__.update(
                    SubAgentRun.objects.filter(pk=run_obj.pk).values()[0]
                )

            with patch.object(SubAgentRun, "refresh_from_db", autospec=True, side_effect=fake_refresh):
                _invoke(CreateSubagentTool, {"prompt": "task", "timeout": 999}, ctx)

        run = SubAgentRun.objects.first()
        self.assertEqual(run.timeout, 540)

    @patch("chat.tasks.run_subagent_task")
    def test_timeout_negative_clamped_to_zero(self, mock_task):
        """timeout=-10 is clamped to 0 — behaves as fire-and-forget."""
        mock_task.delay.return_value = MagicMock(id="celery-neg")

        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {"prompt": "task", "timeout": -10}, ctx)

        self.assertEqual(result["status"], "started")
        run = SubAgentRun.objects.first()
        self.assertEqual(run.timeout, 0)


class CreateSubagentToolBackgroundTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="nonblock@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.tasks.run_subagent_task")
    def test_timeout_zero_returns_started(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-task-123")

        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {
            "prompt": "Background research",
        }, ctx)

        self.assertEqual(result["status"], "started")
        self.assertIn("run_id", result)
        mock_task.delay.assert_called_once()

        run = SubAgentRun.objects.first()
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)
        self.assertEqual(run.timeout, 0)
        self.assertEqual(run.celery_task_id, "celery-task-123")
        self.assertEqual(run.model_used, "")
        self.assertEqual(run.tool_names, [])

    @patch("chat.tasks.run_subagent_task")
    def test_invalid_tier_defaults_to_mid(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t1")

        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(CreateSubagentTool, {
            "prompt": "task", "model_tier": "invalid",
        }, ctx)

        run = SubAgentRun.objects.first()
        self.assertEqual(run.model_tier, "mid")


class CreateSubagentToolLimitTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="limits@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.subagent_limits.SUBAGENT_MAX_PER_USER", 1)
    def test_limit_hit_returns_error(self):
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="running", status=SubAgentRun.Status.RUNNING,
        )

        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {"prompt": "new task"}, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("too many", result["message"])


# ---------------------------------------------------------------------------
# BuildSystemPromptSubagentStatusTests
# ---------------------------------------------------------------------------

class BuildSystemPromptSubagentStatusTests(TestCase):
    """Tests for sub-agent status injection into the system prompt."""

    def test_no_runs_no_status_section(self):
        from chat.prompts import build_system_prompt
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=None)
        self.assertNotIn("# Sub-agent Status", prompt)

    def test_empty_runs_no_status_section(self):
        from chat.prompts import build_system_prompt
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=[])
        self.assertNotIn("# Sub-agent Status", prompt)

    def test_pending_run_shows_in_progress(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "pending",
            "prompt": "Research patent claims", "model_tier": "mid",
            "result": "", "error": "", "result_delivered": False,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("# Sub-agent Status", prompt)
        self.assertIn("PENDING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_running_run_shows_in_progress(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "running",
            "prompt": "Analyze documents", "model_tier": "fast",
            "result": "", "error": "", "result_delivered": False,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("RUNNING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_completed_undelivered_includes_result(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Summarize findings", "model_tier": "top",
            "result": "Found 3 key patents.", "error": "", "result_delivered": False,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("COMPLETED", prompt)
        self.assertIn("Found 3 key patents.", prompt)
        self.assertNotIn("already delivered", prompt.lower())

    def test_completed_delivered_omits_result(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Summarize findings", "model_tier": "top",
            "result": "Found 3 key patents.", "error": "", "result_delivered": True,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("COMPLETED", prompt)
        self.assertIn("already delivered", prompt.lower())
        self.assertNotIn("Found 3 key patents.", prompt)

    def test_failed_shows_error(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "failed",
            "prompt": "Bad task", "model_tier": "mid",
            "result": "", "error": "LLM provider timeout", "result_delivered": False,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("FAILED", prompt)
        self.assertIn("LLM provider timeout", prompt)

    def test_no_check_subagent_status_in_prompt(self):
        from chat.prompts import build_system_prompt
        prompt = build_system_prompt(has_subagent_tool=True)
        self.assertNotIn("check_subagent_status", prompt)

    def test_completed_result_truncated_at_8000_chars(self):
        from chat.prompts import build_system_prompt
        long_result = "x" * 10000
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Big task", "model_tier": "mid",
            "result": long_result, "error": "", "result_delivered": False,
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("(truncated)", prompt)
        # The full 10000-char result should not appear
        self.assertNotIn(long_result, prompt)


# ---------------------------------------------------------------------------
# Celery task tests
# ---------------------------------------------------------------------------

class RunSubagentTaskTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="celery@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_task_runs_subagent(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Task completed successfully."
        mock_response.usage.total_tokens = 500
        mock_response.usage.cost_usd = 0.01
        mock_svc.return_value.run.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="Do research",
        )

        # Call the task function directly (not through broker)
        from chat.tasks import run_subagent_task
        run_subagent_task(str(run.id))

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(run.result, "Task completed successfully.")
        self.assertEqual(run.tokens_used, 500)
        self.assertIsNotNone(run.completed_at)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_task_handles_failure_sets_pending_for_retry(self, mock_prefs, mock_svc):
        """On failure, run_subagent sets PENDING (not FAILED) so Celery can retry."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run.side_effect = RuntimeError("Provider down")

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="Do research",
        )

        from chat.tasks import run_subagent_task
        with self.assertRaises(RuntimeError):
            run_subagent_task(str(run.id))

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)
        self.assertIn("Provider down", run.error)


# ---------------------------------------------------------------------------
# run_subagent service tests
# ---------------------------------------------------------------------------

class RunSubagentServiceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="svc@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_sets_model_used(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.001
        mock_svc.return_value.run.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", model_tier="fast",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.model_used, "openai/gpt-5-nano")
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_passes_data_room_ids(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", data_room_ids=[1, 2],
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        # Verify the request included data room IDs in context
        call_args = mock_svc.return_value.run.call_args
        request = call_args[0][1]
        self.assertEqual(request.context.data_room_ids, [1, 2])

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_warns_on_unresolved_tool_calls_with_empty_content(self, mock_prefs, mock_svc):
        """If response has tool_calls but no content, a warning is logged."""
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = [{"id": "c1", "name": "brave_search"}]
        mock_response.usage.total_tokens = 200
        mock_response.usage.cost_usd = 0.002
        mock_svc.return_value.run.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="research task",
        )

        from chat.subagent_service import run_subagent
        with self.assertLogs("chat.subagent_service", level="WARNING") as cm:
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(run.result, "")
        self.assertTrue(any("unresolved tool calls" in msg for msg in cm.output))

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_failure_sets_pending_for_retry(self, mock_prefs, mock_svc):
        """run_subagent should set PENDING on failure for Celery retry."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run.side_effect = RuntimeError("Transient error")

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task",
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(RuntimeError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.PENDING)
        self.assertIn("Transient error", run.error)


# ---------------------------------------------------------------------------
# Celery on_failure handler tests
# ---------------------------------------------------------------------------

class CeleryOnFailureTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="onfail@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def test_on_failure_sets_permanent_failed_status(self):
        """on_failure handler should mark run as FAILED after retries exhausted."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="doomed task", status=SubAgentRun.Status.PENDING,
        )

        from chat.tasks import run_subagent_task
        # Call the on_failure method on the task instance
        run_subagent_task.on_failure(
            RuntimeError("Final failure"),
            "fake-task-id",
            [str(run.id)],
            {},
            None,
        )

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("Final failure", run.error)
        self.assertIsNotNone(run.completed_at)
