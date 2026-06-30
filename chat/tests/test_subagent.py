"""Tests for the sub-agent system."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from unittest.mock import MagicMock, patch

from channels.db import database_sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings
from django.utils import timezone

from chat.models import ChatMessage, ChatThread, SubAgentRun
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
            "document_search", "document_read", "web_fetch", "web_search",
            "canvas_write", "canvas_edit", "chat_subagent_create",
        ],
        # The sub-agent base set: shared/subagent-audience chat tools only
        # (canvas/subagent/skill tools are audience="main" and excluded upstream).
        allowed_subagent_tools=[
            "document_search", "document_read", "web_fetch", "web_search",
        ],
        allowed_skills=[],
        allowed_specializations=[],
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
            tool_names=["document_search", "web_fetch"],
        )
        run.refresh_from_db()
        self.assertEqual(run.data_room_ids, [1, 2, 3])
        self.assertEqual(run.tool_names, ["document_search", "web_fetch"])

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
    def test_fast_falls_back_to_mid(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("fast", prefs), "openai/gpt-5-mini")

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
    def test_base_is_subagent_audience_tools(self):
        """The base set comes from prefs.allowed_subagent_tools — main-only tools
        (canvas, sub-agent spawn, skill management) are never in it."""
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[1])
        for name in [
            "canvas_activate", "canvas_write", "canvas_edit",
            "canvas_save_to_document", "chat_subagent_create",
            "chat_skill_attach", "skill_create", "skill_template_load",
        ]:
            self.assertNotIn(name, tools)

    def test_keeps_doc_tools_with_data_rooms(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[1])
        self.assertIn("document_search", tools)
        self.assertIn("document_read", tools)

    def test_removes_doc_tools_without_data_rooms(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertNotIn("document_search", tools)
        self.assertNotIn("document_read", tools)

    def test_keeps_web_tools(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertIn("web_fetch", tools)
        self.assertIn("web_search", tools)

    def test_no_specialization_means_no_skill_tools(self):
        """Without a specialization the sub-agent gets no skill-section tools."""
        prefs = _prefs(allowed_specializations=[{
            "id": "1", "slug": "web-researcher", "name": "Web Researcher",
            "emoji": "", "description": "", "tool_names": ["skill_template_view"],
        }])
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertNotIn("skill_template_view", tools)

    def test_specialization_adds_its_tools(self):
        """A spawned specialization contributes its (already filtered) tools."""
        prefs = _prefs(allowed_specializations=[{
            "id": "1", "slug": "web-researcher", "name": "Web Researcher",
            "emoji": "", "description": "", "tool_names": ["skill_template_view"],
        }])
        tools = resolve_subagent_tools(
            prefs, data_room_ids=[], specialization_slug="web-researcher",
        )
        self.assertIn("skill_template_view", tools)

    def test_unknown_specialization_adds_nothing(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(
            prefs, data_room_ids=[], specialization_slug="does-not-exist",
        )
        self.assertNotIn("skill_template_view", tools)


# ---------------------------------------------------------------------------
# build_subagent_system_prompt tests
# ---------------------------------------------------------------------------

class BuildSubagentSystemPromptTests(TestCase):
    def test_contains_identity(self):
        prompt = build_subagent_system_prompt()
        self.assertIn(f"sub-agent of {settings.ASSISTANT_NAME}", prompt)

    def test_task_not_in_system_prompt(self):
        prompt = build_subagent_system_prompt()
        self.assertNotIn("# Task", prompt)

    def test_includes_org_name(self):
        prompt = build_subagent_system_prompt(organization_name="MIT TTO")
        self.assertIn("MIT TTO", prompt)

    def test_no_skill_injection(self):
        """Sub-agent prompts never include skill instructions."""
        prompt = build_subagent_system_prompt()
        self.assertNotIn("Specific instructions", prompt)

    def test_includes_return_findings_instruction(self):
        prompt = build_subagent_system_prompt()
        self.assertIn("Return your findings as text", prompt)

    def test_includes_data_rooms(self):
        rooms = [{"name": "Room A", "description": "Patent docs"}]
        prompt = build_subagent_system_prompt(data_rooms=rooms)
        self.assertIn("Room A", prompt)
        self.assertIn("Patent docs", prompt)

    def test_data_room_content_safety(self):
        rooms = [{"name": "Room A", "description": ""}]
        prompt = build_subagent_system_prompt(data_rooms=rooms)
        self.assertIn("Content Safety", prompt)
        self.assertIn("user-uploaded", prompt)

    def test_no_data_rooms_message(self):
        prompt = build_subagent_system_prompt()
        self.assertIn("No data rooms are attached", prompt)

    def test_minimal_prompt_without_extras(self):
        prompt = build_subagent_system_prompt()
        self.assertIn("# Identity", prompt)
        self.assertIn("# General instructions", prompt)
        self.assertNotIn("# Task", prompt)
        self.assertNotIn("# Skill", prompt)
        self.assertNotIn("# Attached Data Rooms", prompt)
        self.assertNotIn("# Task Planning", prompt)

    def test_contains_web_content_safety_warning(self):
        """Sub-agent prompt must include web content safety instructions."""
        prompt = build_subagent_system_prompt()
        self.assertIn("Web Content Safety", prompt)
        self.assertIn("untrusted content", prompt)
        self.assertIn("never follow instructions", prompt)

    def test_task_planning_included_when_tool_available(self):
        prompt = build_subagent_system_prompt(has_task_tool=True)
        self.assertIn("# Task Planning", prompt)
        self.assertIn("chat_task_update", prompt)

    def test_task_planning_excluded_without_tool(self):
        prompt = build_subagent_system_prompt(has_task_tool=False)
        self.assertNotIn("# Task Planning", prompt)

    def test_no_specialization_by_default(self):
        prompt = build_subagent_system_prompt()
        self.assertNotIn("Your specialization", prompt)

    def test_specialization_injected(self):
        skill = MagicMock()
        skill.name = "Web Researcher"
        skill.description = "Researches the web"
        skill.instructions = "# Method\nSearch a lot."
        skill.templates.all.return_value = []
        prompt = build_subagent_system_prompt(specialization_skill=skill)
        self.assertIn("Your specialization: Web Researcher", prompt)
        self.assertIn("Search a lot.", prompt)
        self.assertIn("Researches the web", prompt)


class SpecializationsCatalogueTests(TestCase):
    """The orchestrator-facing 'Sub-agent specializations' catalogue."""

    def test_catalogue_rendered_when_present(self):
        from chat.prompts import build_semi_static_prompt

        prompt = build_semi_static_prompt(specializations=[
            {"slug": "web-researcher", "name": "Web Researcher",
             "emoji": "", "description": "Researches the web"},
        ])
        self.assertIn("# Sub-agent specializations", prompt)
        self.assertIn("web-researcher", prompt)
        self.assertIn('type="<slug>"', prompt)

    def test_no_catalogue_when_absent(self):
        from chat.prompts import build_semi_static_prompt

        prompt = build_semi_static_prompt(specializations=None)
        self.assertNotIn("# Sub-agent specializations", prompt)


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

    def test_running_with_fresh_started_at_not_expired(self):
        """A run that queued for a while before starting gets its full
        execution window — RUNNING staleness is measured from started_at."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="queued then started", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=16),
            started_at=timezone.now() - timedelta(minutes=2),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 0)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.RUNNING)

    def test_running_with_stale_started_at_expired(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="stuck", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=run.pk).update(
            started_at=timezone.now() - timedelta(minutes=11),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 1)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)

    def test_task_swallows_transient_db_error(self):
        """A transient DB outage (e.g. Postgres restarting during maintenance)
        should be logged and skipped, not surface as an unhandled error so the
        beat task stops generating Sentry noise (WILFRED-5K)."""
        from django.db.utils import OperationalError

        from chat.tasks import expire_stale_subagent_runs

        with patch(
            "chat.subagent_limits._expire_stale_runs",
            side_effect=OperationalError("the database system is starting up"),
        ):
            result = expire_stale_subagent_runs()

        self.assertEqual(result, 0)

    def test_task_propagates_non_transient_db_error(self):
        """Real query/data bugs must still surface — only transient connection
        errors are swallowed, not the whole django.db.Error hierarchy."""
        from django.db.utils import ProgrammingError

        from chat.tasks import expire_stale_subagent_runs

        with patch(
            "chat.subagent_limits._expire_stale_runs",
            side_effect=ProgrammingError("column does not exist"),
        ):
            with self.assertRaises(ProgrammingError):
                expire_stale_subagent_runs()


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


class CreateSubagentToolSpecializationTests(TestCase):
    def setUp(self):
        from agent_skills.models import AgentSkill

        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="spec_tool@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        AgentSkill.objects.create(
            slug="web-researcher", name="Web Researcher", level="user",
            created_by=self.user, audience="subagent",
            tool_names=["skill_template_view"],
        )

    def test_unknown_type_rejected(self):
        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(CreateSubagentTool, {"prompt": "task", "type": "nope"}, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("Unknown sub-agent type", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    @patch("chat.tasks.run_subagent_task")
    def test_valid_type_stored_on_run(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="celery-spec")
        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(
            CreateSubagentTool,
            {"prompt": "task", "type": "web-researcher", "timeout": 0},
            ctx,
        )
        self.assertIn(result["status"], ("started", "queued"))
        run = SubAgentRun.objects.get()
        self.assertEqual(run.skill_slug, "web-researcher")


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
            "result": "", "error": "",         }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("# Sub-agent Status", prompt)
        self.assertIn("PENDING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_running_run_shows_in_progress(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "running",
            "prompt": "Analyze documents", "model_tier": "mid",
            "result": "", "error": "",         }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("RUNNING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_completed_shows_delivered_as_message(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Summarize findings", "model_tier": "top",
            "result": "Found 3 key patents.", "error": "",
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("COMPLETED", prompt)
        self.assertIn("delivered as message", prompt.lower())
        self.assertNotIn("Found 3 key patents.", prompt)

    def test_failed_shows_error(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "failed",
            "prompt": "Bad task", "model_tier": "mid",
            "result": "", "error": "LLM provider timeout",         }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("FAILED", prompt)
        self.assertIn("LLM provider timeout", prompt)

    def test_no_check_subagent_status_in_prompt(self):
        from chat.prompts import build_system_prompt
        prompt = build_system_prompt(has_subagent_tool=True)
        self.assertNotIn("check_subagent_status", prompt)

    def test_completed_result_not_in_prompt(self):
        from chat.prompts import build_system_prompt
        long_result = "x" * 10000
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Big task", "model_tier": "mid",
            "result": long_result, "error": "",
        }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertNotIn(long_result, prompt)
        self.assertIn("delivered as message", prompt.lower())


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
        mock_svc.return_value.run_via_stream.return_value = mock_response

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
    def test_task_handles_failure_sets_failed(self, mock_prefs, mock_svc):
        """On failure, run_subagent sets FAILED directly."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = RuntimeError("Provider down")

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="Do research",
        )

        from chat.tasks import run_subagent_task
        with self.assertRaises(RuntimeError):
            run_subagent_task(str(run.id))

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("Provider down", run.error)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_task_retries_on_transient_llm_error(self, mock_prefs, mock_svc):
        """A transient LLM error triggers self.retry and leaves the run RUNNING
        (not FAILED) so the retry can re-enter it cleanly."""
        from celery.exceptions import Retry
        from llm.service.errors import LLMOverloadedError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMOverloadedError(
            "overloaded", error_code="overloaded",
        )

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="Do research",
        )

        from chat.tasks import run_subagent_task
        with patch.object(run_subagent_task, "retry", side_effect=Retry()) as mock_retry:
            with self.assertRaises(Retry):
                run_subagent_task(str(run.id))
        mock_retry.assert_called_once()

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.RUNNING)
        self.assertEqual(run.error, "")
        self.assertIsNone(run.completed_at)


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
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", model_tier="mid",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.model_used, "openai/gpt-5-mini")
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_passes_data_room_ids(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", data_room_ids=[1, 2],
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        # Verify the request included data room IDs in context
        call_args = mock_svc.return_value.run_via_stream.call_args
        request = call_args[0][1]
        self.assertEqual(request.context.data_room_ids, [1, 2])

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_creates_hidden_message_on_completion(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Research findings here."
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.001
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        msg = ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        ).first()
        self.assertIsNotNone(msg)
        self.assertEqual(msg.role, "user")
        self.assertIsNone(msg.tool_call_id)
        short_id = str(run.id)[:8]
        self.assertTrue(msg.content.startswith(f"[Sub-agent result: {short_id}]"))
        self.assertIn("Research findings here.", msg.content)
        self.assertTrue(msg.is_hidden_from_user)
        self.assertEqual(msg.metadata["source"], "subagent")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_no_message_created_for_empty_result(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = []
        mock_response.usage.total_tokens = 50
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        msg_count = ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        ).count()
        self.assertEqual(msg_count, 0)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_empty_result_marks_failed(self, mock_prefs, mock_svc):
        """If response has no content, run is marked FAILED."""
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = [{"id": "c1", "name": "web_search"}]
        mock_response.usage.total_tokens = 200
        mock_response.usage.cost_usd = 0.002
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="research task",
        )

        from chat.subagent_service import run_subagent
        with self.assertLogs("chat.subagent_service", level="WARNING") as cm:
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run.result, "")
        self.assertIn("no text output", run.error)
        self.assertTrue(any("no content" in msg for msg in cm.output))

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_failure_sets_failed(self, mock_prefs, mock_svc):
        """run_subagent should set FAILED on failure."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = RuntimeError("Transient error")

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task",
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(RuntimeError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("Transient error", run.error)
        self.assertIsNotNone(run.completed_at)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_transient_error_leaves_run_running(self, mock_prefs, mock_svc):
        """A transient LLM error must NOT be recorded as FAILED — the run is left
        RUNNING so a Celery retry re-enters it. (Writing FAILED would make the
        retry a no-op via the RUNNING-transition guard.)"""
        from llm.service.errors import LLMRateLimitError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMRateLimitError(
            "rate limited", error_code="rate_limited",
        )

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(LLMRateLimitError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.RUNNING)
        self.assertEqual(run.error, "")
        self.assertIsNone(run.completed_at)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_uses_streaming_not_blocking_run(self, mock_prefs, mock_svc):
        """run_subagent must drive the streaming path (run_via_stream) with
        request.stream=True, never the non-streaming run() — so a long generation
        can't trip a non-streaming read timeout."""
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 10
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        mock_svc.return_value.run_via_stream.assert_called_once()
        mock_svc.return_value.run.assert_not_called()
        pipeline_id, request = mock_svc.return_value.run_via_stream.call_args[0]
        self.assertEqual(pipeline_id, "simple_chat")
        self.assertTrue(request.stream)


# ---------------------------------------------------------------------------
# is_retryable_subagent_error predicate
# ---------------------------------------------------------------------------

class IsRetryableSubagentErrorTests(TestCase):
    def test_transient_errors_are_retryable(self):
        from chat.subagent_service import is_retryable_subagent_error
        from llm.service.errors import (
            LLMConnectionError, LLMOverloadedError, LLMRateLimitError, LLMTimeoutError,
        )
        for exc in (
            LLMRateLimitError("x"), LLMOverloadedError("x"),
            LLMTimeoutError("x"), LLMConnectionError("x"),
            ConnectionError("x"), TimeoutError("x"), OSError("x"),
        ):
            self.assertTrue(
                is_retryable_subagent_error(exc), type(exc).__name__,
            )

    def test_terminal_errors_are_not_retryable(self):
        from chat.subagent_service import is_retryable_subagent_error
        from llm.service.errors import (
            LLMAuthError, LLMProviderError, LLMRequestTooLargeError,
        )
        for exc in (
            RuntimeError("x"), ValueError("x"),
            LLMAuthError("x"), LLMRequestTooLargeError("x"), LLMProviderError("x"),
        ):
            self.assertFalse(
                is_retryable_subagent_error(exc), type(exc).__name__,
            )

    def test_operational_error_too_many_connections_is_terminal(self):
        from django.db.utils import OperationalError
        from chat.subagent_service import is_retryable_subagent_error
        self.assertFalse(is_retryable_subagent_error(
            OperationalError("FATAL: too many connections for role")))
        self.assertTrue(is_retryable_subagent_error(
            OperationalError("the database system is starting up")))


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

    def test_on_failure_keeps_existing_failure_reason(self):
        """A run already FAILED (e.g. cancelled by the user) must not have its
        error clobbered by the task exception."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="cancelled task", status=SubAgentRun.Status.FAILED,
            error="Cancelled by user.",
        )

        from chat.tasks import run_subagent_task
        run_subagent_task.on_failure(
            RuntimeError("Worker terminated"),
            "fake-task-id",
            [str(run.id)],
            {},
            None,
        )

        run.refresh_from_db()
        self.assertEqual(run.error, "Cancelled by user.")

    def test_on_failure_captures_and_flushes_to_sentry(self):
        """Terminal failures are captured + flushed explicitly so they can't be
        lost by the LoggingIntegration (regression: prod run b268f675, a 562s
        timeout whose error logs never produced a Sentry event)."""
        from llm.service.errors import LLMProviderError

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="doomed task", status=SubAgentRun.Status.RUNNING,
        )
        exc = LLMProviderError("Anthropic timed out", error_code="timeout")

        from chat.tasks import run_subagent_task
        with patch("sentry_sdk.capture_exception") as cap, \
                patch("sentry_sdk.flush") as flush:
            run_subagent_task.on_failure(
                exc, "fake-task-id", [str(run.id)], {}, None,
            )

        cap.assert_called_once_with(exc)
        flush.assert_called_once()

    def test_on_failure_survives_sentry_error(self):
        """A Sentry hiccup must never break the failure-marking path."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="doomed task", status=SubAgentRun.Status.RUNNING,
        )

        from chat.tasks import run_subagent_task
        with patch("sentry_sdk.capture_exception", side_effect=RuntimeError("sentry down")):
            run_subagent_task.on_failure(
                RuntimeError("Final failure"), "fake-task-id", [str(run.id)], {}, None,
            )

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)


# ---------------------------------------------------------------------------
# Cancellation vs. completion races (fix: stop must not be overwritten)
# ---------------------------------------------------------------------------

class RunSubagentCancellationTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cancel@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_cancelled_pending_run_not_resurrected(self, mock_prefs, mock_svc):
        """A run cancelled while PENDING must not be executed at all."""
        mock_prefs.return_value = _prefs()
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.FAILED,
            error="Cancelled by user.",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run.error, "Cancelled by user.")
        mock_svc.return_value.run_via_stream.assert_not_called()

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_cancelled_mid_run_result_discarded(self, mock_prefs, mock_svc):
        """If the user cancels while the LLM call is in flight, the late result
        must not flip the run back to COMPLETED or create a hidden message."""
        mock_prefs.return_value = _prefs()
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )

        def cancel_then_respond(*args, **kwargs):
            # Simulate the user hitting stop mid-run
            SubAgentRun.objects.filter(pk=run.pk).update(
                status=SubAgentRun.Status.FAILED, error="Cancelled by user.",
            )
            mock_response = MagicMock()
            mock_response.message.content = "Late findings."
            mock_response.usage.total_tokens = 100
            mock_response.usage.cost_usd = 0.001
            return mock_response

        mock_svc.return_value.run_via_stream.side_effect = cancel_then_respond

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertEqual(run.error, "Cancelled by user.")
        msg_count = ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        ).count()
        self.assertEqual(msg_count, 0)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_started_at_set_when_running(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 10
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertIsNotNone(run.started_at)
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)


# ---------------------------------------------------------------------------
# Orchestrator claim of unreported results (fix: two-tab duplicate seeds)
# ---------------------------------------------------------------------------

class ClaimUnreportedSubagentsTests(TransactionTestCase):
    def setUp(self):
        from chat.consumers import ChatConsumer

        self.user = User.objects.create_user(email="claim@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user

    @database_sync_to_async
    def _make_completed_run(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.COMPLETED,
            result="Findings.",
        )
        ChatMessage.objects.create(
            thread=self.thread, role="user",
            content=f"[Sub-agent result: {str(run.id)[:8]}]\nFindings.",
            metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            is_hidden_from_user=True,
        )
        return run

    async def _claim(self):
        return await self.consumer._claim_unreported_subagents(str(self.thread.id))

    @database_sync_to_async
    def _backdate_reported_at(self, run, minutes):
        SubAgentRun.objects.filter(pk=run.pk).update(
            reported_at=timezone.now() - timedelta(minutes=minutes),
        )

    async def test_first_claim_wins_second_loses(self):
        run = await self._make_completed_run()

        self.assertTrue(await self._claim())
        # Second claim (e.g. the same thread open in another tab) must lose.
        self.assertFalse(await self._claim())

        await database_sync_to_async(run.refresh_from_db)()
        self.assertIsNotNone(run.reported_at)

    async def test_stale_claim_is_reclaimable(self):
        """If the seeded stream died (no assistant response), the claim expires
        after the lease and the result is retried."""
        run = await self._make_completed_run()
        self.assertTrue(await self._claim())

        await self._backdate_reported_at(run, 6)
        self.assertTrue(await self._claim())

    async def test_reported_run_with_response_not_reclaimed(self):
        """Once an assistant response exists, the run is permanently reported —
        even after the lease window."""
        run = await self._make_completed_run()
        self.assertTrue(await self._claim())

        await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="assistant",
            content="Here is what the sub-agent found...",
            metadata={"subagent_response": True},
        )
        await self._backdate_reported_at(run, 6)
        self.assertFalse(await self._claim())

    async def test_no_completed_runs_returns_false(self):
        self.assertFalse(await self._claim())
