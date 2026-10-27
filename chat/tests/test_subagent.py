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
    running_silence_limit,
    _expire_stale_runs,
    check_subagent_limits,
    create_subagent_run_if_allowed,
)
from chat.subagent_prompts import build_subagent_system_prompt
from chat.subagent_service import resolve_subagent_model, resolve_subagent_tools
from chat.subagent_tool import ContinueSubagentTool, CreateSubagentTool
from chat.tool_groups import DATA_ROOM_TOOL_NAMES
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
            "subagent_canvas_load_template",
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
            model_tier="medium",
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

    def test_low_maps_to_cheap_slot(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("low", prefs), "openai/gpt-5-nano")

    def test_medium_maps_to_mid_slot(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("medium", prefs), "openai/gpt-5-mini")

    def test_high_maps_to_top_slot(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("high", prefs), "openai/gpt-5")

    def test_org_override_honored_for_low(self):
        prefs = _prefs(feature_models={"subagent_low": "openai/gpt-5-custom-low"})
        self.assertEqual(resolve_subagent_model("low", prefs), "openai/gpt-5-custom-low")

    def test_legacy_tier_names_resolve_as_medium(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("mid", prefs), "openai/gpt-5-mini")
        self.assertEqual(resolve_subagent_model("top", prefs), "openai/gpt-5-mini")

    def test_invalid_tier_defaults_to_medium(self):
        prefs = _prefs()
        self.assertEqual(resolve_subagent_model("invalid", prefs), "openai/gpt-5-mini")

    def test_org_override_honored_for_top(self):
        prefs = _prefs(feature_models={"subagent_high": "openai/gpt-5-custom-top"})
        self.assertEqual(resolve_subagent_model("high", prefs), "openai/gpt-5-custom-top")

    def test_org_override_honored_for_mid(self):
        prefs = _prefs(feature_models={"subagent_medium": "openai/gpt-5-custom-mid"})
        self.assertEqual(resolve_subagent_model("medium", prefs), "openai/gpt-5-custom-mid")

    def test_no_subagent_override_falls_back_to_slot_model(self):
        # feature_models present but without the sub-agent keys -> slot fallback.
        prefs = _prefs(feature_models={"chat": "openai/gpt-5"})
        self.assertEqual(resolve_subagent_model("high", prefs), "openai/gpt-5")
        self.assertEqual(resolve_subagent_model("medium", prefs), "openai/gpt-5-mini")


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
            "canvas_write", "canvas_edit", "canvas_read",
            "canvas_save_to_document", "chat_subagent_create",
            "chat_skill_attach", "chat_skill_detach", "skill_create",
            "skill_resource_load",
        ]:
            self.assertNotIn(name, tools)

    def test_keeps_doc_tools_with_data_rooms(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[1])
        self.assertIn("document_search", tools)
        self.assertIn("document_read", tools)

    def test_removes_all_data_room_tools_without_data_rooms(self):
        prefs = _prefs(allowed_subagent_tools=[
            *DATA_ROOM_TOOL_NAMES,
            "web_fetch",
        ])
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertTrue(DATA_ROOM_TOOL_NAMES.isdisjoint(tools))
        self.assertIn("web_fetch", tools)

    def test_keeps_web_tools(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertIn("web_fetch", tools)
        self.assertIn("web_search", tools)

    def test_no_specialization_hides_loader_and_skill_tools(self):
        """A general sub-agent gets neither specialization tools nor its loader."""
        prefs = _prefs(allowed_specializations=[{
            "id": "1", "slug": "web-researcher", "name": "Web Researcher",
            "emoji": "", "description": "", "tool_names": ["specialized_tool"],
        }])
        tools = resolve_subagent_tools(prefs, data_room_ids=[])
        self.assertNotIn("specialized_tool", tools)
        self.assertNotIn("subagent_canvas_load_template", tools)

    def test_specialization_adds_its_tools(self):
        """A spawned specialization contributes its (already filtered) tools."""
        prefs = _prefs(allowed_specializations=[{
            "id": "1", "slug": "web-researcher", "name": "Web Researcher",
            "emoji": "", "description": "", "tool_names": ["specialized_tool"],
        }])
        tools = resolve_subagent_tools(
            prefs, data_room_ids=[], specialization_slug="web-researcher",
        )
        self.assertIn("specialized_tool", tools)
        self.assertIn("subagent_canvas_load_template", tools)

    def test_unknown_specialization_adds_nothing_and_hides_loader(self):
        prefs = _prefs()
        tools = resolve_subagent_tools(
            prefs, data_room_ids=[], specialization_slug="does-not-exist",
        )
        self.assertNotIn("subagent_canvas_load_template", tools)

    def test_specialization_cannot_restore_data_room_tools_without_rooms(self):
        prefs = _prefs(allowed_specializations=[{
            "id": "1", "slug": "document-worker", "name": "Document Worker",
            "emoji": "", "description": "", "tool_names": ["document_edit"],
        }])
        tools = resolve_subagent_tools(
            prefs, data_room_ids=[], specialization_slug="document-worker",
        )
        self.assertNotIn("document_edit", tools)


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
        self.assertIn("Deliver your findings in your working canvas", prompt)
        self.assertIn('"Not covered" list', prompt)
        self.assertIn("Never repeat what is in the canvas", prompt)

    def test_round_budget_section_only_with_a_budget(self):
        self.assertNotIn("# Round budget", build_subagent_system_prompt())
        prompt = build_subagent_system_prompt(round_budget=6)
        self.assertIn("# Round budget\nYou have a budget of 6 rounds.", prompt)
        self.assertIn("Round 6/6 is your final round with tools", prompt)
        self.assertNotIn("Working canvas\" below", prompt)

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
        self.assertIn("subagent_task_update", prompt)
        self.assertNotIn("chat_task_update", prompt)

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
        # Nested under "## Specialization instructions:".
        self.assertIn("\n### Method\n", prompt)
        self.assertNotIn("\n# Method\n", prompt)


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
            model_tier="medium",
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
        # Handed to Celery 8 minutes ago and never picked up. (A PENDING row
        # with no dispatched_at is merely waiting in line and is not stale.)
        SubAgentRun.objects.filter(pk=run.pk).update(
            created_at=timezone.now() - timedelta(minutes=11),
            dispatched_at=timezone.now() - timedelta(minutes=8),
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
            created_at=timezone.now() - timedelta(seconds=running_silence_limit("") + 60),
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
            created_at=timezone.now() - timedelta(seconds=running_silence_limit("") + 60),
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
            started_at=timezone.now() - timedelta(seconds=running_silence_limit("") + 60),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 1)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)

    def test_long_running_run_with_recent_heartbeat_not_expired(self):
        """No wall clock: a run that started long ago but is still making
        progress (recent heartbeat) is left alone."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="deep research", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=run.pk).update(
            started_at=timezone.now() - timedelta(hours=2),
            last_activity_at=timezone.now() - timedelta(seconds=30),
        )
        self.assertEqual(_expire_stale_runs(), 0)

    def test_silence_limit_is_one_request_or_one_tool_call_plus_margin(self):
        """With the heartbeat beating on every chunk and retry wait, silence
        means one stuck step: the longer of one model request and the slowest
        tool, plus the margin — not four attempts plus a tool (46 min on Gemini)."""
        from chat.subagent_limits import STALE_SILENCE_MARGIN_SECONDS, _longest_tool_timeout
        from llm.core.model_factory import request_timeout_for_model

        longest_tool = _longest_tool_timeout()
        for model in ("openai/gpt-6-luna", "gemini/gemini-4-argon"):
            expected = max(request_timeout_for_model(model), longest_tool) + STALE_SILENCE_MARGIN_SECONDS
            self.assertEqual(running_silence_limit(model), expected)
            self.assertEqual(running_silence_limit(model, longest_tool), expected)
        self.assertEqual(running_silence_limit("gemini/gemini-4-argon", 30.0), 300 + 120)
        self.assertLessEqual(running_silence_limit("gemini/gemini-4-argon"), 15 * 60)

    def test_sweep_scans_the_tool_registry_once(self):
        from chat import subagent_limits

        for _ in range(3):
            run = SubAgentRun.objects.create(
                thread=self.thread, user=self.user, prompt="x", model_used="openai/gpt-6-luna",
                status=SubAgentRun.Status.RUNNING,
            )
            SubAgentRun.objects.filter(pk=run.pk).update(
                started_at=timezone.now() - timedelta(hours=2),
                last_activity_at=timezone.now() - timedelta(hours=2),
            )
        with patch.object(
            subagent_limits, "_longest_tool_timeout", wraps=subagent_limits._longest_tool_timeout,
        ) as scan:
            self.assertEqual(_expire_stale_runs(), 3)
        self.assertEqual(scan.call_count, 1)

    def test_silence_limit_follows_the_model_request_timeout(self):
        """An Argon run (300 s request timeout) may sit silent far longer than an
        OpenAI one (120 s) before it counts as stale."""
        openai_limit = running_silence_limit("openai/gpt-6-luna")
        gemini_limit = running_silence_limit("gemini/gemini-4-argon")
        self.assertGreater(gemini_limit, openai_limit)
        quiet = timedelta(seconds=(openai_limit + gemini_limit) / 2)
        runs = {}
        for model in ("openai/gpt-6-luna", "gemini/gemini-4-argon"):
            runs[model] = SubAgentRun.objects.create(
                thread=self.thread, user=self.user, prompt="x", model_used=model,
                status=SubAgentRun.Status.RUNNING,
            )
            SubAgentRun.objects.filter(pk=runs[model].pk).update(
                started_at=timezone.now() - timedelta(hours=1),
                last_activity_at=timezone.now() - quiet,
            )
        self.assertEqual(_expire_stale_runs(), 1)
        runs["openai/gpt-6-luna"].refresh_from_db()
        runs["gemini/gemini-4-argon"].refresh_from_db()
        self.assertEqual(runs["openai/gpt-6-luna"].status, SubAgentRun.Status.FAILED)
        self.assertEqual(runs["gemini/gemini-4-argon"].status, SubAgentRun.Status.RUNNING)

    def test_heartbeat_is_throttled_and_stamps_running_rows(self):
        from chat import subagent_service

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="x", status=SubAgentRun.Status.RUNNING,
        )
        with patch.object(subagent_service, "HEARTBEAT_MIN_INTERVAL_SECONDS", 0):
            subagent_service._heartbeat(run.pk)()
        run.refresh_from_db()
        self.assertIsNotNone(run.last_activity_at)
        stamped = run.last_activity_at
        beat = subagent_service._heartbeat(run.pk)  # default interval: first call too soon
        beat()
        run.refresh_from_db()
        self.assertEqual(run.last_activity_at, stamped)

    def test_heartbeat_writes_live_round_count(self):
        from chat import subagent_service

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="x", status=SubAgentRun.Status.RUNNING,
        )
        obs = {}
        beat = subagent_service._heartbeat(run.pk, rounds=lambda: obs.get("rounds"))
        with patch.object(subagent_service, "HEARTBEAT_MIN_INTERVAL_SECONDS", 0):
            beat()  # no round counted yet: rounds_used stays unset
            run.refresh_from_db()
            self.assertIsNone(run.rounds_used)
            obs["rounds"] = 4
            beat()
        run.refresh_from_db()
        self.assertEqual(run.rounds_used, 4)

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
# ContinueSubagentTool + result header + continuation brief
# ---------------------------------------------------------------------------

class ContinueSubagentToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="continue@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.previous = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="Find prior art for X",
            reason="Prior art for X", status=SubAgentRun.Status.COMPLETED,
            skill_slug="patent-searcher", model_tier="low", round_budget="standard",
            rounds_used=16, budget_exhausted=True,
            result="Summary. Not covered: family check.",
            canvas="# Report\nFindings", canvas_title="Report", scratchpad="note 1",
            task_plan=[{"id": "a", "title": "Search", "status": "completed", "order": 0}],
            data_room_ids=[3], shared_canvases=[{"title": "Draft", "content": "d"}],
        )

    def _continue(self, **args):
        ctx = _ctx(self.user.pk, self.thread.id)
        return _invoke(ContinueSubagentTool, {"round_budget": "quick", **args}, ctx)

    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_creates_linked_run_carrying_work_forward(self, mock_task, _mock_notify):
        mock_task.delay.return_value = MagicMock(id="t1")
        short_id = str(self.previous.id)[:8]
        result = self._continue(run_id=short_id, instructions="Check the families.")

        self.assertEqual(result["status"], "started")
        run = SubAgentRun.objects.exclude(pk=self.previous.pk).get()
        self.assertEqual(run.continued_from_id, self.previous.id)
        self.assertEqual(run.chain_depth, 1)
        self.assertEqual(run.round_budget, "quick")
        self.assertEqual(run.model_tier, "low")
        for field in ("prompt", "skill_slug", "canvas", "canvas_title", "scratchpad",
                      "task_plan", "data_room_ids", "shared_canvases", "reason"):
            self.assertEqual(getattr(run, field), getattr(self.previous, field), field)
        self.assertEqual(run.continuation_instructions, "Check the families.")

    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_model_tier_override(self, mock_task, _mock_notify):
        mock_task.delay.return_value = MagicMock(id="t1")
        self._continue(run_id=str(self.previous.id), model_tier="high")
        run = SubAgentRun.objects.exclude(pk=self.previous.pk).get()
        self.assertEqual(run.model_tier, "high")

    def test_rejects_running_run(self):
        SubAgentRun.objects.filter(pk=self.previous.pk).update(status=SubAgentRun.Status.RUNNING)
        result = self._continue(run_id=str(self.previous.id)[:8])
        self.assertEqual(result["status"], "error")
        self.assertIn("still running", result["message"])

    def test_rejects_run_of_another_thread(self):
        other = ChatThread.objects.create(created_by=self.user)
        ctx = _ctx(self.user.pk, other.id)
        result = _invoke(ContinueSubagentTool, {
            "run_id": str(self.previous.id)[:8], "round_budget": "quick",
        }, ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("No sub-agent run", result["message"])

    def test_rejects_unknown_id(self):
        result = self._continue(run_id="zzzzzzzz")
        self.assertEqual(result["status"], "error")

    def test_chain_cap(self):
        SubAgentRun.objects.filter(pk=self.previous.pk).update(chain_depth=9)
        result = self._continue(run_id=str(self.previous.id)[:8])
        self.assertEqual(result["status"], "error")
        self.assertIn("the limit", result["message"])

    @patch("chat.subagent_limits.SUBAGENT_MAX_PER_USER", 1)
    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_counts_against_per_user_cap(self, mock_task, _mock_notify):
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="busy", status=SubAgentRun.Status.RUNNING,
        )
        result = self._continue(run_id=str(self.previous.id)[:8])
        self.assertEqual(result["status"], "error")
        self.assertIn("too many", result["message"])


class ResultHeaderTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="header@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _run(self, **kw):
        now = timezone.now()
        return SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.COMPLETED,
            started_at=now - timedelta(seconds=252), completed_at=now, **kw,
        )

    def test_finished_header_shows_rounds_time_and_continue_hint(self):
        from chat.subagent_service import result_header

        run = self._run(round_budget="standard", rounds_used=9)
        short = str(run.id)[:8]
        self.assertEqual(
            result_header(run),
            f"[Sub-agent result: {short} — finished (9/20 rounds (standard), 4:12). If more "
            f'work is needed, you may continue it with chat_subagent_continue(run_id="{short}").]',
        )

    def test_forced_header_when_budget_ran_out(self):
        """Neutral wording: the forced reply may still be complete."""
        from chat.subagent_service import result_header

        run = self._run(round_budget="quick", rounds_used=11, budget_exhausted=True)
        short = str(run.id)[:8]
        self.assertEqual(
            result_header(run),
            f"[Sub-agent result: {short} — round budget reached: its final reply was forced "
            "after 10/10 rounds (quick), 4:12. Check its reply for anything it did not cover; "
            "if more work is needed, consider continuing it.]",
        )
        self.assertNotIn("PARTIAL", result_header(run))

    def test_forced_header_and_brief_when_output_was_cut_off(self):
        """A truncation stop (two cut-off rounds) also forced the final reply,
        but it must not be blamed on the round budget."""
        from chat.subagent_service import result_header, task_message

        previous = self._run(round_budget="standard", rounds_used=3, stop_reason="truncation",
                             result="Summary. Not covered: Z.")
        header = result_header(previous)
        self.assertIn(
            "— its output was cut off repeatedly: its final reply was forced after "
            "3/20 rounds (standard), 4:12.",
            header,
        )
        self.assertNotIn("budget", header)
        run = self._run(round_budget="standard", continued_from=previous, chain_depth=1)
        self.assertIn(
            "A previous sub-agent had its output cut off repeatedly (its reply was forced) "
            "after 3/20 rounds (standard), 4:12.",
            task_message(run),
        )

    def test_continuation_header_and_brief(self):
        from chat.subagent_service import result_header, task_message

        previous = self._run(round_budget="quick", rounds_used=11, budget_exhausted=True,
                             result="Summary. Not covered: Y.")
        run = self._run(round_budget="standard", rounds_used=3, continued_from=previous,
                        chain_depth=1, continuation_instructions="Focus on Y.")
        self.assertIn("(continuation 1) — finished", result_header(run))
        message = task_message(run)
        # The original task and the continuation brief are separate sections.
        self.assertTrue(message.startswith("# Original task\nt\n\n# Continuing earlier work\n"))
        self.assertIn(
            "A previous sub-agent ran out of rounds (its reply was forced) after 10/10 rounds "
            "(quick), 4:12. Its working canvas and scratchpad were carried over to you as a "
            "starting point. Continue where it left off.",
            message,
        )
        self.assertIn("## Its final reply\nSummary. Not covered: Y.", message)
        self.assertIn("## Direction from the orchestrator\nFocus on Y.", message)
        self.assertEqual(task_message(previous), "t")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_continuation_keeps_carried_canvas_and_gets_brief(self, mock_prefs, mock_svc):
        """run_subagent: a continuation's carried-over canvas is not replaced by the
        specialization template, and its task message carries the brief."""
        mock_prefs.return_value = _prefs()
        response = MagicMock()
        response.message.content = "Done"
        response.usage.total_tokens = 1
        response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = response
        previous = self._run(result="Summary.", canvas="# Report\nold work")
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", skill_slug="web-researcher",
            canvas="# Report\nold work", continued_from=previous, chain_depth=1,
        )
        skill = MagicMock()
        skill.templates.all.return_value.first.return_value = MagicMock(content="# Template", name="T")
        skill.instructions, skill.description, skill.name = "", "", "Web"

        from chat.subagent_service import run_subagent
        with patch("chat.subagent_service.get_run_specialization_skill", return_value=skill):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.canvas, "# Report\nold work")
        _, request = mock_svc.return_value.run_via_stream.call_args[0]
        self.assertTrue(request.messages[1].content.startswith(
            "# Original task\ntask\n\n# Continuing earlier work"))

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_prompt_shows_own_plan_not_the_threads(self, mock_prefs, mock_svc):
        """The thread's ThreadTask rows are the orchestrator's plan: a sub-agent
        never sees them; a continuation sees its carried-over run plan."""
        from chat.models import ThreadTask
        from chat.subagent_service import run_subagent

        mock_prefs.return_value = _prefs()
        response = MagicMock()
        response.message.content = "Done"
        response.usage.total_tokens = 1
        response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = response
        ThreadTask.objects.create(thread=self.thread, title="Orchestrator step", status="pending")

        fresh = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")
        run_subagent(fresh.id)
        _, request = mock_svc.return_value.run_via_stream.call_args[0]
        self.assertNotIn("Orchestrator step", request.messages[0].content)
        self.assertNotIn("# Your Task Plan", request.messages[0].content)

        continued = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", continued_from=fresh,
            chain_depth=1,
            task_plan=[{"id": "x", "title": "Check families", "status": "pending", "order": 0}],
        )
        run_subagent(continued.id)
        _, request = mock_svc.return_value.run_via_stream.call_args[0]
        self.assertIn("# Your Task Plan", request.messages[0].content)
        self.assertIn("[ ] Check families", request.messages[0].content)
        self.assertNotIn("Orchestrator step", request.messages[0].content)


class CreateSubagentToolSpecializationTests(TestCase):
    def setUp(self):
        from agent_skills.models import AgentSkill

        from agent_skills.resources import compute_skill_content_hash

        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="spec_tool@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        skill = AgentSkill.objects.create(
            slug="web-researcher", name="Web Researcher", level="user",
            created_by=self.user, audience="subagent",
            tool_names=["skill_resource_view"],
        )
        # chat_subagent_create applies the same safety-scan gate as
        # chat_skill_attach: a user skill is on offer only once its scan passed.
        skill.scan_state = AgentSkill.ScanState.APPROVED
        skill.approved_content_hash = compute_skill_content_hash(skill)
        skill.save(update_fields=["scan_state", "approved_content_hash"])

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
            {"prompt": "task", "type": "web-researcher"},
            ctx,
        )
        self.assertIn(result["status"], ("started", "queued"))
        run = SubAgentRun.objects.get()
        self.assertEqual(run.skill_slug, "web-researcher")

    @patch("chat.tasks.run_subagent_task")
    def test_skills_only_type_accepted_without_a_naming_skill(self, mock_task):
        # always_available only decides what the orchestrator is told about;
        # an enabled skills-only specialization is still accepted (lenient).
        from agent_skills.models import AgentSkill

        AgentSkill.objects.filter(slug="web-researcher").update(always_available=False)
        mock_task.delay.return_value = MagicMock(id="celery-spec")
        ctx = _ctx(self.user.pk, self.thread.id)
        result = _invoke(
            CreateSubagentTool, {"prompt": "task", "type": "web-researcher"}, ctx,
        )
        self.assertIn(result["status"], ("started", "queued"))
        self.assertEqual(SubAgentRun.objects.get().skill_slug, "web-researcher")


class CreateSubagentToolBackgroundTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="nonblock@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.tasks.run_subagent_task")
    def test_create_returns_started(self, mock_task):
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

    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_reason_saved_and_panel_notified(self, mock_task, mock_notify):
        mock_task.delay.return_value = MagicMock(id="t1")

        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(CreateSubagentTool, {
            "prompt": "task", "reason": "  Check the licence terms  ",
        }, ctx)

        run = SubAgentRun.objects.first()
        self.assertEqual(run.reason, "Check the licence terms")
        mock_notify.assert_called_once_with(str(self.thread.id), str(run.id))

    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_invalid_tier_defaults_to_medium(self, mock_task, _mock_notify):
        mock_task.delay.return_value = MagicMock(id="t1")

        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(CreateSubagentTool, {
            "prompt": "task", "model_tier": "invalid",
        }, ctx)

        run = SubAgentRun.objects.first()
        self.assertEqual(run.model_tier, "medium")


class ActiveSubagentRunsPayloadTests(TestCase):
    """The run list behind the user's sub-agent panel."""

    def setUp(self):
        self.user = User.objects.create_user(email="panel@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _payload(self):
        from chat.subagent_service import active_subagent_runs_payload
        return active_subagent_runs_payload(self.thread.id)

    def test_states_and_timestamps(self):
        from django.utils import timezone

        now = timezone.now()
        queued = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="q", reason="Queued one",
            status=SubAgentRun.Status.PENDING,
        )
        dispatched = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="d", reason="Dispatched one",
            status=SubAgentRun.Status.PENDING, dispatched_at=now,
        )
        running = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="r", reason="Running one",
            status=SubAgentRun.Status.RUNNING, dispatched_at=now, started_at=now,
        )
        for status in (SubAgentRun.Status.COMPLETED, SubAgentRun.Status.FAILED):
            SubAgentRun.objects.create(
                thread=self.thread, user=self.user, prompt="done", status=status,
            )
        other = ChatThread.objects.create(created_by=self.user)
        SubAgentRun.objects.create(
            thread=other, user=self.user, prompt="elsewhere",
            status=SubAgentRun.Status.RUNNING,
        )

        payload = self._payload()
        runs = {r["id"]: r for r in payload["runs"]}
        self.assertEqual(set(runs), {str(queued.id), str(dispatched.id), str(running.id)})
        self.assertEqual(runs[str(queued.id)]["state"], "queued")
        self.assertIsNone(runs[str(queued.id)]["started_at"])
        # Dispatched but not yet picked up by the worker counts as running.
        self.assertEqual(runs[str(dispatched.id)]["state"], "running")
        self.assertEqual(runs[str(dispatched.id)]["started_at"], now.isoformat())
        self.assertEqual(runs[str(running.id)]["started_at"], now.isoformat())
        self.assertEqual(runs[str(running.id)]["reason"], "Running one")
        self.assertTrue(payload["server_now"])
        # Oldest first.
        self.assertEqual(payload["runs"][0]["id"], str(queued.id))

    def test_reason_falls_back_to_prompt(self):
        SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="  Look up\n" + "x" * 300,
            status=SubAgentRun.Status.RUNNING,
        )
        reason = self._payload()["runs"][0]["reason"]
        self.assertTrue(reason.startswith("Look up x"))
        self.assertTrue(reason.endswith("…"))
        self.assertEqual(len(reason), 121)

    def test_empty(self):
        self.assertEqual(self._payload()["runs"], [])


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
            "prompt": "Research patent claims", "model_tier": "medium",
            "result": "", "error": "",         }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("# Sub-agent Status", prompt)
        self.assertIn("PENDING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_running_run_shows_in_progress(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "running",
            "prompt": "Analyze documents", "model_tier": "medium",
            "result": "", "error": "",         }]
        prompt = build_system_prompt(has_subagent_tool=True, subagent_runs=runs)
        self.assertIn("RUNNING", prompt)
        self.assertIn("Still in progress", prompt)

    def test_completed_shows_delivered_as_message(self):
        from chat.prompts import build_system_prompt
        runs = [{
            "id": uuid.uuid4(), "status": "completed",
            "prompt": "Summarize findings", "model_tier": "high",
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
            "prompt": "Bad task", "model_tier": "medium",
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
            "prompt": "Big task", "model_tier": "medium",
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
        from llm.service.errors import LLMTimeoutError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMTimeoutError(
            "timed out", error_code="timeout",
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

    @patch("chat.tasks._notify_consumer")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_task_does_not_restart_the_run_on_overload(self, mock_prefs, mock_svc, mock_notify):
        """The provider already backed off before raising an overload / rate
        limit: no whole-run self.retry, the run ends FAILED."""
        from llm.service.errors import LLMOverloadedError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMOverloadedError(
            "overloaded", error_code="overloaded",
        )
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="Do research",
        )

        from chat.tasks import run_subagent_task
        with patch.object(run_subagent_task, "retry") as mock_retry:
            with self.assertRaises(LLMOverloadedError):
                run_subagent_task(str(run.id))
        mock_retry.assert_not_called()

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)


# ---------------------------------------------------------------------------
# run_subagent service tests
# ---------------------------------------------------------------------------

class RunSubagentServiceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="svc@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    @patch("chat.tasks.notify_subagent_status")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_start_notifies_sub_agent_panel(self, mock_prefs, mock_svc, mock_notify):
        """Entering RUNNING pushes a panel update (Queued → live timer)."""
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 1
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        mock_notify.assert_called_once_with(str(self.thread.id), str(run.id))

    @patch("chat.tasks.notify_subagent_status")
    def test_terminal_run_does_not_notify(self, mock_notify):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            status=SubAgentRun.Status.COMPLETED,
        )
        from chat.subagent_service import run_subagent
        run_subagent(run.id)
        mock_notify.assert_not_called()

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
            prompt="task", model_tier="medium",
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
    def test_tier_budget_sets_max_context_tokens(self, mock_prefs, mock_svc):
        """An org budget for the run's tier is passed into the request params so
        the mid-turn pruner sizes to it (mid tier -> mid budget)."""
        mock_prefs.return_value = _prefs(subagent_context_budgets={"medium": 120_000})
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", model_tier="medium",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        request = mock_svc.return_value.run_via_stream.call_args[0][1]
        self.assertEqual(request.params.get("max_context_tokens"), 120_000)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_no_budget_falls_back_to_org_context_limit(self, mock_prefs, mock_svc):
        """With no budget for the tier, the sub-agent uses the org's context limit
        (default 200k) — never the model's full window."""
        mock_prefs.return_value = _prefs(
            subagent_context_budgets={"high": 150_000}, max_context_tokens=90_000,
        )
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", model_tier="medium",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        request = mock_svc.return_value.run_via_stream.call_args[0][1]
        self.assertEqual(request.params.get("max_context_tokens"), 90_000)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_tier_budget_floored_at_minimum(self, mock_prefs, mock_svc):
        """A tiny budget is floored at MIN_CONTEXT_TOKENS to avoid prune thrash."""
        from core.preferences import MIN_CONTEXT_TOKENS

        mock_prefs.return_value = _prefs(subagent_context_budgets={"high": 10_000})
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", model_tier="high",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        request = mock_svc.return_value.run_via_stream.call_args[0][1]
        self.assertEqual(request.params.get("max_context_tokens"), MIN_CONTEXT_TOKENS)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_org_tier_model_override_sets_model_used(self, mock_prefs, mock_svc):
        """An org sub-agent-tier model override (feature_models) wins over the
        tier's slot model when resolving model_used."""
        mock_prefs.return_value = _prefs(
            feature_models={"subagent_medium": "openai/gpt-5-nano"},
        )
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 100
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", model_tier="medium",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.model_used, "openai/gpt-5-nano")

    def _run_params(self, mock_prefs, mock_svc, **pref_overrides):
        mock_prefs.return_value = _prefs(
            feature_models={"subagent_medium": "openai/gpt-6-luna"}, **pref_overrides,
        )
        mock_response = MagicMock()
        mock_response.message.content = "Done"
        mock_response.usage.total_tokens = 1
        mock_response.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = mock_response
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", model_tier="medium",
        )
        from chat.subagent_service import run_subagent
        run_subagent(run.id)
        return mock_svc.return_value.run_via_stream.call_args[0][1].params

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_org_tier_reasoning_level_sets_thinking_level(self, mock_prefs, mock_svc):
        params = self._run_params(mock_prefs, mock_svc, subagent_reasoning={
            "medium": {"model": "openai/gpt-6-luna", "level": "medium"},
        })
        self.assertEqual(params.get("thinking_level"), "medium")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_reasoning_level_for_another_model_is_ignored(self, mock_prefs, mock_svc):
        """A level chosen for a model the tier no longer runs on falls back to
        the registry default (no thinking_level key)."""
        params = self._run_params(mock_prefs, mock_svc, subagent_reasoning={
            "medium": {"model": "anthropic/claude-sonnet-5-5", "level": "medium"},
        })
        self.assertNotIn("thinking_level", params)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_reasoning_level_invalid_for_model_is_ignored(self, mock_prefs, mock_svc):
        params = self._run_params(mock_prefs, mock_svc, subagent_reasoning={
            "medium": {"model": "openai/gpt-6-luna", "level": "bogus"},
        })
        self.assertNotIn("thinking_level", params)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_no_reasoning_entry_leaves_registry_default(self, mock_prefs, mock_svc):
        params = self._run_params(mock_prefs, mock_svc, subagent_reasoning={
            "high": {"model": "openai/gpt-6-luna", "level": "low"},
        })
        self.assertNotIn("thinking_level", params)

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
        self.assertTrue(msg.content.startswith(f"[Sub-agent result: {short_id} — finished"))
        self.assertIn("Research findings here.", msg.content)
        self.assertTrue(msg.is_hidden_from_user)
        self.assertEqual(msg.metadata["source"], "subagent")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_failure_message_created_for_empty_result(self, mock_prefs, mock_svc):
        """A no-content run gets a hidden failure message (not a result one) so
        the orchestrator still reacts instead of the thread going silent."""
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

        msgs = list(ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        ))
        self.assertEqual(len(msgs), 1)
        short_id = str(run.id)[:8]
        self.assertTrue(msgs[0].content.startswith(f"[Sub-agent failed: {short_id}"))
        self.assertIn("no text output", msgs[0].content)
        self.assertTrue(msgs[0].is_hidden_from_user)

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
    def test_empty_result_from_cancel_logs_info_not_warning(self, mock_prefs, mock_svc):
        """An empty finish caused by a user cancel logs INFO, not the Sentry-bound
        'no content' WARNING (WILFRED-8E)."""
        mock_prefs.return_value = _prefs()
        mock_response = MagicMock()
        mock_response.message.content = ""
        mock_response.message.tool_calls = []
        mock_response.usage.total_tokens = 0
        mock_response.usage.cost_usd = 0.0

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="research task",
        )

        # Simulate a concurrent stop: the run is marked FAILED mid-flight (as
        # _cancel_active_subagents does) and the stream returns empty.
        def _cancel_then_empty(*args, **kwargs):
            SubAgentRun.objects.filter(pk=run.id).update(status=SubAgentRun.Status.FAILED)
            return mock_response
        mock_svc.return_value.run_via_stream.side_effect = _cancel_then_empty

        from chat.subagent_service import run_subagent
        with self.assertLogs("chat.subagent_service", level="INFO") as cm:
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertTrue(any("cancelled before producing output" in m for m in cm.output))
        self.assertFalse(any("no content" in m for m in cm.output))

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

    def _outcome_messages(self, run):
        return ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        )

    @patch("chat.tasks._notify_consumer")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_terminal_exception_without_canvas_writes_failure_message(
        self, mock_prefs, mock_svc, mock_notify,
    ):
        """A canvas-less terminal exception used to leave no hidden message at
        all (on_failure is a no-op once the row is FAILED), so the orchestrator
        never heard that the run had died."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = RuntimeError("Provider exploded")
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")

        from chat.subagent_service import run_subagent
        with self.assertRaises(RuntimeError):
            run_subagent(run.id)

        msgs = list(self._outcome_messages(run))
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].content.startswith(f"[Sub-agent failed: {str(run.id)[:8]}"))
        self.assertIn("Provider exploded", msgs[0].content)
        self.assertIn("Model: ", msgs[0].content)
        self.assertNotIn("provider capacity problem", msgs[0].content)
        self.assertTrue(msgs[0].is_hidden_from_user)
        mock_notify.assert_called_once_with(str(run.id), str(self.thread.id))

    @patch("chat.tasks._notify_consumer")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_terminal_exception_with_canvas_notifies_after_failed(
        self, mock_prefs, mock_svc, mock_notify,
    ):
        """The canvas is surfaced as the outcome, and the row is FAILED *before*
        the consumer is notified: its claim only looks at terminal rows, so the
        old order (message + notify, then FAILED) left this outcome to the
        watchdog's next tick."""
        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = RuntimeError("Crash after canvas")
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", canvas="Partial work",
        )
        status_at_notify = []
        mock_notify.side_effect = lambda run_id, thread_id: status_at_notify.append(
            SubAgentRun.objects.get(pk=run_id).status
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(RuntimeError):
            run_subagent(run.id)

        msgs = list(self._outcome_messages(run))
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].content.startswith(f"[Sub-agent result: {str(run.id)[:8]} — finished"))
        self.assertIn('Report: canvas "Working document"', msgs[0].content)
        self.assertIn("chat_subagent_report_read", msgs[0].content)
        self.assertEqual(status_at_notify, [SubAgentRun.Status.FAILED])

    @patch("chat.tasks._notify_consumer")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_terminal_exception_after_user_cancel_writes_nothing(
        self, mock_prefs, mock_svc, mock_notify,
    ):
        """A run the user cancelled mid-flight is already FAILED: the exception
        path must neither surface a message nor notify, and must keep the
        cancellation reason."""
        mock_prefs.return_value = _prefs()
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", canvas="Partial work",
        )

        def cancel_then_raise(*args, **kwargs):
            SubAgentRun.objects.filter(pk=run.pk).update(
                status=SubAgentRun.Status.FAILED, error="Cancelled by user.",
            )
            raise RuntimeError("Crash")

        mock_svc.return_value.run_via_stream.side_effect = cancel_then_raise

        from chat.subagent_service import run_subagent
        with self.assertRaises(RuntimeError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.error, "Cancelled by user.")
        self.assertEqual(self._outcome_messages(run).count(), 0)
        mock_notify.assert_not_called()

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_transient_error_leaves_run_running(self, mock_prefs, mock_svc):
        """A transient LLM error must NOT be recorded as FAILED — the run is left
        RUNNING so a Celery retry re-enters it. (Writing FAILED would make the
        retry a no-op via the RUNNING-transition guard.)"""
        from llm.service.errors import LLMTimeoutError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMTimeoutError(
            "timed out", error_code="timeout",
        )

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(LLMTimeoutError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.RUNNING)
        self.assertEqual(run.error, "")
        self.assertIsNone(run.completed_at)

    @patch("chat.tasks._notify_consumer")
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_rate_limit_fails_the_run_once_with_model_and_capacity_hint(
        self, mock_prefs, mock_svc, mock_notify,
    ):
        """A rate limit that outlasted the provider's own backoff ends the run
        (no whole-run Celery restart), and the orchestrator's notice names the
        model and says a same-model retry will likely fail too."""
        from llm.service.errors import LLMRateLimitError

        mock_prefs.return_value = _prefs()
        mock_svc.return_value.run_via_stream.side_effect = LLMRateLimitError(
            "Gemini is rate limiting requests right now.", error_code="rate_limited",
        )
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            model_used="gemini/gemini-3.8-flash", model_tier="low",
        )

        from chat.subagent_service import run_subagent
        with self.assertRaises(LLMRateLimitError):
            run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.FAILED)
        self.assertIn("rate limiting", run.error)
        msgs = list(self._outcome_messages(run))
        self.assertEqual(len(msgs), 1)
        content = msgs[0].content
        self.assertTrue(content.startswith(f"[Sub-agent failed: {str(run.id)[:8]}"))
        self.assertIn("Model: ", content)
        self.assertIn("provider capacity problem", content)
        self.assertIn("different model_tier", content)
        mock_notify.assert_called_once_with(str(run.id), str(self.thread.id))

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
        # The run's round budget (default "standard") caps the tool loop.
        self.assertEqual(request.params["max_tool_iterations"], 20)
        self.assertIn("You have a budget of 20 rounds.", request.messages[0].content)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_round_budget_sets_iterations_and_records_outcome(self, mock_prefs, mock_svc):
        """A deep run gets 40 rounds; the loop's round count and stop reason are
        stored on the run (budget → budget_exhausted)."""
        mock_prefs.return_value = _prefs()

        def _run_via_stream(pipeline_id, request):
            request.context.observability["rounds"] = 41
            request.context.observability["loop_stop"] = "budget"
            response = MagicMock()
            response.message.content = "Summary. Not covered: X."
            response.usage.total_tokens = 10
            response.usage.cost_usd = 0.0
            return response

        mock_svc.return_value.run_via_stream.side_effect = _run_via_stream
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", round_budget="deep",
        )

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        _, request = mock_svc.return_value.run_via_stream.call_args[0]
        self.assertEqual(request.params["max_tool_iterations"], 40)
        self.assertIsNone(request.context.deadline_seconds)
        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(run.rounds_used, 41)
        self.assertTrue(run.budget_exhausted)
        self.assertEqual(run.stop_reason, "budget")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_truncation_stop_is_recorded_without_exhausting_the_budget(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()

        def _run_via_stream(pipeline_id, request):
            request.context.observability["rounds"] = 3
            request.context.observability["loop_stop"] = "truncation"
            response = MagicMock()
            response.message.content = "Summary. Not covered: X."
            response.usage.total_tokens = 10
            response.usage.cost_usd = 0.0
            return response

        mock_svc.return_value.run_via_stream.side_effect = _run_via_stream
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")

        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        self.assertEqual(run.stop_reason, "truncation")
        self.assertFalse(run.budget_exhausted)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_untouched_template_is_decided_once_per_run(self, mock_prefs, mock_svc):
        """A canvas-only finish needs the "is it still the template?" answer for
        `delivered` and again for the result message: two queries, asked once."""
        from chat import subagent_service
        from chat.models import ChatMessage

        mock_prefs.return_value = _prefs()
        run = SubAgentRun.objects.create(thread=self.thread, user=self.user, prompt="task")

        def _run_via_stream(pipeline_id, request):
            # Tools write the canvas straight to the row during the run.
            SubAgentRun.objects.filter(pk=run.pk).update(
                canvas="# Report\nFindings.", canvas_title="Report",
            )
            response = MagicMock()
            response.message.content = ""
            response.usage.total_tokens = 10
            response.usage.cost_usd = 0.0
            return response

        mock_svc.return_value.run_via_stream.side_effect = _run_via_stream
        with patch.object(
            subagent_service, "_is_untouched_template",
            wraps=subagent_service._is_untouched_template,
        ) as decide:
            subagent_service.run_subagent(run.id)

        self.assertEqual(decide.call_count, 1)
        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        message = ChatMessage.objects.get(
            thread=self.thread, metadata__subagent_run_id=str(run.id),
        )
        self.assertIn('Report: canvas "Report"', message.content)

    @patch("chat.tasks.notify_subagent_status")
    @patch("chat.tasks.run_subagent_task")
    def test_create_stores_round_budget(self, mock_task, _mock_notify):
        mock_task.delay.return_value = MagicMock(id="t1")
        ctx = _ctx(self.user.pk, self.thread.id)
        _invoke(CreateSubagentTool, {"prompt": "task", "round_budget": "quick"}, ctx)
        self.assertEqual(SubAgentRun.objects.get().round_budget, "quick")


class UntouchedTemplateTests(TestCase):
    """A specialization's auto-loaded template the sub-agent never edited is not
    a deliverable: it is neither returned nor counted as output."""

    def setUp(self):
        self.user = User.objects.create_user(email="tmpl@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _skill(self, content):
        skill = MagicMock()
        skill.templates.all.return_value.first.return_value = MagicMock(content=content)
        return skill

    def test_untouched_template_is_not_the_report(self):
        from chat.subagent_service import build_result_notice, report_source

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", skill_slug="web-researcher",
            canvas="# Template", canvas_title="Report", result="Short answer.",
        )
        with patch("chat.subagent_service.get_run_specialization_skill",
                   return_value=self._skill("# Template")):
            self.assertEqual(report_source(run), ("Short answer.", "final_message"))
            notice = build_result_notice(run)
        self.assertNotIn("Report: canvas", notice)
        self.assertIn("Short answer.", notice)

    def test_edited_template_is_the_report(self):
        from chat.subagent_service import build_result_notice, report_source

        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", skill_slug="web-researcher",
            canvas="# Template\nFindings", canvas_title="Report",
        )
        with patch("chat.subagent_service.get_run_specialization_skill",
                   return_value=self._skill("# Template")):
            self.assertEqual(report_source(run), ("# Template\nFindings", "canvas"))
            notice = build_result_notice(run)
        self.assertIn('Report: canvas "Report" (~', notice)
        self.assertIn("2 lines", notice)


class ResultNoticeTests(TestCase):
    """The result message is a pointer to the report plus the capped final
    message — never the report itself."""

    def setUp(self):
        self.user = User.objects.create_user(email="notice@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _run(self, **kw):
        return SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t",
            status=SubAgentRun.Status.COMPLETED, **kw,
        )

    def test_canvas_report_is_a_pointer(self):
        from chat.subagent_service import build_result_notice

        run = self._run(canvas="# Findings\n" + "detail line\n" * 50, canvas_title="Prior art",
                        result="Done. Not covered: X.")
        notice = build_result_notice(run)
        short = str(run.id)[:8]
        self.assertTrue(notice.startswith(f"[Sub-agent result: {short}"))
        self.assertIn('Report: canvas "Prior art" (~', notice)
        self.assertIn("52 lines", notice)
        self.assertIn(f'chat_subagent_report_read(run_id="{short}")', notice)
        self.assertIn("Accompanying message:\nDone. Not covered: X.", notice)
        self.assertNotIn("detail line", notice)

    @override_settings(SUBAGENT_RESULT_INLINE_MAX_TOKENS=50)
    def test_accompanying_message_is_capped(self):
        from chat.subagent_service import build_result_notice

        run = self._run(canvas="report", canvas_title="R", result="word " * 400)
        notice = build_result_notice(run)
        self.assertIn("Accompanying message cut at ~50 tokens", notice)
        self.assertLess(notice.count("word"), 100)

    @override_settings(SUBAGENT_RESULT_INLINE_MAX_TOKENS=50)
    def test_no_canvas_final_message_inline_and_capped(self):
        from chat.subagent_service import build_result_notice

        run = self._run(result="word " * 400)
        notice = build_result_notice(run)
        self.assertNotIn("Report: canvas", notice)
        self.assertIn("Final message cut at ~50 tokens", notice)
        self.assertIn("chat_subagent_report_read", notice)

    def test_short_final_message_without_canvas_is_inline(self):
        from chat.subagent_service import build_result_notice

        notice = build_result_notice(self._run(result="The answer is 42."))
        self.assertIn("The answer is 42.", notice)
        self.assertNotIn("cut at", notice)
        self.assertNotIn("chat_subagent_report_read", notice)


class ReadSubagentReportToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="report@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)

    def _read(self, args):
        from chat.subagent_tool import ReadSubagentReportTool

        return _invoke(ReadSubagentReportTool, args, self.ctx)

    def test_main_audience_only(self):
        from chat.subagent_tool import ReadSubagentReportTool

        self.assertEqual(ReadSubagentReportTool().audience, "main")

    def test_reads_canvas_by_short_id(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.COMPLETED,
            canvas="line one\nline two\nline three", canvas_title="Report", result="fin",
        )
        out = self._read({"run_id": str(run.id)[:8]})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["source"], "canvas")
        self.assertEqual(out["title"], "Report")
        self.assertEqual(out["content"], "line one\nline two\nline three")
        out = self._read({"run_id": str(run.id), "start_line": 2, "end_line": 2})
        self.assertEqual(out["content"], "line two")
        self.assertEqual(out["total_lines"], 3)

    def test_falls_back_to_final_message(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.COMPLETED,
            result="Only a final message.",
        )
        out = self._read({"run_id": str(run.id)[:8]})
        self.assertEqual(out["source"], "final_message")
        self.assertEqual(out["content"], "Only a final message.")

    def test_failed_run_with_canvas_is_readable(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.FAILED,
            canvas="partial", canvas_title="Draft",
        )
        self.assertEqual(self._read({"run_id": str(run.id)[:8]})["content"], "partial")

    def test_running_run_is_refused(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.RUNNING,
            canvas="draft",
        )
        out = self._read({"run_id": str(run.id)[:8]})
        self.assertEqual(out["status"], "error")
        self.assertIn("still running", out["message"])

    def test_other_threads_run_is_not_found(self):
        other = ChatThread.objects.create(created_by=self.user)
        run = SubAgentRun.objects.create(
            thread=other, user=self.user, prompt="t", status=SubAgentRun.Status.COMPLETED,
            canvas="secret",
        )
        out = self._read({"run_id": str(run.id)})
        self.assertEqual(out["status"], "error")
        self.assertIn("No sub-agent run", out["message"])

    def test_empty_run_is_an_error(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="t", status=SubAgentRun.Status.COMPLETED,
        )
        out = self._read({"run_id": str(run.id)[:8]})
        self.assertEqual(out["status"], "error")
        self.assertIn("no report", out["message"])


# ---------------------------------------------------------------------------
# is_retryable_subagent_error predicate
# ---------------------------------------------------------------------------

class IsRetryableSubagentErrorTests(TestCase):
    def test_transient_errors_are_retryable(self):
        from chat.subagent_service import is_retryable_subagent_error
        from llm.service.errors import LLMConnectionError, LLMTimeoutError
        for exc in (
            LLMTimeoutError("x"), LLMConnectionError("x"),
            ConnectionError("x"), TimeoutError("x"), OSError("x"),
        ):
            self.assertTrue(
                is_retryable_subagent_error(exc), type(exc).__name__,
            )

    def test_terminal_errors_are_not_retryable(self):
        """Rate limits / overloads are terminal for the run: the provider layer
        already backed off 4 times, and a whole-run restart replays paid rounds."""
        from chat.subagent_service import is_retryable_subagent_error
        from llm.service.errors import (
            LLMAuthError, LLMOverloadedError, LLMProviderError, LLMRateLimitError,
            LLMRequestTooLargeError,
        )
        for exc in (
            RuntimeError("x"), ValueError("x"),
            LLMAuthError("x"), LLMRequestTooLargeError("x"), LLMProviderError("x"),
            LLMRateLimitError("x"), LLMOverloadedError("x"),
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

    def test_run_task_tags_sentry_with_run_thread_and_type(self):
        """Every event a sub-agent task emits (deadline warnings, EPO faults, provider
        errors) must carry the run, its thread and its specialization, so one Sentry
        query covers a thread's whole run. Regression: the 2026-10-13 prior-art review
        found 18 events from one thread, every one with an empty run tag."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="tagged task",
            skill_slug="patent-searcher", status=SubAgentRun.Status.PENDING,
        )

        from chat.tasks import run_subagent_task
        with patch("sentry_sdk.set_tag") as set_tag,                 patch("chat.subagent_service.run_subagent") as run_sub:
            run_subagent_task(str(run.id))

        run_sub.assert_called_once()
        tags = {c.args[0]: c.args[1] for c in set_tag.call_args_list}
        self.assertEqual(tags["subagent_run_id"], str(run.id))
        self.assertEqual(tags["thread_id"], str(self.thread.id))
        self.assertEqual(tags["subagent_type"], "patent-searcher")
        self.assertEqual(tags["agent_kind"], "subagent")

    def test_run_task_tags_survive_missing_sentry(self):
        """A Sentry SDK hiccup while tagging must not stop the run."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="tagged task",
            status=SubAgentRun.Status.PENDING,
        )

        from chat.tasks import run_subagent_task
        with patch("sentry_sdk.set_tag", side_effect=RuntimeError("sentry down")),                 patch("chat.subagent_service.run_subagent") as run_sub:
            run_subagent_task(str(run.id))

        run_sub.assert_called_once()

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

    async def test_delivered_run_not_reclaimed(self):
        """Once the outcome reached the model (delivered_at is stamped when the
        reply that replayed it is persisted) the run is permanently reported —
        even after the lease window."""
        run = await self._make_completed_run()
        self.assertTrue(await self._claim())

        await self.consumer._mark_subagents_delivered(str(self.thread.id), [str(run.id)])
        await self._backdate_reported_at(run, 6)
        self.assertFalse(await self._claim())

    async def test_undelivered_run_claimable_despite_newer_assistant_message(self):
        """The regression: run A's seeded turn replied AFTER run B's outcome
        landed, but that reply came from a prompt loaded before B existed.
        Reporting is per run, never "a later reply exists" — B stays claimable
        (the old timestamp heuristic counted A's reply as B's report)."""
        run_a = await self._make_completed_run()
        self.assertTrue(await self._claim())
        run_b = await self._make_completed_run()  # lands mid-turn
        await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="assistant",
            content="Here is what sub-agent A found...",
            metadata={"subagent_delivered_run_ids": [str(run_a.id)]},
        )
        await self.consumer._mark_subagents_delivered(str(self.thread.id), [str(run_a.id)])

        ids = await self.consumer._get_unreported_subagent_run_ids(str(self.thread.id))
        self.assertEqual(ids, [run_b.id])
        self.assertTrue(await self._claim())

    async def test_hidden_message_for_foreign_run_ignored(self):
        """A branched thread copies hidden outcome messages whose run ids belong
        to the source thread (and tests use ids that are not even UUIDs):
        never claimable, never an error."""
        await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="user",
            content="[Sub-agent result: deadbeef]\nCopied.",
            metadata={"source": "subagent", "subagent_run_id": "not-a-uuid"},
            is_hidden_from_user=True,
        )
        other_thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user,
        )
        foreign = await database_sync_to_async(SubAgentRun.objects.create)(
            thread=other_thread, user=self.user, prompt="task",
            status=SubAgentRun.Status.COMPLETED, result="Theirs.",
        )
        await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="user",
            content=f"[Sub-agent result: {str(foreign.id)[:8]}]\nTheirs.",
            metadata={"source": "subagent", "subagent_run_id": str(foreign.id)},
            is_hidden_from_user=True,
        )

        ids = await self.consumer._get_unreported_subagent_run_ids(str(self.thread.id))
        self.assertEqual(ids, [])
        self.assertFalse(await self._claim())

    async def test_no_completed_runs_returns_false(self):
        self.assertFalse(await self._claim())

    @database_sync_to_async
    def _make_failed_run(self, *, with_message=True, error="Boom"):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.FAILED,
            error=error,
        )
        if with_message:
            from chat.subagent_service import _create_subagent_failure_message
            _create_subagent_failure_message(run)
        return run

    async def test_failed_run_with_failure_message_is_claimable(self):
        """A terminal failure is surfaced to the orchestrator like a result."""
        await self._make_failed_run()
        self.assertTrue(await self._claim())

    async def test_cancelled_run_without_message_not_claimed(self):
        """User-cancelled runs never get a hidden message, so they must never
        trigger a seeded continuation."""
        await self._make_failed_run(with_message=False, error="Cancelled by user.")
        self.assertFalse(await self._claim())

    async def test_unreported_ids_helper_matches_claim(self):
        run = await self._make_completed_run()
        ids = await self.consumer._get_unreported_subagent_run_ids(str(self.thread.id))
        self.assertEqual(ids, [run.id])
        # Read-only: the helper must not take the claim.
        self.assertTrue(await self._claim())


class MarkSubagentsDeliveredTests(TransactionTestCase):
    """``_mark_subagents_delivered``: the per-run report marker."""

    def setUp(self):
        from chat.consumers import ChatConsumer

        self.user = User.objects.create_user(email="deliver@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user

    @database_sync_to_async
    def _make_run(self, thread=None):
        return SubAgentRun.objects.create(
            thread=thread or self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.COMPLETED, result="Findings.",
        )

    @database_sync_to_async
    def _delivered_at(self, run):
        return SubAgentRun.objects.get(pk=run.pk).delivered_at

    async def _mark(self, run_ids):
        return await self.consumer._mark_subagents_delivered(str(self.thread.id), run_ids)

    async def test_marks_null_rows_once(self):
        run = await self._make_run()
        self.assertEqual(await self._mark([str(run.id)]), 1)
        first = await self._delivered_at(run)
        self.assertIsNotNone(first)
        # Idempotent: a duplicate report (two tabs) never moves the stamp.
        self.assertEqual(await self._mark([str(run.id)]), 0)
        self.assertEqual(await self._delivered_at(run), first)

    async def test_thread_scoped_and_empty_safe(self):
        other = await database_sync_to_async(ChatThread.objects.create)(created_by=self.user)
        run = await self._make_run(thread=other)
        self.assertEqual(await self._mark([str(run.id)]), 0)
        self.assertIsNone(await self._delivered_at(run))
        self.assertEqual(await self._mark([]), 0)


class BackfillDeliveredAtTests(TestCase):
    """Data migration 0052 reproduces what the old heuristic considered settled,
    so deploying does not seed a continuation on every historical thread."""

    def setUp(self):
        import importlib

        self.backfill = importlib.import_module(
            "chat.migrations.0052_subagentrun_delivered_at"
        ).backfill_delivered_at
        self.user = User.objects.create_user(email="backfill@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.base = timezone.now() - timedelta(hours=1)

    def _run(self, **overrides):
        fields = dict(
            thread=self.thread, user=self.user, prompt="task",
            status=SubAgentRun.Status.COMPLETED, result="Findings.",
        )
        fields.update(overrides)
        return SubAgentRun.objects.create(**fields)

    def _at(self, msg, minutes):
        ChatMessage.objects.filter(pk=msg.pk).update(
            created_at=self.base + timedelta(minutes=minutes),
        )
        return ChatMessage.objects.get(pk=msg.pk)

    def _hidden(self, run, minutes):
        return self._at(ChatMessage.objects.create(
            thread=self.thread, role="user", content="[Sub-agent result]",
            metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            is_hidden_from_user=True,
        ), minutes)

    def _reply(self, minutes, metadata=None):
        return self._at(ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="reply", metadata=metadata or {},
        ), minutes)

    def _apply(self):
        from django.apps import apps

        self.backfill(apps, None)

    def _delivered_at(self, run):
        return SubAgentRun.objects.get(pk=run.pk).delivered_at

    def test_tagged_reply_after_hidden_message_marks_delivered(self):
        run = self._run()
        self._hidden(run, 0)
        reply = self._reply(1, {"subagent_response": True})
        self._apply()
        self.assertEqual(self._delivered_at(run), reply.created_at)

    def test_tagged_reply_before_hidden_message_does_not_count(self):
        run = self._run()
        self._reply(0, {"subagent_response": True})
        self._hidden(run, 1)
        self._apply()
        self.assertIsNone(self._delivered_at(run))

    def test_inline_blocking_delivery_marks_delivered(self):
        """Blocking waits returned the outcome inline and never tagged the reply."""
        run = self._run(timeout=60)
        SubAgentRun.objects.filter(pk=run.pk).update(reported_at=self.base)
        self._hidden(run, 1)
        reply = self._reply(2)
        self._apply()
        self.assertEqual(self._delivered_at(run), reply.created_at)

    def test_failed_with_canvas_blocking_run_stays_undelivered(self):
        """The inline error never carried the canvas, so that outcome is still owed."""
        run = self._run(
            timeout=60, status=SubAgentRun.Status.FAILED, error="boom", canvas="Partial",
        )
        SubAgentRun.objects.filter(pk=run.pk).update(reported_at=self.base)
        self._hidden(run, 1)
        self._reply(2)
        self._apply()
        self.assertIsNone(self._delivered_at(run))

    def test_background_claim_in_flight_stays_undelivered(self):
        """A seeded turn claimed the run but has not replied yet (or died): the
        lease expiry re-seeds it after deploy, as it would have before."""
        run = self._run(timeout=0)
        SubAgentRun.objects.filter(pk=run.pk).update(reported_at=timezone.now())
        self._hidden(run, 0)
        self._reply(1)  # untagged: not a report of this run
        self._apply()
        self.assertIsNone(self._delivered_at(run))

    def test_cancelled_run_without_message_stays_null(self):
        run = self._run(status=SubAgentRun.Status.FAILED, error="Cancelled by user.")
        self._reply(1, {"subagent_response": True})
        self._apply()
        self.assertIsNone(self._delivered_at(run))

    def test_already_delivered_untouched(self):
        run = self._run()
        SubAgentRun.objects.filter(pk=run.pk).update(delivered_at=self.base)
        self._hidden(run, 1)
        self._reply(2, {"subagent_response": True})
        self._apply()
        self.assertEqual(self._delivered_at(run), self.base)


# ---------------------------------------------------------------------------
# Hidden failure messages (fix: terminally-failed runs were invisible — the
# status bar decremented but the orchestrator never reacted)
# ---------------------------------------------------------------------------

class SubagentFailureMessageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="failmsg@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _failure_messages(self, run):
        return ChatMessage.objects.filter(
            thread=self.thread,
            metadata__source="subagent",
            metadata__subagent_run_id=str(run.id),
        )

    def test_on_failure_creates_hidden_failure_message(self):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="doomed", status=SubAgentRun.Status.RUNNING,
        )

        from chat.tasks import run_subagent_task
        run_subagent_task.on_failure(
            RuntimeError("Provider exploded"), "fake-task-id", [str(run.id)], {}, None,
        )

        msgs = list(self._failure_messages(run))
        self.assertEqual(len(msgs), 1)
        short_id = str(run.id)[:8]
        self.assertTrue(msgs[0].content.startswith(f"[Sub-agent failed: {short_id}"))
        self.assertIn("Provider exploded", msgs[0].content)
        self.assertEqual(msgs[0].role, "user")
        self.assertTrue(msgs[0].is_hidden_from_user)

    def test_on_failure_cancelled_run_gets_no_message(self):
        """A user-cancelled run is already FAILED; on_failure must not surface it."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="cancelled", status=SubAgentRun.Status.FAILED,
            error="Cancelled by user.",
        )

        from chat.tasks import run_subagent_task
        run_subagent_task.on_failure(
            RuntimeError("Worker terminated"), "fake-task-id", [str(run.id)], {}, None,
        )

        self.assertEqual(self._failure_messages(run).count(), 0)

    def test_failure_message_deduplicated_against_existing_result(self):
        """The except-path may already have persisted a canvas result message;
        on_failure must not add a second hidden message for the same run."""
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="canvas then crash", status=SubAgentRun.Status.RUNNING,
            canvas="Partial work",
        )
        from chat.subagent_service import _create_subagent_result_message
        _create_subagent_result_message(run)

        from chat.tasks import run_subagent_task
        run_subagent_task.on_failure(
            RuntimeError("Crash after canvas"), "fake-task-id", [str(run.id)], {}, None,
        )

        self.assertEqual(self._failure_messages(run).count(), 1)

    def test_expiry_creates_failure_messages(self):
        """The stale-run sweeper surfaces expired runs to the orchestrator."""
        pending = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="never started", status=SubAgentRun.Status.PENDING,
        )
        running = SubAgentRun.objects.create(
            thread=self.thread, user=self.user,
            prompt="stuck", status=SubAgentRun.Status.RUNNING,
        )
        SubAgentRun.objects.filter(pk=pending.pk).update(
            created_at=timezone.now() - timedelta(minutes=11),
            dispatched_at=timezone.now() - timedelta(minutes=11),
        )
        SubAgentRun.objects.filter(pk=running.pk).update(
            started_at=timezone.now() - timedelta(seconds=running_silence_limit("") + 60),
        )

        expired = _expire_stale_runs()
        self.assertEqual(expired, 2)

        for run in (pending, running):
            msgs = list(self._failure_messages(run))
            self.assertEqual(len(msgs), 1, f"missing failure message for {run.prompt}")
            self.assertIn("Expired:", msgs[0].content)


# ---------------------------------------------------------------------------
# Self-heal watchdog (fix: the completion group_send is at-most-once; a lost
# event left the status bar stuck and the orchestrator never continued)
# ---------------------------------------------------------------------------

class SubagentWatchdogTests(TransactionTestCase):
    def setUp(self):
        from unittest.mock import AsyncMock

        from chat.consumers import ChatConsumer

        self.user = User.objects.create_user(email="watch@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user
        self.consumer.resolved_prefs = _prefs()
        self.consumer._stopped = False
        self.consumer._current_thread_id = str(self.thread.id)
        self.consumer._stream_task = None
        self.consumer._subagent_watch_task = None
        self.consumer.send = AsyncMock()
        self.consumer._handle_chat_message = AsyncMock()

    def _sent_counts(self):
        return [
            json.loads(c.kwargs["text_data"])["active_count"]
            for c in self.consumer.send.call_args_list
            if json.loads(c.kwargs["text_data"]).get("event_type") == "subagents.updated"
        ]

    @database_sync_to_async
    def _make_completed_unreported_run(self):
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

    async def _tick(self):
        return await self.consumer._subagent_watch_tick(str(self.thread.id))

    async def test_tick_resyncs_count_and_seeds_unreported(self):
        """The core self-heal: a completed run whose completion event was lost
        gets its count pushed and a continuation seeded on the next tick."""
        await self._make_completed_unreported_run()

        keep_watching = await self._tick()

        self.assertTrue(keep_watching)
        self.assertEqual(self._sent_counts(), [0])
        self.consumer._handle_chat_message.assert_awaited_once_with(
            {"thread_id": str(self.thread.id), "content": ""}, seed_mode=True,
        )

    async def test_tick_stops_when_idle(self):
        """Nothing running, nothing unreported → the watchdog retires."""
        self.assertFalse(await self._tick())
        self.assertEqual(self._sent_counts(), [0])
        self.consumer._handle_chat_message.assert_not_awaited()

    async def test_tick_keeps_watching_while_runs_active(self):
        await database_sync_to_async(SubAgentRun.objects.create)(
            thread=self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.RUNNING,
        )
        self.assertTrue(await self._tick())
        self.assertEqual(self._sent_counts(), [1])
        self.consumer._handle_chat_message.assert_not_awaited()

    async def test_tick_reports_waiting_count(self):
        """A run still queued for an execution slot counts as active (keeps the
        watchdog alive) and is reported separately as waiting."""
        await database_sync_to_async(SubAgentRun.objects.create)(
            thread=self.thread, user=self.user,
            prompt="task", status=SubAgentRun.Status.PENDING,
        )
        self.assertTrue(await self._tick())
        frames = [
            json.loads(c.kwargs["text_data"])
            for c in self.consumer.send.call_args_list
            if json.loads(c.kwargs["text_data"]).get("event_type") == "subagents.updated"
        ]
        self.assertEqual(frames[0]["active_count"], 1)
        self.assertEqual(frames[0]["waiting_count"], 1)
        # The panel's per-run list rides along.
        self.assertEqual(len(frames[0]["runs"]), 1)
        self.assertEqual(frames[0]["runs"][0]["state"], "queued")
        self.assertIn("server_now", frames[0])

    async def test_status_event_resends_run_list(self):
        """subagent.status (queued/started push) → subagents.updated with runs,
        even while a stream is live (a blocking sub-agent runs mid-turn)."""
        import asyncio

        run = await database_sync_to_async(SubAgentRun.objects.create)(
            thread=self.thread, user=self.user, prompt="task", reason="Why",
            status=SubAgentRun.Status.RUNNING,
        )
        self.consumer._stream_task = asyncio.create_task(asyncio.sleep(10))
        try:
            await self.consumer.subagent_status(
                {"type": "subagent.status", "run_id": str(run.id), "thread_id": str(self.thread.id)}
            )
        finally:
            self.consumer._stream_task.cancel()
        frame = json.loads(self.consumer.send.call_args.kwargs["text_data"])
        self.assertEqual(frame["event_type"], "subagents.updated")
        self.assertEqual(frame["active_count"], 1)
        self.assertEqual(frame["runs"][0]["reason"], "Why")
        self.consumer._handle_chat_message.assert_not_awaited()

    async def test_status_event_for_other_thread_ignored(self):
        await self.consumer.subagent_status(
            {"type": "subagent.status", "run_id": "x", "thread_id": str(uuid.uuid4())}
        )
        self.consumer.send.assert_not_awaited()

    async def test_tick_defers_to_live_stream(self):
        """While a stream runs, the post-stream claim owns delivery — the tick
        must neither push counts nor seed, but must keep watching."""
        import asyncio

        self.consumer._stream_task = asyncio.create_task(asyncio.sleep(10))
        try:
            await self._make_completed_unreported_run()
            self.assertTrue(await self._tick())
            self.assertEqual(self._sent_counts(), [])
            self.consumer._handle_chat_message.assert_not_awaited()
        finally:
            self.consumer._stream_task.cancel()

    async def test_tick_exits_on_thread_switch(self):
        self.consumer._current_thread_id = str(uuid.uuid4())
        self.assertFalse(await self._tick())
        self.consumer.send.assert_not_awaited()

    async def test_tick_exits_when_stopped(self):
        self.consumer._stopped = True
        self.assertFalse(await self._tick())
        self.consumer.send.assert_not_awaited()

    async def test_tick_keeps_watching_when_claim_lease_held(self):
        """Another tab holds a fresh claim: don't seed, but keep watching so a
        died seeded turn is retried once the lease expires."""
        run = await self._make_completed_unreported_run()
        await database_sync_to_async(
            lambda: SubAgentRun.objects.filter(pk=run.pk).update(
                reported_at=timezone.now(),
            )
        )()

        self.assertTrue(await self._tick())
        self.consumer._handle_chat_message.assert_not_awaited()

    async def test_tick_skips_delivered_run(self):
        """A delivered outcome is settled for good: no seed, nothing to watch."""
        run = await self._make_completed_unreported_run()
        await self.consumer._mark_subagents_delivered(str(self.thread.id), [str(run.id)])

        self.assertFalse(await self._tick())
        self.consumer._handle_chat_message.assert_not_awaited()

    async def test_tick_seeds_undelivered_despite_newer_assistant_message(self):
        """A reply persisted after the outcome landed is not its report unless
        it replayed it — the watchdog still seeds (the old heuristic stopped
        here and the outcome was orphaned)."""
        await self._make_completed_unreported_run()
        await database_sync_to_async(ChatMessage.objects.create)(
            thread=self.thread, role="assistant",
            content="Reply generated from an earlier prompt.",
            metadata={"subagent_delivered_run_ids": [str(uuid.uuid4())]},
        )

        self.assertTrue(await self._tick())
        self.consumer._handle_chat_message.assert_awaited_once_with(
            {"thread_id": str(self.thread.id), "content": ""}, seed_mode=True,
        )
