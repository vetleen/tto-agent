"""Tests for the sub-agent working canvas.

Covers the three ``subagent_canvas_*`` tools, the audience wiring that makes them
available to sub-agents (and not the main agent), template auto-load, the
delimited canvas returned to the orchestrator, failure durability, the broadened
claim path, and the prompt changes.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase

from agent_skills.models import AgentSkill, SkillTemplate
from chat.models import ChatMessage, ChatThread, SubAgentRun
from chat.subagent_canvas_tools import (
    SubagentCanvasEditTool,
    SubagentCanvasLoadTemplateTool,
    SubagentCanvasWriteTool,
)
from chat.subagent_prompts import build_subagent_system_prompt
from core.preferences import ResolvedPreferences
from llm.tools import get_tool_registry
from llm.types.context import RunContext

User = get_user_model()

_CANVAS_TOOLS = (
    "subagent_canvas_write",
    "subagent_canvas_edit",
    "subagent_canvas_load_template",
)


def _prefs(**overrides):
    defaults = dict(
        top_model="openai/gpt-5",
        mid_model="openai/gpt-5-mini",
        cheap_model="openai/gpt-5-nano",
        allowed_models=["openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano"],
        allowed_tools=["web_search"],
        allowed_subagent_tools=["web_search"],
        allowed_skills=[],
        allowed_specializations=[],
        theme="light",
    )
    defaults.update(overrides)
    return ResolvedPreferences(**defaults)


def _ctx(user_id, thread_id, run_id=None):
    ctx = RunContext.create(user_id=user_id, conversation_id=str(thread_id))
    if run_id is not None:
        ctx.run_id = str(run_id)
    return ctx


def _invoke(tool_cls, args, ctx):
    tool = tool_cls()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


def _mock_llm(mock_svc, content="Done", tokens=100, cost=0.0):
    resp = MagicMock()
    resp.message.content = content
    resp.usage.total_tokens = tokens
    resp.usage.cost_usd = cost
    mock_svc.return_value.run_via_stream.return_value = resp
    return resp


# ---------------------------------------------------------------------------
# Audience / registration
# ---------------------------------------------------------------------------

class SubagentCanvasAudienceTests(TestCase):
    def test_tools_registered_with_subagent_audience(self):
        reg = get_tool_registry().list_tools()
        for name in _CANVAS_TOOLS:
            self.assertIn(name, reg)
            self.assertEqual(reg[name].audience, "subagent")
            self.assertEqual(getattr(reg[name], "section", "chat"), "chat")

    def test_available_to_subagents_not_main(self):
        """Replicates the audience filter in core.preferences: the canvas tools
        land in the sub-agent set and never in the main-agent set."""
        tools = get_tool_registry().list_tools()
        subagent_set = {
            n for n, t in tools.items()
            if getattr(t, "section", "chat") == "chat"
            and getattr(t, "audience", "shared") in ("subagent", "shared")
        }
        main_set = {
            n for n, t in tools.items()
            if getattr(t, "section", "chat") == "chat"
            and getattr(t, "audience", "shared") in ("main", "shared")
        }
        for name in _CANVAS_TOOLS:
            self.assertIn(name, subagent_set)
            self.assertNotIn(name, main_set)


# ---------------------------------------------------------------------------
# write / edit tools
# ---------------------------------------------------------------------------

class SubagentCanvasWriteEditTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cw@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )
        self.ctx = _ctx(self.user.id, self.thread.id, run_id=self.run.id)

    def test_write_sets_canvas(self):
        out = _invoke(
            SubagentCanvasWriteTool,
            {"content": "Hello world", "title": "Doc"},
            self.ctx,
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["chars"], len("Hello world"))
        self.run.refresh_from_db()
        self.assertEqual(self.run.canvas, "Hello world")
        self.assertEqual(self.run.canvas_title, "Doc")

    def test_write_truncates_at_cap(self):
        from chat.services import CANVAS_MAX_CHARS

        _invoke(
            SubagentCanvasWriteTool,
            {"content": "x" * (CANVAS_MAX_CHARS + 50)},
            self.ctx,
        )
        self.run.refresh_from_db()
        self.assertEqual(len(self.run.canvas), CANVAS_MAX_CHARS)

    def test_edit_applies_and_echoes_content(self):
        self.run.canvas = "The quick brown fox"
        self.run.save(update_fields=["canvas"])
        out = _invoke(
            SubagentCanvasEditTool,
            {"edits": [{"old_text": "quick brown", "new_text": "slow red"}]},
            self.ctx,
        )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["applied"], 1)
        self.assertIn("slow red", out["content"])
        self.run.refresh_from_db()
        self.assertEqual(self.run.canvas, "The slow red fox")

    def test_edit_no_match_errors_and_leaves_canvas_untouched(self):
        self.run.canvas = "original text"
        self.run.save(update_fields=["canvas"])
        out = _invoke(
            SubagentCanvasEditTool,
            {"edits": [{"old_text": "absent", "new_text": "x"}]},
            self.ctx,
        )
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["applied"], 0)
        self.run.refresh_from_db()
        self.assertEqual(self.run.canvas, "original text")

    def test_unknown_run_is_inert(self):
        ctx = _ctx(self.user.id, self.thread.id, run_id=uuid.uuid4())
        out = _invoke(SubagentCanvasWriteTool, {"content": "x"}, ctx)
        self.assertEqual(out["status"], "error")


# ---------------------------------------------------------------------------
# load_template tool
# ---------------------------------------------------------------------------

class SubagentCanvasLoadTemplateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="lt@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", skill_slug="spec",
        )
        self.ctx = _ctx(self.user.id, self.thread.id, run_id=self.run.id)
        self.skill = AgentSkill.objects.create(
            slug="spec", name="Spec", instructions="Do it.",
            audience=AgentSkill.Audience.SUBAGENT, level=AgentSkill.Level.SYSTEM,
        )
        SkillTemplate.objects.create(skill=self.skill, name="Alpha", content="ALPHA")
        SkillTemplate.objects.create(skill=self.skill, name="Beta", content="BETA")

    def test_load_named_template_replaces_canvas(self):
        self.run.canvas = "old stuff"
        self.run.save(update_fields=["canvas"])
        with patch(
            "chat.subagent_service.get_run_specialization_skill", return_value=self.skill
        ):
            out = _invoke(
                SubagentCanvasLoadTemplateTool, {"template_name": "Beta"}, self.ctx
            )
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["content"], "BETA")
        self.run.refresh_from_db()
        self.assertEqual(self.run.canvas, "BETA")
        self.assertEqual(self.run.canvas_title, "Beta")

    def test_unknown_template_errors_with_available(self):
        with patch(
            "chat.subagent_service.get_run_specialization_skill", return_value=self.skill
        ):
            out = _invoke(
                SubagentCanvasLoadTemplateTool, {"template_name": "Gamma"}, self.ctx
            )
        self.assertEqual(out["status"], "error")
        self.assertIn("Alpha", out["available_templates"])
        self.assertIn("Beta", out["available_templates"])

    def test_no_specialization_errors(self):
        with patch(
            "chat.subagent_service.get_run_specialization_skill", return_value=None
        ):
            out = _invoke(
                SubagentCanvasLoadTemplateTool, {"template_name": "Alpha"}, self.ctx
            )
        self.assertEqual(out["status"], "error")


# ---------------------------------------------------------------------------
# run_subagent: auto-load, return injection, failure durability
# ---------------------------------------------------------------------------

class SubagentCanvasRunTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="run@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)

    def _subagent_message(self):
        return ChatMessage.objects.filter(
            thread=self.thread, metadata__source="subagent",
        ).first()

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_auto_loads_first_template_by_name(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        _mock_llm(mock_svc, content="done")
        skill = AgentSkill.objects.create(
            slug="spec", name="Spec", instructions="Do it.",
            audience=AgentSkill.Audience.SUBAGENT, level=AgentSkill.Level.SYSTEM,
        )
        SkillTemplate.objects.create(skill=skill, name="Beta", content="BETA BODY")
        SkillTemplate.objects.create(skill=skill, name="Alpha", content="ALPHA BODY")
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task", skill_slug="spec",
        )

        with patch(
            "chat.subagent_service.get_run_specialization_skill", return_value=skill
        ):
            from chat.subagent_service import run_subagent
            run_subagent(run.id)

        run.refresh_from_db()
        # "Alpha" sorts before "Beta" → name-first auto-load.
        self.assertEqual(run.canvas, "ALPHA BODY")
        self.assertEqual(run.canvas_title, "Alpha")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_no_specialization_leaves_canvas_empty(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        _mock_llm(mock_svc, content="done")
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
        )
        from chat.subagent_service import run_subagent
        run_subagent(run.id)
        run.refresh_from_db()
        self.assertEqual(run.canvas, "")

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_canvas_returned_to_orchestrator_delimited(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        _mock_llm(mock_svc, content="Final answer")
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            canvas="CANVAS BODY", canvas_title="My Doc",
        )
        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        msg = self._subagent_message()
        self.assertIsNotNone(msg)
        self.assertIn("=== SUBAGENT CANVAS: My Doc ===", msg.content)
        self.assertIn("CANVAS BODY", msg.content)
        self.assertIn("Final answer", msg.content)

    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_canvas_only_no_final_text_is_completed_and_surfaced(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        _mock_llm(mock_svc, content="")  # used tools, emitted no final message
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            canvas="PARTIAL WORK", canvas_title="Draft",
        )
        from chat.subagent_service import run_subagent
        run_subagent(run.id)

        run.refresh_from_db()
        self.assertEqual(run.status, SubAgentRun.Status.COMPLETED)
        msg = self._subagent_message()
        self.assertIsNotNone(msg)
        self.assertIn("PARTIAL WORK", msg.content)
        self.assertIn("no final message", msg.content)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class SubagentCanvasPromptTests(TestCase):
    def test_working_canvas_section_present_when_empty(self):
        prompt = build_subagent_system_prompt()
        self.assertIn("# Working canvas", prompt)
        self.assertIn("currently empty", prompt)

    def test_working_canvas_injects_content(self):
        prompt = build_subagent_system_prompt(
            canvas_content="DRAFT TEXT", canvas_title="Report",
        )
        self.assertIn("# Working canvas", prompt)
        self.assertIn("DRAFT TEXT", prompt)
        self.assertIn('Current canvas content: "Report"', prompt)

    def test_specialization_mentions_loaded_template_not_no_canvas(self):
        skill = MagicMock()
        skill.name = "Spec"
        skill.description = ""
        skill.instructions = "Do it."
        t1, t2 = MagicMock(), MagicMock()
        t1.name, t2.name = "Alpha", "Beta"
        skill.templates.all.return_value = [t1, t2]

        prompt = build_subagent_system_prompt(specialization_skill=skill)
        self.assertIn("loaded into your working canvas", prompt)
        self.assertIn("Alpha", prompt)
        self.assertIn("Beta", prompt)
        self.assertNotIn("you have no canvas", prompt)
        self.assertNotIn("reproduce the relevant structure", prompt)


# ---------------------------------------------------------------------------
# Broadened claim path (canvas counts as a deliverable)
# ---------------------------------------------------------------------------

class SubagentCanvasClaimTests(TransactionTestCase):
    def setUp(self):
        from chat.consumers import ChatConsumer

        self.user = User.objects.create_user(email="ccanvas@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user

    async def _claim(self):
        return await self.consumer._claim_unreported_subagents(str(self.thread.id))

    @database_sync_to_async
    def _make_run(self, *, status, result="", canvas="", with_message=True):
        run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            status=status, result=result, canvas=canvas,
            canvas_title="Doc" if canvas else "",
        )
        if with_message:
            ChatMessage.objects.create(
                thread=self.thread, role="user",
                content=f"[Sub-agent result: {str(run.id)[:8]}]\n{result or 'canvas only'}",
                metadata={"source": "subagent", "subagent_run_id": str(run.id)},
                is_hidden_from_user=True,
            )
        return run

    async def test_canvas_only_completed_is_claimable(self):
        await self._make_run(status=SubAgentRun.Status.COMPLETED, result="", canvas="BODY")
        self.assertTrue(await self._claim())

    async def test_failed_with_canvas_and_message_is_claimable(self):
        await self._make_run(status=SubAgentRun.Status.FAILED, result="", canvas="BODY")
        self.assertTrue(await self._claim())

    async def test_cancelled_failed_with_canvas_but_no_message_not_claimed(self):
        await self._make_run(
            status=SubAgentRun.Status.FAILED, result="", canvas="BODY", with_message=False,
        )
        self.assertFalse(await self._claim())
