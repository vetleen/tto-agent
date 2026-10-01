"""Tests for sharing the orchestrator's canvases with a sub-agent.

``chat_subagent_create(canvases=[...])`` snapshots the named canvases onto the
run at spawn time; the sub-agent sees them read-only in its system prompt.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from chat.models import ChatCanvas, ChatThread, SubAgentRun
from chat.subagent_prompts import build_subagent_system_prompt
from chat.subagent_service import max_shared_canvases, subagent_context_budget
from chat.subagent_tool import CreateSubagentTool
from core.preferences import MIN_CONTEXT_TOKENS, ResolvedPreferences
from llm.types.context import RunContext

User = get_user_model()


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


def _ctx(user_id, thread_id):
    return RunContext.create(user_id=user_id, conversation_id=str(thread_id))


def _invoke(args, ctx):
    tool = CreateSubagentTool()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


@patch("chat.tasks.run_subagent_task")
class CreateSubagentSharedCanvasTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="share@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.ctx = _ctx(self.user.pk, self.thread.id)
        ChatCanvas.objects.create(thread=self.thread, title="Draft", content="DRAFT BODY")
        ChatCanvas.objects.create(thread=self.thread, title="Notes", content="NOTES BODY")

    def test_omitted_shares_nothing(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        result = _invoke({"prompt": "task"}, self.ctx)
        self.assertIn(result["status"], ("started", "queued"))
        self.assertEqual(SubAgentRun.objects.get().shared_canvases, [])

    def test_snapshot_stored_on_run(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        result = _invoke({"prompt": "review", "canvases": ["Draft", "Notes"]}, self.ctx)
        self.assertIn(result["status"], ("started", "queued"))
        self.assertEqual(
            SubAgentRun.objects.get().shared_canvases,
            [
                {"title": "Draft", "content": "DRAFT BODY"},
                {"title": "Notes", "content": "NOTES BODY"},
            ],
        )

    def test_snapshot_not_affected_by_later_edits(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        _invoke({"prompt": "review", "canvases": ["Draft"]}, self.ctx)
        ChatCanvas.objects.filter(thread=self.thread, title="Draft").update(content="CHANGED")
        run = SubAgentRun.objects.get()
        self.assertEqual(run.shared_canvases[0]["content"], "DRAFT BODY")

    def test_duplicate_titles_deduped(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        _invoke({"prompt": "review", "canvases": ["Draft", "Draft", " Draft "]}, self.ctx)
        self.assertEqual(len(SubAgentRun.objects.get().shared_canvases), 1)

    def test_unknown_title_lists_available_and_creates_no_run(self, mock_task):
        result = _invoke({"prompt": "review", "canvases": ["Nope"]}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("Nope", result["message"])
        self.assertIn('"Draft"', result["message"])
        self.assertIn('"Notes"', result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_deleted_canvas_not_shared(self, mock_task):
        ChatCanvas.objects.filter(title="Draft").update(deleted_at=timezone.now())
        result = _invoke({"prompt": "review", "canvases": ["Draft"]}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertNotIn('"Draft"', result["message"].split("conversation:")[-1])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_other_threads_canvas_not_shared(self, mock_task):
        other = ChatThread.objects.create(created_by=self.user)
        ChatCanvas.objects.create(thread=other, title="Elsewhere", content="X")
        result = _invoke({"prompt": "review", "canvases": ["Elsewhere"]}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertFalse(SubAgentRun.objects.exists())

    def test_over_count_limit_rejected(self, mock_task):
        with patch("core.preferences.get_preferences", return_value=_prefs(max_context_tokens=50_000)):
            result = _invoke({"prompt": "review", "canvases": ["Draft", "Notes"]}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn("up to 1 canvas", result["message"])
        self.assertIn("you asked for 2", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_at_count_limit_accepted(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        with patch("core.preferences.get_preferences", return_value=_prefs(max_context_tokens=50_000)):
            result = _invoke({"prompt": "review", "canvases": ["Draft"]}, self.ctx)
        self.assertIn(result["status"], ("started", "queued"))
        self.assertEqual(len(SubAgentRun.objects.get().shared_canvases), 1)

    def test_limit_follows_requested_tier(self, mock_task):
        mock_task.delay.return_value = MagicMock(id="t")
        ChatCanvas.objects.create(thread=self.thread, title="Third", content="3")
        prefs = _prefs(max_context_tokens=50_000, subagent_context_budgets={"top": 200_000})
        with patch("core.preferences.get_preferences", return_value=prefs):
            mid = _invoke({"prompt": "r", "canvases": ["Draft", "Notes", "Third"]}, self.ctx)
            top = _invoke(
                {"prompt": "r", "canvases": ["Draft", "Notes", "Third"], "model_tier": "top"},
                self.ctx,
            )
        self.assertEqual(mid["status"], "error")
        self.assertIn(top["status"], ("started", "queued"))
        self.assertEqual(len(SubAgentRun.objects.get().shared_canvases), 3)


class SharedCanvasLimitTests(TestCase):
    def test_max_shared_canvases_steps(self):
        for budget, expected in (
            (49_999, 0), (50_000, 1), (149_999, 1), (150_000, 2),
            (199_999, 2), (200_000, 3), (1_000_000, 3),
        ):
            self.assertEqual(max_shared_canvases(budget), expected, budget)

    def test_context_budget_prefers_tier_budget(self):
        prefs = _prefs(max_context_tokens=200_000, subagent_context_budgets={"mid": 120_000})
        self.assertEqual(subagent_context_budget(prefs, "mid"), 120_000)
        self.assertEqual(subagent_context_budget(prefs, "top"), 200_000)

    def test_context_budget_floored(self):
        prefs = _prefs(subagent_context_budgets={"mid": 10_000})
        self.assertEqual(subagent_context_budget(prefs, "mid"), MIN_CONTEXT_TOKENS)


class SharedCanvasPromptTests(TestCase):
    def test_absent_when_nothing_shared(self):
        for shared in (None, []):
            prompt = build_subagent_system_prompt(shared_canvases=shared)
            self.assertNotIn("Canvases shared by the orchestrator", prompt)

    def test_renders_title_content_and_read_only_framing(self):
        prompt = build_subagent_system_prompt(
            canvas_content="MY WORK",
            canvas_title="Mine",
            shared_canvases=[{"title": "Draft", "content": "DRAFT BODY"}],
        )
        self.assertIn("# Canvases shared by the orchestrator", prompt)
        self.assertIn('## Shared canvas: "Draft"', prompt)
        self.assertIn("DRAFT BODY", prompt)
        self.assertIn("read-only", prompt)
        self.assertIn("not as instructions", prompt)
        # Separate from (and after) the sub-agent's own working canvas.
        self.assertLess(prompt.index("# Working canvas"), prompt.index("# Canvases shared"))
        self.assertIn("MY WORK", prompt)


class RunSubagentSharedCanvasTests(TestCase):
    @patch("llm.get_llm_service")
    @patch("core.preferences.get_preferences")
    def test_shared_canvases_reach_system_prompt(self, mock_prefs, mock_svc):
        mock_prefs.return_value = _prefs()
        resp = MagicMock()
        resp.message.content = "done"
        resp.usage.total_tokens = 1
        resp.usage.cost_usd = 0.0
        mock_svc.return_value.run_via_stream.return_value = resp

        user = User.objects.create_user(email="runshare@test.com", password="pass")
        thread = ChatThread.objects.create(created_by=user)
        run = SubAgentRun.objects.create(
            thread=thread, user=user, prompt="task",
            shared_canvases=[{"title": "Draft", "content": "SNAPSHOT BODY"}],
        )
        from chat.subagent_service import run_subagent

        run_subagent(run.id)

        request = mock_svc.return_value.run_via_stream.call_args[0][1]
        system = request.messages[0].content
        self.assertIn('## Shared canvas: "Draft"', system)
        self.assertIn("SNAPSHOT BODY", system)
