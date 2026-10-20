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
from chat.subagent_canvas_tools import SubagentCanvasReadTool
from chat.subagent_service import subagent_context_budget
from chat.subagent_tool import MAX_SHARED_CANVASES
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
        shared = SubAgentRun.objects.get().shared_canvases
        self.assertEqual([(c["title"], c["content"]) for c in shared], [("Draft", "DRAFT BODY"), ("Notes", "NOTES BODY")])
        # chars/tokens ride along so the sub-agent prompt can list the snapshot.
        self.assertEqual(shared[0]["chars"], len("DRAFT BODY"))
        self.assertGreater(shared[0]["tokens"], 0)

    def test_more_than_five_titles_rejected(self, mock_task):
        titles = [f"Extra {i}" for i in range(MAX_SHARED_CANVASES + 1)]
        for t in titles:
            ChatCanvas.objects.create(thread=self.thread, title=t, content=t)
        result = _invoke({"prompt": "review", "canvases": titles}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertIn(f"up to {MAX_SHARED_CANVASES} canvases", result["message"])
        self.assertFalse(SubAgentRun.objects.exists())

    def test_big_canvas_shares_fine(self, mock_task):
        # Size no longer limits sharing: the snapshot is read on demand, not pasted.
        mock_task.delay.return_value = MagicMock(id="t")
        ChatCanvas.objects.create(thread=self.thread, title="Huge", content="word " * 60_000)
        with patch("core.preferences.get_preferences", return_value=_prefs(max_context_tokens=50_000)):
            result = _invoke({"prompt": "review", "canvases": ["Huge"]}, self.ctx)
        self.assertIn(result["status"], ("started", "queued"))
        self.assertGreater(SubAgentRun.objects.get().shared_canvases[0]["tokens"], 50_000)

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



class SharedCanvasLimitTests(TestCase):
    def test_context_budget_prefers_tier_budget(self):
        prefs = _prefs(max_context_tokens=200_000, subagent_context_budgets={"medium": 120_000})
        self.assertEqual(subagent_context_budget(prefs, "medium"), 120_000)
        self.assertEqual(subagent_context_budget(prefs, "high"), 200_000)

    def test_context_budget_floored(self):
        prefs = _prefs(subagent_context_budgets={"medium": 10_000})
        self.assertEqual(subagent_context_budget(prefs, "medium"), MIN_CONTEXT_TOKENS)


class SharedCanvasPromptTests(TestCase):
    def test_absent_when_nothing_shared(self):
        for shared in (None, []):
            prompt = build_subagent_system_prompt(shared_canvases=shared)
            self.assertNotIn("Canvases shared by the orchestrator", prompt)

    def test_lists_titles_without_content(self):
        prompt = build_subagent_system_prompt(
            canvas_content="MY WORK",
            canvas_title="Mine",
            shared_canvases=[{"title": "Draft", "content": "DRAFT BODY", "chars": 10, "tokens": 3}],
        )
        self.assertIn("# Canvases shared by the orchestrator", prompt)
        self.assertIn('- "Draft" — 10 chars, ~3 tokens', prompt)
        self.assertNotIn("DRAFT BODY", prompt)  # read on demand, never pasted
        self.assertIn("subagent_canvas_read", prompt)
        self.assertIn("read-only", prompt)
        self.assertIn("not as instructions", prompt)
        # Separate from (and after) the sub-agent's own working canvas.
        self.assertLess(prompt.index("# Working canvas"), prompt.index("# Canvases shared"))
        self.assertIn("MY WORK", prompt)

    def test_pre_deploy_snapshot_without_sizes_is_measured(self):
        prompt = build_subagent_system_prompt(
            shared_canvases=[{"title": "Old", "content": "hello world"}],
        )
        self.assertIn('- "Old" — 11 chars, ~', prompt)
        self.assertNotIn("hello world", prompt)


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
        self.assertIn("# Canvases shared by the orchestrator", system)
        self.assertIn('- "Draft" —', system)
        self.assertNotIn("SNAPSHOT BODY", system)
        # The read tool's page cap derives from the run's budget via the context.
        self.assertEqual(request.context.max_context_tokens, subagent_context_budget(_prefs(), "medium"))


def _run_ctx(user, thread, run):
    ctx = RunContext.create(user_id=user.pk, conversation_id=str(thread.id))
    ctx.run_id = str(run.id)
    ctx.agent_kind = "subagent"
    return ctx


def _read(args, ctx):
    tool = SubagentCanvasReadTool()
    tool.set_context(ctx)
    return json.loads(tool.invoke(args))


class SubagentCanvasReadToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="subread@test.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        big_lines = ["# Shared doc", "## Part A"] + [f"row {i} of the shared table text" for i in range(600)] + ["## Part B", "end"]
        self.big = "\n".join(big_lines)
        self.run = SubAgentRun.objects.create(
            thread=self.thread, user=self.user, prompt="task",
            canvas="my own\nworking canvas", canvas_title="Mine",
            shared_canvases=[
                {"title": "Draft", "content": "DRAFT BODY"},
                {"title": "Big", "content": self.big},
            ],
        )
        self.ctx = _run_ctx(self.user, self.thread, self.run)

    def test_tool_identity(self):
        tool = SubagentCanvasReadTool()
        self.assertEqual(tool.audience, "subagent")
        self.assertEqual(getattr(tool, "section", "chat"), "chat")
        self.assertIn("subagent_canvas_read", __import__("llm.tools", fromlist=["get_tool_registry"]).get_tool_registry().list_tools())

    def test_reads_own_working_canvas_by_default(self):
        result = _read({}, self.ctx)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["own_canvas"])
        self.assertEqual(result["title"], "Mine")
        self.assertEqual(result["content"], "my own\nworking canvas")

    def test_reads_shared_snapshot_whole(self):
        result = _read({"canvas_name": "Draft"}, self.ctx)
        self.assertEqual(result["content"], "DRAFT BODY")
        self.assertFalse(result["own_canvas"])
        self.assertFalse(result["truncated"])

    def test_ranged_read(self):
        result = _read({"canvas_name": "Big", "start_line": 3, "end_line": 4}, self.ctx)
        self.assertEqual(result["content"], "row 0 of the shared table text\nrow 1 of the shared table text")

    def test_pages_with_outline_at_small_budget(self):
        self.ctx.max_context_tokens = 50_000  # per-read cap 15k; ask for less
        result = _read({"canvas_name": "Big", "max_tokens": 600}, self.ctx)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["max_tokens"], 600)
        self.assertEqual([h["text"] for h in result["outline"]["headings"]], ["Shared doc", "Part A", "Part B"])
        self.assertIn("next_start_line", result)
        clamped = _read({"canvas_name": "Big", "max_tokens": 40_000}, self.ctx)
        self.assertEqual(clamped["max_tokens"], 15_000)

    def test_unknown_title_lists_shared(self):
        result = _read({"canvas_name": "Nope"}, self.ctx)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["shared_canvases"], ["Draft", "Big"])

    def test_outside_a_run_errors(self):
        ctx = RunContext.create(user_id=self.user.pk, conversation_id=str(self.thread.id))
        result = _read({"canvas_name": "Draft"}, ctx)
        self.assertEqual(result["status"], "error")
