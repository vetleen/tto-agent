"""Between-turn history (chat/history_checkpoint.py + ChatConsumer._load_history).

Between checkpoints the history is append-only, so consecutive turns share a
byte-identical prefix (prompt cache). What the tool loop rewrote (stubs, trimmed
calls, dropped calls) is saved and replayed. A checkpoint summarises only when the
conversation overflows its budget; tool pressure stubs, the stub cap drops.
"""

from unittest.mock import AsyncMock, patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat.consumers import MAX_HISTORY_TOKENS, ChatConsumer
from chat.history_checkpoint import (
    CTX_BODY,
    CTX_CALLS,
    CTX_DROPPED,
    CTX_TOKENS,
    TOOL_INDEX_HEADER,
    apply_checkpoint_rows,
    history_usage,
    join_summary,
    needs_checkpoint,
    persist_context_edit,
    plan_checkpoint,
    run_checkpoint,
    save_summary,
    split_summary,
)
from chat.models import ChatMessage, ChatThread
from core.tokens import count_tokens
from llm.context_budget import HistoryBudgets

User = get_user_model()

# history 10k: tool pressure past 7k; conversation 3k; stubs 500.
BUDGETS = HistoryBudgets(history=10_000, conversation=3_000, summary_target=1_500, stub_cap=500)
ROOMY = HistoryBudgets(history=1_000_000, conversation=500_000, summary_target=5_000, stub_cap=500_000)


class _ThreadMixin:
    def setUp(self):
        self.user = User.objects.create_user(email="ckpt@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.consumer.user = self.user
        self.consumer.resolved_prefs = None

    def _msg(self, content, role="user", *, tool_call_id=None, tool_calls=None,
             hidden=False, tokens=None, metadata=None):
        meta = dict(metadata or {})
        if tool_calls:
            meta["tool_calls"] = tool_calls
        return ChatMessage.objects.create(
            thread=self.thread, role=role, content=content, tool_call_id=tool_call_id,
            metadata=meta, is_hidden_from_user=hidden,
            token_count=tokens if tokens is not None else count_tokens(content),
        )

    def _turn(self, n, *, tool=True, tokens=None, tool_tokens=None):
        """One user round: question, a tool call + result, an answer."""
        self._msg(f"Question {n}", tokens=tokens)
        if tool:
            self._msg("", "assistant", hidden=True,
                      tool_calls=[{"id": f"c{n}", "name": "web_search", "arguments": {"query": str(n)}}])
            self._msg(f"RESULT {n}", "tool", tool_call_id=f"c{n}",
                      tokens=tool_tokens if tool_tokens is not None else tokens)
        self._msg(f"Answer {n}", "assistant", tokens=tokens)

    def _tool(self, cid):
        return ChatMessage.objects.get(thread=self.thread, role="tool", tool_call_id=cid)


class AppendOnlyHistoryTests(_ThreadMixin, TransactionTestCase):

    async def _load(self):
        return await self.consumer._load_history(self.thread)

    async def test_next_turn_extends_previous_history_byte_for_byte(self):
        """With no checkpoint in between, turn N+1's history starts with turn
        N's history — tool results from older turns stay as they were (no moving
        stub boundary), so the prompt-cache prefix holds."""
        for n in range(4):
            await database_sync_to_async(self._turn)(n)
        with self.settings(CONTEXT_RAW_TOOL_TURNS=2):
            before = (await self._load())["messages"]
            await database_sync_to_async(self._turn)(4)
            await database_sync_to_async(self._turn)(5)
            after = (await self._load())["messages"]
        self.assertEqual(after[:len(before)], before)
        self.assertGreater(len(after), len(before))
        tool_contents = [m["content"] for m in after if m["role"] == "tool"]
        self.assertEqual(tool_contents, [f"RESULT {n}" for n in range(6)])

    async def test_conversation_overflow_is_due(self):
        # Without a model the conversation budget is MAX_HISTORY_TOKENS // 6.
        await database_sync_to_async(self._msg)("big", tokens=MAX_HISTORY_TOKENS // 5)
        meta = (await self._load())["meta"]
        self.assertTrue(meta["needs_summary"])
        self.assertFalse(meta["over_budget"])
        self.assertIn("budgets", meta)

    async def test_tool_pressure_is_due(self):
        await database_sync_to_async(self._turn)(0, tokens=10, tool_tokens=int(MAX_HISTORY_TOKENS * 0.75))
        meta = (await self._load())["meta"]
        self.assertTrue(meta["needs_summary"])
        self.assertFalse(meta["over_budget"])

    async def test_small_history_needs_nothing(self):
        await database_sync_to_async(self._turn)(0, tokens=100)
        meta = (await self._load())["meta"]
        self.assertFalse(meta["needs_summary"])
        self.assertFalse(meta["over_budget"])

    async def test_over_budget_falls_back_to_sliding_window(self):
        await database_sync_to_async(self._msg)("OLD", tokens=MAX_HISTORY_TOKENS)
        # Fills the fallback's recency overlap (2k without a model) on its own.
        await database_sync_to_async(self._msg)("NEW", tokens=2_000)
        result = await self._load()
        self.assertTrue(result["meta"]["over_budget"])
        self.assertTrue(result["meta"]["needs_summary"])
        self.assertEqual([m["content"] for m in result["messages"]], ["NEW"])

    async def test_saved_body_and_effective_tokens_are_replayed(self):
        """A saved stub is what the history replays, and what it costs: a raw
        result over budget no longer pushes the history into the fallback."""
        @database_sync_to_async
        def _setup():
            self._turn(0, tokens=10, tool_tokens=MAX_HISTORY_TOKENS * 2)
            row = self._tool("c0")
            row.metadata = {**row.metadata, CTX_BODY: "[Earlier result of web_search(query=0) …]",
                            CTX_TOKENS: 12}
            row.save(update_fields=["metadata"])
            self._turn(1, tokens=10)

        await _setup()
        result = await self._load()
        self.assertFalse(result["meta"]["over_budget"])
        tools = [m["content"] for m in result["messages"] if m["role"] == "tool"]
        self.assertEqual(tools, ["[Earlier result of web_search(query=0) …]", "RESULT 1"])

    async def test_dropped_calls_are_not_replayed(self):
        @database_sync_to_async
        def _setup():
            self._turn(0)
            self._turn(1)
            for row in ChatMessage.objects.filter(thread=self.thread).exclude(role="user"):
                if row.tool_call_id == "c0" or (row.metadata or {}).get("tool_calls", [{}])[0].get("id") == "c0":
                    row.metadata = {**row.metadata, CTX_DROPPED: True}
                    row.save(update_fields=["metadata"])

        await _setup()
        messages = (await self._load())["messages"]
        ids = [m.get("tool_call_id") for m in messages if m["role"] == "tool"]
        self.assertEqual(ids, ["c1"])
        calls = [tc["id"] for m in messages for tc in m.get("tool_calls") or []]
        self.assertEqual(calls, ["c1"])
        # The narrative of both rounds stays.
        self.assertIn("Answer 0", [m["content"] for m in messages])

    async def test_trimmed_calls_are_replayed(self):
        @database_sync_to_async
        def _setup():
            self._turn(0)
            row = ChatMessage.objects.get(thread=self.thread, role="assistant", is_hidden_from_user=True)
            row.metadata = {**row.metadata, CTX_CALLS: [
                {"id": "c0", "name": "web_search", "arguments": {}},
            ]}
            row.save(update_fields=["metadata"])

        await _setup()
        messages = (await self._load())["messages"]
        call = [tc for m in messages for tc in m.get("tool_calls") or []][0]
        self.assertEqual(call["arguments"], {})

    async def test_pending_outcome_before_cutoff_still_shown(self):
        """An undelivered sub-agent outcome a checkpoint folded is still shown
        (else the turn it wakes wouldn't see it)."""
        from chat.models import SubAgentRun

        @database_sync_to_async
        def _setup():
            run = SubAgentRun.objects.create(
                thread=self.thread, user=self.user, prompt="t",
                status=SubAgentRun.Status.COMPLETED,
            )
            outcome = ChatMessage.objects.create(
                thread=self.thread, role="user", content="[Sub-agent result: x] done",
                is_hidden_from_user=True, token_count=5,
                metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            )
            cutoff = self._msg("folded", "assistant")
            self._msg("after cutoff")
            ChatThread.objects.filter(pk=self.thread.pk).update(
                summary="S", summary_token_count=1, summary_up_to_message_id=cutoff.id,
            )
            return outcome

        await _setup()
        contents = [m["content"] for m in (await self._load())["messages"] if m["role"] != "system"]
        self.assertIn("[Sub-agent result: x] done", contents[-1])
        self.assertNotIn("folded", contents)


class UsageTests(_ThreadMixin, TransactionTestCase):

    def test_conversation_excludes_tool_noise_and_counts_hidden_notices(self):
        self._turn(0, tokens=100, tool_tokens=5_000)
        self._msg("[Sub-agent result: a] r", hidden=True, tokens=300)
        usage = history_usage(self.thread)
        self.assertEqual(usage.conversation, 100 + 100 + 300)
        self.assertEqual(usage.history, 100 + 0 + 5_000 + 100 + 300)

    def test_needs_checkpoint(self):
        self._turn(0, tokens=100)
        self.assertFalse(needs_checkpoint(self.thread, BUDGETS))
        self._msg("long question", tokens=3_000)
        self.assertTrue(needs_checkpoint(self.thread, BUDGETS))   # conversation > 3k
        self.assertTrue(needs_checkpoint(self.thread, BUDGETS.as_dict()))
        self.assertFalse(needs_checkpoint(self.thread, None))

    def test_summary_index_round_trip(self):
        text = join_summary("Narrative.", ["- a(x=1)", "- b"])
        self.assertIn(TOOL_INDEX_HEADER, text)
        self.assertEqual(split_summary(text), ("Narrative.", ["- a(x=1)", "- b"]))
        self.assertEqual(split_summary("Plain"), ("Plain", []))


class PlanCheckpointTests(_ThreadMixin, TransactionTestCase):

    def test_tool_pressure_stubs_older_results_without_summarising(self):
        for n in range(4):
            self._turn(n, tokens=50, tool_tokens=2_000)  # history 8.6k > 7k, conversation 600
        plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2)
        self.assertFalse(plan.summarise)
        self.assertEqual(sorted(r.tool_call_id for r in plan.stub_rows), ["c0", "c1"])
        stub = plan.stub_rows[0].metadata[CTX_BODY]
        self.assertTrue(stub.startswith("[Earlier result of web_search(query="))
        self.assertEqual(plan.stub_rows[0].metadata[CTX_TOKENS], count_tokens(stub))
        apply_checkpoint_rows(plan)
        self.assertEqual(self._tool("c0").content, "RESULT 0")  # content stays raw
        self.assertNotIn(CTX_BODY, self._tool("c2").metadata)  # kept rounds raw
        # Applied: nothing left to stub, so no plan.
        self.assertIsNone(plan_checkpoint(self.thread, BUDGETS, keep_rounds=2))

    def test_conversation_overflow_summarises_and_keeps_two_rounds(self):
        for n in range(4):
            self._turn(n, tokens=800, tool_tokens=10)  # conversation 6.4k > 3k
        plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2)
        self.assertTrue(plan.summarise)
        self.assertEqual(plan.folded_count, 8)
        self.assertEqual(
            [m.content for m in plan.to_summarise],
            ["Question 0", "Answer 0", "Question 1", "Answer 1"],
        )
        self.assertEqual(ChatMessage.objects.get(pk=plan.new_cutoff_id).content, "Answer 1")
        self.assertEqual(plan.index_lines, ["- web_search(query=0)", "- web_search(query=1)"])
        self.assertEqual(plan.summary_target, BUDGETS.summary_target)
        self.assertFalse(plan.stub_rows)

    def test_tool_index_is_capped_and_extends_the_old_one(self):
        first = self._msg("old")
        ChatThread.objects.filter(pk=self.thread.pk).update(
            summary=join_summary("Old.", ["- older(x=1)"]), summary_token_count=5,
            summary_up_to_message_id=first.id,
        )
        for n in range(4):
            self._turn(n)
        with self.settings(CONTEXT_TOOL_INDEX_MAX_ENTRIES=2):
            plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2, force_summary=True)
        self.assertEqual(plan.index_lines, ["- web_search(query=0)", "- web_search(query=1)"])
        with self.settings(CONTEXT_TOOL_INDEX_MAX_ENTRIES=20):
            plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2, force_summary=True)
        self.assertEqual(plan.index_lines[0], "- older(x=1)")

    def test_tool_index_is_one_line_per_call(self):
        self._msg("Q0")
        self._msg("", "assistant", hidden=True, tool_calls=[
            {"id": "w", "name": "canvas_write", "arguments": {"content": "# Title\n\nBody\nmore"}},
        ])
        self._msg("ok", "tool", tool_call_id="w")
        self._msg("A0", "assistant")
        self._turn(1)
        self._turn(2)
        plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2, force_summary=True)
        self.assertEqual(plan.index_lines, ["- canvas_write(content=# Title Body more)"])

    def test_hidden_user_rows_do_not_start_rounds(self):
        self._turn(0)
        self._turn(1)
        # A sub-agent notice (hidden user row) is not a round of its own.
        self._msg("[Sub-agent result: a] r", hidden=True)
        self._msg("seeded reply", "assistant")
        # Only two real rounds: nothing before them to fold.
        self.assertIsNone(plan_checkpoint(self.thread, BUDGETS, keep_rounds=2, force_summary=True))

    def test_nothing_due_is_no_plan(self):
        self._turn(0)
        self._turn(1)
        self._turn(2)
        self.assertIsNone(plan_checkpoint(self.thread, ROOMY, keep_rounds=2))

    def test_keep_zero_folds_everything(self):
        self._turn(0)
        self._turn(1)
        plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=0, force_summary=True)
        self.assertEqual(plan.folded_count, 8)

    def test_stub_cap_drops_the_oldest_stubbed_calls(self):
        for n in range(30):
            self._turn(n, tokens=10, tool_tokens=10)
        for n in range(28):  # every older result already a saved stub (~25 tokens each)
            row = self._tool(f"c{n}")
            body = f"[Earlier result of web_search(query={n}) was cleared to save context. Call the tool again if you still need it.]"
            row.metadata = {**row.metadata, CTX_BODY: body, CTX_TOKENS: count_tokens(body)}
            row.save(update_fields=["metadata"])
        plan = plan_checkpoint(self.thread, BUDGETS, keep_rounds=2)
        dropped_calls = sorted({r.tool_call_id for r in plan.drop_rows if r.role == "tool"},
                               key=lambda c: int(c[1:]))
        self.assertTrue(dropped_calls)
        self.assertEqual(dropped_calls[0], "c0")  # oldest first
        # Each dropped call takes its assistant row with it.
        self.assertEqual(
            sum(1 for r in plan.drop_rows if r.role == "assistant"), len(dropped_calls),
        )
        apply_checkpoint_rows(plan)
        left = sum(
            count_tokens(self._tool(f"c{n}").metadata[CTX_BODY])
            for n in range(28) if not self._tool(f"c{n}").metadata.get(CTX_DROPPED)
        )
        self.assertLessEqual(left, int(BUDGETS.stub_cap * 0.25))

    def test_save_summary_lock(self):
        a = self._msg("a")
        b = self._msg("b")
        self.assertTrue(save_summary(self.thread.id, "S1", a.id, 1, expected_cutoff_id=None))
        # A writer that planned against "no summary" loses to the one that landed.
        self.assertFalse(save_summary(self.thread.id, "S2", b.id, 2, expected_cutoff_id=None))
        self.assertTrue(save_summary(self.thread.id, "S2", b.id, 2, expected_cutoff_id=a.id))


class PersistContextEditTests(_ThreadMixin, TransactionTestCase):

    def test_saves_bodies_calls_and_drops(self):
        self._turn(0)
        self._turn(1)
        persist_context_edit(self.thread.id, {
            "results": {"c1": "STUB 1"},
            "calls": {"c1": {"query": ""}},
            "dropped": ["c0"],
        })
        r1 = self._tool("c1")
        self.assertEqual(r1.content, "RESULT 1")
        self.assertEqual(r1.metadata[CTX_BODY], "STUB 1")
        self.assertEqual(r1.metadata[CTX_TOKENS], count_tokens("STUB 1"))
        self.assertTrue(self._tool("c0").metadata[CTX_DROPPED])
        calls = {
            r.metadata["tool_calls"][0]["id"]: r for r in
            ChatMessage.objects.filter(thread=self.thread, role="assistant", is_hidden_from_user=True)
        }
        self.assertEqual(calls["c1"].metadata[CTX_CALLS][0]["arguments"], {"query": ""})
        self.assertTrue(calls["c0"].metadata[CTX_DROPPED])

    async def test_consumer_splits_pending_and_stored_edits(self):
        """This round's not-yet-saved calls/results get the edit on their pending
        entries; earlier rows are updated in place."""
        await database_sync_to_async(self._turn)(0)
        pending_calls = [{"tool_call_id": "c9", "tool_name": "web_search", "arguments": {"q": "x"}}]
        pending_results = [{"tool_call_id": "c9", "result": "RAW 9"}]
        await self.consumer._apply_context_edit(self.thread, {
            "results": {"c0": "STUB 0", "c9": "DEDUP 9"},
            "calls": {"c9": {"q": ""}},
            "dropped": [],
        }, pending_calls, pending_results)
        self.assertEqual(pending_results[0]["ctx_body"], "DEDUP 9")
        self.assertEqual(pending_calls[0]["ctx_arguments"], {"q": ""})
        row = await database_sync_to_async(self._tool)("c0")
        self.assertEqual(row.metadata[CTX_BODY], "STUB 0")

        await self.consumer._persist_tool_loop_messages(self.thread, pending_calls, pending_results)
        saved = await database_sync_to_async(self._tool)("c9")
        self.assertEqual(saved.content, "RAW 9")
        self.assertEqual(saved.metadata[CTX_BODY], "DEDUP 9")

    async def test_a_midturn_stub_survives_into_later_turns(self):
        """The bug this fixes: a prune stubbed an earlier turn's result, the next
        turn reloaded it raw (cache break + re-prune), every turn."""
        @database_sync_to_async
        def _setup():
            self._turn(0)
            self._turn(1)

        await _setup()
        await self.consumer._apply_context_edit(
            self.thread, {"results": {"c0": "STUB 0"}, "calls": {}, "dropped": []}, [], [],
        )
        first = (await self.consumer._load_history(self.thread))["messages"]
        await database_sync_to_async(self._turn)(2)
        second = (await self.consumer._load_history(self.thread))["messages"]
        self.assertEqual(second[:len(first)], first)
        self.assertEqual([m["content"] for m in second if m["role"] == "tool"],
                         ["STUB 0", "RESULT 1", "RESULT 2"])


@patch("chat.services.generate_summary", new_callable=AsyncMock, return_value="NEW SUMMARY")
class RunCheckpointTests(_ThreadMixin, TransactionTestCase):

    async def test_conversation_overflow_summarises_and_history_follows(self, gen):
        for n in range(4):
            await database_sync_to_async(self._turn)(n, tokens=800, tool_tokens=10)
        result = await run_checkpoint(self.thread, user_id=self.user.pk, model=None, budgets=BUDGETS)
        self.assertTrue(result.saved)
        self.assertTrue(result.summarised)
        self.assertEqual(result.folded_count, 8)
        gen.assert_awaited_once()
        self.assertEqual(
            [m.content for m in gen.await_args.args[0]],
            ["Question 0", "Answer 0", "Question 1", "Answer 1"],
        )
        self.assertEqual(gen.await_args.kwargs["target_tokens"], BUDGETS.summary_target)

        messages = (await self.consumer._load_history(self.thread))["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(messages[0]["content"].startswith("Summary of earlier conversation:\nNEW SUMMARY"))
        self.assertIn("- web_search(query=1)", messages[0]["content"])
        users = [m["content"] for m in messages if m["role"] == "user"]
        self.assertEqual(users, ["Question 2", "Question 3"])

    async def test_tool_pressure_never_calls_the_summariser(self, gen):
        for n in range(4):
            await database_sync_to_async(self._turn)(n, tokens=50, tool_tokens=2_000)
        result = await run_checkpoint(self.thread, user_id=self.user.pk, model=None, budgets=BUDGETS)
        gen.assert_not_awaited()
        self.assertTrue(result.saved)
        self.assertEqual(result.stubbed, 2)
        messages = (await self.consumer._load_history(self.thread))["messages"]
        tools = [m["content"] for m in messages if m["role"] == "tool"]
        self.assertTrue(tools[0].startswith("[Earlier result of web_search"))
        self.assertEqual(tools[2:], ["RESULT 2", "RESULT 3"])
        # All four questions are still there, verbatim.
        self.assertEqual([m["content"] for m in messages if m["role"] == "user"],
                         [f"Question {n}" for n in range(4)])

    async def test_nothing_to_fold_saves_nothing(self, gen):
        await database_sync_to_async(self._turn)(0, tokens=5_000)
        result = await run_checkpoint(self.thread, user_id=self.user.pk, model=None, budgets=BUDGETS)
        self.assertFalse(result.saved)
        gen.assert_not_awaited()

    async def test_consumer_checkpoint_if_due_skips_when_not_due(self, gen):
        await database_sync_to_async(self._turn)(0)
        await self.consumer._checkpoint_if_due(self.thread, ROOMY.as_dict())
        gen.assert_not_awaited()

    async def test_consumer_checkpoint_if_due_runs_when_due(self, gen):
        for n in range(3):
            await database_sync_to_async(self._turn)(n, tokens=1_000)
        await self.consumer._checkpoint_if_due(self.thread, BUDGETS.as_dict())
        gen.assert_awaited_once()
        await database_sync_to_async(self.thread.refresh_from_db)()
        self.assertTrue(self.thread.summary.startswith("NEW SUMMARY"))

    @override_settings(CONTEXT_CHECKPOINT_WAIT_SECONDS=5)
    async def test_scheduled_checkpoint_survives_and_is_awaited(self, gen):
        for n in range(3):
            await database_sync_to_async(self._turn)(n, tokens=1_000)
        self.consumer._schedule_checkpoint(self.thread, BUDGETS.as_dict())
        await self.consumer._await_pending_checkpoint()
        self.assertTrue(self.consumer._checkpoint_task.done())
        await database_sync_to_async(self.thread.refresh_from_db)()
        self.assertTrue(self.thread.summary.startswith("NEW SUMMARY"))

    async def test_compact_works_when_a_summary_exists(self, gen):
        """/compact used to save with no expected cutoff, which can never match
        once a summary exists — it reported success and saved nothing."""
        self.consumer._send_command_result = AsyncMock()
        self.consumer._get_thread_by_id = AsyncMock(return_value=self.thread)

        @database_sync_to_async
        def _setup():
            first = self._msg("old")
            ChatThread.objects.filter(pk=self.thread.pk).update(
                summary="OLD SUMMARY", summary_token_count=2, summary_up_to_message_id=first.id,
            )
            self._turn(1)

        await _setup()
        await self.consumer._cmd_compact("", {"thread_id": str(self.thread.id)})
        status, text = self.consumer._send_command_result.await_args.args[1:3]
        self.assertEqual(status, "ok", text)
        self.assertIn("Compacted 4 messages", text)
        self.assertEqual(gen.await_args.kwargs["existing_summary"], "OLD SUMMARY")
        await database_sync_to_async(self.thread.refresh_from_db)()
        self.assertTrue(self.thread.summary.startswith("NEW SUMMARY"))
