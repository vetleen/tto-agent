"""Tests for the token-aware _load_history in ChatConsumer."""

from datetime import timedelta

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TransactionTestCase
from django.utils import timezone

from chat.consumers import (
    MAX_HISTORY_TOKENS,
    OVERLAP_TOKENS,
    ChatConsumer,
    _ensure_turn_starts_on_user,
)
from chat.models import ChatMessage, ChatThread, Loop, SubAgentRun
from chat.prompts import SUBAGENT_WAKE_NOTE, TURN_CONTINUATION_NOTE
from core.tokens import count_tokens

User = get_user_model()


class LoadHistoryTests(TransactionTestCase):
    """Test _load_history returns correct structure and respects token budget."""

    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", password="pass"
        )
        self.thread = ChatThread.objects.create(
            created_by=self.user
        )
        self.consumer = ChatConsumer()

    @database_sync_to_async
    def _make_msg(self, content, role="user", tool_call_id=None):
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(content),
        )

    @database_sync_to_async
    def _make_tool_call_msg(self, tool_calls):
        """Create an assistant message with tool_calls stored in metadata."""
        return ChatMessage.objects.create(
            thread=self.thread,
            role="assistant",
            content="",
            metadata={"tool_calls": tool_calls},
            token_count=0,
        )

    @database_sync_to_async
    def _update_thread(self, **kwargs):
        ChatThread.objects.filter(pk=self.thread.pk).update(**kwargs)

    async def test_returns_dict_with_messages_and_meta(self):
        await self._make_msg("Hello")
        result = await self.consumer._load_history(self.thread)
        self.assertIn("messages", result)
        self.assertIn("meta", result)
        self.assertIsInstance(result["messages"], list)
        self.assertIsInstance(result["meta"], dict)

    async def test_meta_keys(self):
        await self._make_msg("Hello")
        meta = (await self.consumer._load_history(self.thread))["meta"]
        self.assertIn("total_messages", meta)
        self.assertIn("included_messages", meta)
        self.assertIn("has_summary", meta)
        self.assertIn("needs_summary", meta)
        self.assertIn("history_budget_tokens", meta)
        self.assertIn("included_history_tokens", meta)
        self.assertIn("summary_tokens", meta)
        self.assertIn("history_truncated", meta)

    async def test_all_messages_included_when_under_budget(self):
        for i in range(5):
            await self._make_msg(f"Message {i}")
        result = await self.consumer._load_history(self.thread)
        self.assertEqual(result["meta"]["total_messages"], 5)
        self.assertEqual(result["meta"]["included_messages"], 5)
        self.assertFalse(result["meta"]["needs_summary"])
        self.assertFalse(result["meta"]["history_truncated"])
        self.assertGreater(result["meta"]["included_history_tokens"], 0)

    async def test_respects_token_budget(self):
        # Create messages that will exceed MAX_HISTORY_TOKENS
        big_content = "word " * 5000  # ~5000 tokens each
        for _ in range(6):  # ~30k tokens > 20k budget
            await self._make_msg(big_content)

        result = await self.consumer._load_history(self.thread)
        meta = result["meta"]
        self.assertEqual(meta["total_messages"], 6)
        self.assertLess(meta["included_messages"], 6)
        self.assertTrue(meta["needs_summary"])
        self.assertTrue(meta["history_truncated"])

    async def test_includes_summary_as_system_message(self):
        summary_text = "Previous conversation about cats."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
        )

        await self._make_msg("New message")
        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Previous conversation about cats", messages[0]["content"])
        self.assertTrue(result["meta"]["has_summary"])
        self.assertEqual(
            result["meta"]["summary_tokens"], count_tokens(summary_text),
        )
        self.assertGreater(
            result["meta"]["included_history_tokens"],
            result["meta"]["summary_tokens"],
        )

    async def test_no_summary_when_empty(self):
        await self._make_msg("Hello")
        result = await self.consumer._load_history(self.thread)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 0)
        self.assertFalse(result["meta"]["has_summary"])

    async def test_messages_in_chronological_order(self):
        await self._make_msg("First")
        await self._make_msg("Second")
        await self._make_msg("Third")

        result = await self.consumer._load_history(self.thread)
        contents = [m["content"] for m in result["messages"]]
        self.assertEqual(contents, ["First", "Second", "Third"])

    async def test_summarised_msgs_not_replayed(self):
        """History is the summary + the messages after its cutoff: a summarised
        message is never replayed raw, however small (no overlap window)."""
        m1 = await self._make_msg("Short old message")
        await self._make_msg("Short new message")

        summary_text = "Summary of old stuff."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
            summary_up_to_message_id=m1.id,
            summary_message_count=1,
        )

        result = await self.consumer._load_history(self.thread)
        non_system = [m["content"] for m in result["messages"] if m["role"] != "system"]
        self.assertEqual(non_system, ["Short new message"])

    async def test_tool_messages_included_in_history(self):
        """Tool and assistant-with-tool-calls messages are loaded from history."""
        await self._make_msg("What's the weather?")
        # Assistant requesting tool call (metadata stores tool_calls)
        await self._make_tool_call_msg(
            [{"id": "tc1", "name": "get_weather", "arguments": {"city": "Oslo"}}]
        )
        # Tool result
        await self._make_msg('{"temp": 5}', role="tool", tool_call_id="tc1")
        # Final assistant response
        await self._make_msg("It's 5 degrees in Oslo.", role="assistant")

        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        # Should have 4 messages total
        self.assertEqual(len(messages), 4)
        # 2nd message: assistant with tool_calls in metadata
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("tool_calls", messages[1])
        self.assertEqual(messages[1]["tool_calls"][0]["name"], "get_weather")
        # 3rd message: tool result
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "tc1")

    async def test_large_summarised_msgs_excluded_from_raw_history(self):
        """A summarised message that is outside the overlap window is not shown raw."""
        # Each message alone exceeds OVERLAP_TOKENS so it pushes the other out of the window
        big_content = "word " * 2500  # well over 2000 tokens
        m1 = await self._make_msg(big_content)
        await self._make_msg(big_content)

        summary_text = "Summary of old stuff."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
            summary_up_to_message_id=m1.id,
            summary_message_count=1,
        )

        result = await self.consumer._load_history(self.thread)
        # Overlap = [m2] alone already >= OVERLAP_TOKENS, so m1 is outside the overlap
        non_system = [m for m in result["messages"] if m["role"] != "system"]
        self.assertEqual(len(non_system), 1)
        self.assertEqual(non_system[0]["content"], big_content)

    async def test_summarisation_excludes_redacted_messages(self):
        """Redacted (guardrail-blocked) messages must not leak into summaries."""
        from chat.history_checkpoint import plan_checkpoint

        await self._make_msg("older message")

        @database_sync_to_async
        def _make_redacted():
            return ChatMessage.objects.create(
                thread=self.thread,
                role="user",
                content="[This message was removed by the content safety system.]",
                is_redacted=True,
                token_count=0,
            )

        await _make_redacted()
        await self._make_msg("newest message")

        plan = await database_sync_to_async(plan_checkpoint)(
            self.thread, keep_rounds=0, force_summary=True,
        )
        self.assertEqual(
            [m.content for m in plan.to_summarise], ["older message", "newest message"],
        )
        to_summarise = plan.to_summarise
        for msg in to_summarise:
            self.assertFalse(
                msg.is_redacted,
                "Redacted messages must not appear in summarisation candidates",
            )

    async def test_orphan_tool_messages_stripped(self):
        """Tool results with no matching assistant tool_call are excluded."""
        await self._make_msg("Hello")
        # Valid pair: assistant with tool_calls + matching tool result
        await self._make_tool_call_msg(
            [{"id": "tc_valid", "name": "search", "arguments": {}}]
        )
        await self._make_msg("search result", role="tool", tool_call_id="tc_valid")
        # Orphan: tool result with no matching tool call
        await self._make_msg("orphan result", role="tool", tool_call_id="no_such_call")
        await self._make_msg("Final answer", role="assistant")

        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "tc_valid")
        self.assertTrue(
            all(m.get("tool_call_id") != "no_such_call" for m in messages),
            "Orphan tool message should have been stripped",
        )

    async def test_consecutive_subagent_user_messages_merged(self):
        """Consecutive user messages from sub-agent results are merged."""
        await self._make_msg("[Sub-agent result: aaa11111]\nResult A")
        await self._make_msg("[Sub-agent result: bbb22222]\nResult B")
        await self._make_msg("OK", role="assistant")

        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        user_msgs = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertIn("[Sub-agent result: aaa11111]", user_msgs[0]["content"])
        self.assertIn("[Sub-agent result: bbb22222]", user_msgs[0]["content"])
        self.assertIn("---", user_msgs[0]["content"])

    async def test_consecutive_normal_user_messages_not_merged(self):
        """Consecutive non-sub-agent user messages stay separate."""
        await self._make_msg("First message")
        await self._make_msg("Second message")
        await self._make_msg("OK", role="assistant")

        result = await self.consumer._load_history(self.thread)
        messages = result["messages"]

        user_msgs = [m for m in messages if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 2)

    async def test_consecutive_subagent_failure_messages_merged(self):
        """Failure messages merge like results — two failed sub-agents landing
        together must not produce consecutive user messages (Anthropic rejects
        non-alternating roles)."""
        await self._make_msg("[Sub-agent failed: aaa11111]\nError: timeout")
        await self._make_msg("[Sub-agent result: bbb22222]\nResult B")
        await self._make_msg("OK", role="assistant")

        result = await self.consumer._load_history(self.thread)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertIn("[Sub-agent failed: aaa11111]", user_msgs[0]["content"])
        self.assertIn("[Sub-agent result: bbb22222]", user_msgs[0]["content"])

    async def test_user_dicts_carry_message_id(self):
        msg = await self._make_msg("Hello")
        await self._make_msg("Hi", role="assistant")
        result = await self.consumer._load_history(self.thread)
        user = [m for m in result["messages"] if m["role"] == "user"][0]
        assistant = [m for m in result["messages"] if m["role"] == "assistant"][0]
        self.assertEqual(user["message_id"], str(msg.id))
        self.assertNotIn("message_id", assistant)

    async def test_merge_keeps_attachment_ids_of_both_messages(self):
        await self._make_msg("[Sub-agent result: aaa11111]\nResult A")
        upload = await self._make_msg("See file")

        @database_sync_to_async
        def _set_meta():
            ChatMessage.objects.filter(pk=upload.pk).update(
                metadata={"attachment_ids": ["att-1"]}
            )

        await _set_meta()
        result = await self.consumer._load_history(self.thread)
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        self.assertEqual(len(user_msgs), 1)
        self.assertEqual(user_msgs[0]["attachment_ids"], ["att-1"])
        self.assertEqual(user_msgs[0]["message_id"], str(upload.id))

    @database_sync_to_async
    def _make_subagent_run(self, **overrides):
        fields = dict(
            thread=self.thread, user=self.user, prompt="task",
            status=SubAgentRun.Status.COMPLETED, result="Findings",
        )
        fields.update(overrides)
        return SubAgentRun.objects.create(**fields)

    @database_sync_to_async
    def _make_subagent_msg(self, content, run_id="fake-run-id"):
        return ChatMessage.objects.create(
            thread=self.thread,
            role="user",
            content=content,
            metadata={"source": "subagent", "subagent_run_id": str(run_id)},
            token_count=count_tokens(content),
            is_hidden_from_user=True,
        )

    async def test_meta_flags_subagent_results_in_history(self):
        run = await self._make_subagent_run()
        await self._make_subagent_msg("[Sub-agent result: aaa11111]\nFindings", run.id)
        meta = (await self.consumer._load_history(self.thread))["meta"]
        self.assertTrue(meta["has_subagent_results"])
        self.assertEqual(meta["subagent_run_ids"], [str(run.id)])

    async def test_meta_no_subagent_results_flag_false(self):
        await self._make_msg("Just a normal message")
        meta = (await self.consumer._load_history(self.thread))["meta"]
        self.assertFalse(meta["has_subagent_results"])
        self.assertEqual(meta["subagent_run_ids"], [])

    async def test_meta_excludes_delivered_runs(self):
        """Only outcomes still owed to the model are snapshotted; a run an
        earlier reply already reported stays in history but is not re-marked."""
        delivered = await self._make_subagent_run(delivered_at=timezone.now())
        await self._make_subagent_msg("[Sub-agent result: d]\nOld findings", delivered.id)
        pending = await self._make_subagent_run()
        await self._make_subagent_msg("[Sub-agent result: p]\nNew findings", pending.id)

        result = await self.consumer._load_history(self.thread)
        self.assertEqual(result["meta"]["subagent_run_ids"], [str(pending.id)])
        blob = " ".join(m["content"] for m in result["messages"])
        self.assertIn("Old findings", blob)
        self.assertIn("New findings", blob)

    async def test_meta_ignores_hidden_message_without_run(self):
        """A branched thread carries copied hidden messages whose run ids are
        foreign (here: not even a UUID): replayed as history, never snapshotted,
        never an error."""
        await self._make_subagent_msg("[Sub-agent result: aaa11111]\nFindings")
        result = await self.consumer._load_history(self.thread)
        self.assertEqual(result["meta"]["subagent_run_ids"], [])
        self.assertFalse(result["meta"]["has_subagent_results"])
        self.assertIn("Findings", result["messages"][-1]["content"])

    async def _previous_turn_with_midstream_result(self, run_id):
        """The 2026-10-15 prod shape: a sub-agent outcome persisted while the
        previous turn was still streaming sorts before that turn's last tool
        round and final reply."""
        await self._make_msg("Read the documents")
        await self._make_subagent_msg("[Sub-agent result: aaa11111]\nFindings", run_id)
        await self._make_tool_call_msg([{"id": "call_1", "name": "canvas_edit", "arguments": {}}])
        await self._make_msg('{"status": "ok"}', role="tool", tool_call_id="call_1")
        await self._make_msg("Updated the table.", role="assistant")

    async def test_pending_outcome_that_landed_midstream_is_loaded_last(self):
        run = await self._make_subagent_run()
        await self._previous_turn_with_midstream_result(run.id)

        messages = (await self.consumer._load_history(self.thread))["messages"]

        self.assertEqual(
            [m["role"] for m in messages], ["user", "assistant", "tool", "assistant", "user"],
        )
        self.assertIn("Findings", messages[-1]["content"])
        # The moved message doesn't break the tool call / result pairing.
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")

    async def test_delivered_outcome_keeps_timestamp_order(self):
        run = await self._make_subagent_run(delivered_at=timezone.now())
        await self._previous_turn_with_midstream_result(run.id)

        messages = (await self.consumer._load_history(self.thread))["messages"]

        self.assertEqual(
            [m["role"] for m in messages], ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(messages[-1]["content"], "Updated the table.")

    async def test_several_pending_outcomes_merge_into_one_trailing_user_turn(self):
        first = await self._make_subagent_run()
        second = await self._make_subagent_run()
        await self._previous_turn_with_midstream_result(first.id)
        await self._make_subagent_msg("[Sub-agent result: bbb22222]\nMore findings", second.id)

        messages = (await self.consumer._load_history(self.thread))["messages"]

        self.assertEqual(
            [m["role"] for m in messages], ["user", "assistant", "tool", "assistant", "user"],
        )
        self.assertIn("Findings", messages[-1]["content"])
        self.assertIn("More findings", messages[-1]["content"])

    async def test_moved_outcome_keeps_tool_results_raw(self):
        # A history that fits is replayed raw: no tool result is stubbed, even
        # one older than CONTEXT_RAW_TOOL_TURNS (stubbing is the over-budget
        # safety net only — it would rewrite mid-history every turn).
        await self._make_msg("Earlier request")
        await self._make_tool_call_msg([{"id": "call_0", "name": "canvas_read", "arguments": {}}])
        await self._make_msg('{"old": "result"}', role="tool", tool_call_id="call_0")
        run = await self._make_subagent_run()
        await self._previous_turn_with_midstream_result(run.id)

        with self.settings(CONTEXT_RAW_TOOL_TURNS=2):
            messages = (await self.consumer._load_history(self.thread))["messages"]

        tools = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}
        self.assertEqual(tools["call_1"], '{"status": "ok"}')
        self.assertEqual(tools["call_0"], '{"old": "result"}')


class EnsureTurnStartsOnUserTests(SimpleTestCase):
    """_ensure_turn_starts_on_user: a turn never starts on the agent's own messages."""

    def _msgs(self, *roles):
        from llm.types import Message

        return [Message(role="system", content="sys")] + [
            Message(role=r, content=f"{r} text", tool_call_id="c1" if r == "tool" else None)
            for r in roles
        ]

    def test_seeded_turn_ending_on_assistant_gets_wake_note(self):
        messages = self._msgs("user", "assistant")
        _ensure_turn_starts_on_user(messages, seed_mode=True)
        self.assertEqual(messages[-1].role, "user")
        self.assertEqual(messages[-1].content, SUBAGENT_WAKE_NOTE)
        self.assertEqual(len(messages), 4)

    def test_seeded_turn_ending_on_tool_result_gets_wake_note(self):
        messages = self._msgs("user", "assistant", "tool")
        _ensure_turn_starts_on_user(messages, seed_mode=True)
        self.assertEqual(messages[-1].content, SUBAGENT_WAKE_NOTE)

    def test_turn_ending_on_user_is_unchanged(self):
        messages = self._msgs("user", "assistant", "user")
        _ensure_turn_starts_on_user(messages, seed_mode=True)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1].content, "user text")

    def test_unseeded_turn_gets_generic_continuation(self):
        messages = self._msgs("user", "assistant")
        _ensure_turn_starts_on_user(messages, seed_mode=False)
        self.assertEqual(messages[-1].content, TURN_CONTINUATION_NOTE)

    def test_system_only_is_unchanged(self):
        messages = self._msgs()
        _ensure_turn_starts_on_user(messages, seed_mode=True)
        self.assertEqual(len(messages), 1)


class LoopPassScopingTests(TransactionTestCase):
    """`_load_history(scope_to_current_pass=True)` limits history to the current
    loop pass (messages at/after the latest ``loop_run`` marker), skips the
    cross-pass summary, and never asks for summarization.

    This is the post-sub-agent continuation on a fresh loop thread: it must see
    its own pass + the sub-agent result, not the unbounded 1:1 loop backlog.
    """

    def setUp(self):
        self.user = User.objects.create_user(email="scope@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()
        self.base = timezone.now() - timedelta(hours=1)

    @database_sync_to_async
    def _make_msg_at(self, content, minutes, role="user", metadata=None, hidden=False):
        """Create a message with an explicit created_at (auto_now_add can't be
        set on create, so update() it afterwards for deterministic ordering)."""
        msg = ChatMessage.objects.create(
            thread=self.thread, role=role, content=content,
            token_count=count_tokens(content),
            metadata=metadata or {}, is_hidden_from_user=hidden,
        )
        ChatMessage.objects.filter(pk=msg.pk).update(
            created_at=self.base + timedelta(minutes=minutes)
        )
        msg.refresh_from_db(fields=["created_at"])
        return msg

    @database_sync_to_async
    def _update_thread(self, **kwargs):
        ChatThread.objects.filter(pk=self.thread.pk).update(**kwargs)

    async def _build_two_pass_thread(self):
        """Prior pass (should be excluded) then current pass + sub-agent result."""
        await self._make_msg_at("old prompt", 0, metadata={"loop_run": True})
        await self._make_msg_at("PRIOR_REPLY", 1, role="assistant")
        await self._make_msg_at("CURRENT_PROMPT", 10, metadata={"loop_run": True})
        await self._make_msg_at("Started the scan.", 11, role="assistant")
        await self._make_msg_at(
            "[Sub-agent result: abcd1234]\nFINDINGS", 12,
            metadata={"source": "subagent", "subagent_run_id": "abcd1234"},
            hidden=True,
        )

    async def test_scope_excludes_prior_pass_keeps_current(self):
        await self._build_two_pass_thread()
        result = await self.consumer._load_history(
            self.thread, scope_to_current_pass=True
        )
        blob = " ".join(m["content"] for m in result["messages"])
        self.assertNotIn("old prompt", blob)
        self.assertNotIn("PRIOR_REPLY", blob)
        self.assertIn("CURRENT_PROMPT", blob)
        self.assertIn("FINDINGS", blob)
        # No cross-pass summary system message.
        self.assertFalse(any(m["role"] == "system" for m in result["messages"]))

    async def test_unscoped_control_includes_prior_pass(self):
        """Same thread, without scoping, replays everything (the buggy behaviour)."""
        await self._build_two_pass_thread()
        result = await self.consumer._load_history(self.thread)
        blob = " ".join(m["content"] for m in result["messages"])
        self.assertIn("PRIOR_REPLY", blob)
        self.assertIn("FINDINGS", blob)

    async def test_scope_skips_cross_pass_summary(self):
        marker = await self._make_msg_at("CURRENT_PROMPT", 10, metadata={"loop_run": True})
        await self._make_msg_at("Reply.", 11, role="assistant")
        summary_text = "Summary of many earlier passes."
        await self._update_thread(
            summary=summary_text,
            summary_token_count=count_tokens(summary_text),
            summary_up_to_message_id=marker.id,
        )
        result = await self.consumer._load_history(
            self.thread, scope_to_current_pass=True
        )
        self.assertFalse(result["meta"]["has_summary"])
        self.assertFalse(any(m["role"] == "system" for m in result["messages"]))

    async def test_scope_forces_needs_summary_false(self):
        """A large current pass must not trigger summarization when scoped —
        that would write a partial summary onto the fresh loop thread."""
        await self._make_msg_at("CURRENT_PROMPT", 10, metadata={"loop_run": True})
        big = "word " * 5000  # ~5k tokens
        for i in range(6):  # ~30k tokens > MAX_HISTORY_TOKENS budget
            await self._make_msg_at(big, 11 + i, role="assistant")

        scoped = await self.consumer._load_history(
            self.thread, scope_to_current_pass=True
        )
        self.assertFalse(scoped["meta"]["needs_summary"])
        # Control: unscoped, the same overflow does ask for summarization.
        unscoped = await self.consumer._load_history(self.thread)
        self.assertTrue(unscoped["meta"]["needs_summary"])

    async def test_scoped_pass_shows_pre_pass_undelivered_outcome(self):
        """A sub-agent dispatched in the previous pass that finishes after the
        next pass started: its outcome predates the marker but is still shown
        (and snapshotted) instead of being marked delivered unseen — or, worse,
        re-seeding a scoped turn that excludes it again, forever."""
        run = await database_sync_to_async(SubAgentRun.objects.create)(
            thread=self.thread, user=self.user, prompt="scan",
            status=SubAgentRun.Status.COMPLETED, result="LATE_FINDINGS",
        )
        await self._make_msg_at("old prompt", 0, metadata={"loop_run": True})
        await self._make_msg_at("PRIOR_REPLY", 1, role="assistant")
        await self._make_msg_at(
            "[Sub-agent result: late]\nLATE_FINDINGS", 2,
            metadata={"source": "subagent", "subagent_run_id": str(run.id)},
            hidden=True,
        )
        await self._make_msg_at("CURRENT_PROMPT", 10, metadata={"loop_run": True})

        result = await self.consumer._load_history(
            self.thread, scope_to_current_pass=True
        )
        blob = " ".join(m["content"] for m in result["messages"])
        self.assertIn("LATE_FINDINGS", blob)
        self.assertIn("CURRENT_PROMPT", blob)
        self.assertNotIn("PRIOR_REPLY", blob)
        self.assertNotIn("old prompt", blob)
        self.assertEqual(result["meta"]["subagent_run_ids"], [str(run.id)])

    async def test_scope_fallback_when_no_marker(self):
        """No loop_run marker → behave exactly like an unscoped load."""
        await self._make_msg_at("Just a message", 0)
        await self._make_msg_at("Another one", 1, role="assistant")
        scoped = await self.consumer._load_history(
            self.thread, scope_to_current_pass=True
        )
        self.assertEqual(scoped["meta"]["total_messages"], 2)
        self.assertEqual(scoped["meta"]["included_messages"], 2)

    async def test_is_fresh_loop_thread_detection(self):
        # Fresh loop → True
        fresh = await database_sync_to_async(Loop.objects.create)(
            thread=self.thread, created_by=self.user, prompt="p",
            history_mode=Loop.HistoryMode.FRESH,
            cadence_kind=Loop.Cadence.INTERVAL, interval_seconds=3600,
            next_run=self.base,
        )
        self.assertTrue(await self.consumer._is_fresh_loop_thread(self.thread.id))

        # Conversational loop → False
        await database_sync_to_async(
            Loop.objects.filter(pk=fresh.pk).update
        )(history_mode=Loop.HistoryMode.CONVERSATIONAL)
        self.assertFalse(await self.consumer._is_fresh_loop_thread(self.thread.id))

        # Non-loop thread → False
        other = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user
        )
        self.assertFalse(await self.consumer._is_fresh_loop_thread(other.id))
