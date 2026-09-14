"""Tests for context pruning: tool-result stubbing between turns + the rolling
summary excluding tool noise (Phase 5 of the context overhaul)."""

from __future__ import annotations

from types import SimpleNamespace

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TestCase, TransactionTestCase, override_settings

from chat.consumers import ChatConsumer, _tool_call_meta_map, _tool_stub_boundary
from chat.models import ChatMessage, ChatThread
from chat.tool_stub import build_tool_result_stub
from core.tokens import count_tokens

User = get_user_model()


class BuildToolResultStubTests(TestCase):
    def test_includes_tool_name_and_args(self):
        stub = build_tool_result_stub("document_view_native", {"doc_indices": [3], "pages": "3-5"})
        self.assertIn("document_view_native", stub)
        self.assertIn("doc_indices=[3]", stub)
        self.assertIn("pages=3-5", stub)
        self.assertIn("cleared", stub.lower())

    def test_drops_reason_field(self):
        stub = build_tool_result_stub("web_search", {"reason": "because", "query": "cats"})
        self.assertNotIn("because", stub)
        self.assertIn("query=cats", stub)

    def test_handles_missing_args(self):
        stub = build_tool_result_stub("some_tool", None)
        self.assertIn("some_tool", stub)

    def test_truncates_huge_values(self):
        stub = build_tool_result_stub("t", {"blob": "x" * 5000})
        self.assertLess(len(stub), 400)


class ToolStubBoundaryTests(TestCase):
    def _msgs(self, roles):
        return [SimpleNamespace(role=r, metadata={}) for r in roles]

    def test_disabled_when_negative(self):
        msgs = self._msgs(["user", "tool", "user", "tool", "user"])
        self.assertEqual(_tool_stub_boundary(msgs, -1), 0)

    def test_zero_keeps_only_after_last_user(self):
        # roles: u t a u t  -> last user at index 3; keep raw idx >= 4
        msgs = self._msgs(["user", "tool", "assistant", "user", "tool"])
        self.assertEqual(_tool_stub_boundary(msgs, 0), 4)

    def test_two_turns_kept_raw(self):
        # 3 user turns at indices 0,3,6; keep last 2 -> boundary at index 3
        msgs = self._msgs(
            ["user", "assistant", "tool", "user", "assistant", "tool", "user", "tool"]
        )
        self.assertEqual(_tool_stub_boundary(msgs, 2), 3)

    def test_no_stub_when_few_turns(self):
        msgs = self._msgs(["user", "tool", "user", "tool"])
        self.assertEqual(_tool_stub_boundary(msgs, 2), 0)


class ToolCallMetaMapTests(TestCase):
    def test_maps_call_id_to_name_and_args(self):
        msgs = [
            SimpleNamespace(
                role="assistant",
                metadata={"tool_calls": [{"id": "c1", "name": "web_search", "arguments": {"query": "x"}}]},
            ),
            SimpleNamespace(role="tool", metadata={}),
        ]
        out = _tool_call_meta_map(msgs)
        self.assertEqual(out["c1"], ("web_search", {"query": "x"}))


class LoadHistoryStubbingTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="stub@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()

    @database_sync_to_async
    def _msg(self, content, role="user", tool_call_id=None, tool_calls=None):
        metadata = {"tool_calls": tool_calls} if tool_calls else {}
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            metadata=metadata,
            token_count=count_tokens(content),
        )

    async def _build_three_turns(self):
        # Turn 1 (oldest): a tool call + result that should age out.
        await self._msg("Question one", role="user")
        await self._msg(
            "", role="assistant",
            tool_calls=[{"id": "call-old", "name": "document_view_native", "arguments": {"doc_indices": [3]}}],
        )
        await self._msg("OLD_TOOL_RESULT_DATA", role="tool", tool_call_id="call-old")
        # Turn 2.
        await self._msg("Question two", role="user")
        await self._msg("Answer two", role="assistant")
        # Turn 3 (newest): a tool call + result that must stay raw.
        await self._msg("Question three", role="user")
        await self._msg(
            "", role="assistant",
            tool_calls=[{"id": "call-new", "name": "web_search", "arguments": {"query": "cats"}}],
        )
        await self._msg("RECENT_TOOL_RESULT_DATA", role="tool", tool_call_id="call-new")

    @override_settings(CONTEXT_RAW_TOOL_TURNS=2)
    async def test_old_tool_result_stubbed_recent_kept(self):
        await self._build_three_turns()
        messages = (await self.consumer._load_history(self.thread))["messages"]
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        old, new = tool_msgs[0], tool_msgs[1]
        # Old result collapsed to a precise stub; recent result untouched.
        self.assertNotIn("OLD_TOOL_RESULT_DATA", old["content"])
        self.assertIn("document_view_native", old["content"])
        self.assertEqual(old["tool_call_id"], "call-old")  # preserved for pairing
        self.assertIn("RECENT_TOOL_RESULT_DATA", new["content"])

    @override_settings(CONTEXT_RAW_TOOL_TURNS=-1)
    async def test_stubbing_disabled(self):
        await self._build_three_turns()
        messages = (await self.consumer._load_history(self.thread))["messages"]
        tool_contents = " ".join(m["content"] for m in messages if m["role"] == "tool")
        self.assertIn("OLD_TOOL_RESULT_DATA", tool_contents)
        self.assertIn("RECENT_TOOL_RESULT_DATA", tool_contents)

    @override_settings(CONTEXT_RAW_TOOL_TURNS=2)
    async def test_stubbed_tool_result_not_orphan_stripped(self):
        """A stubbed tool result keeps its tool_call_id, so the orphan-strip must
        keep it (its assistant tool_calls entry is still present)."""
        await self._build_three_turns()
        messages = (await self.consumer._load_history(self.thread))["messages"]
        call_ids = {m.get("tool_call_id") for m in messages if m["role"] == "tool"}
        self.assertIn("call-old", call_ids)


class SummariseExcludesToolNoiseTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="sum@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user)
        self.consumer = ChatConsumer()

    @database_sync_to_async
    def _msg(self, content, role="user", tool_call_id=None, tool_calls=None, hidden=False):
        metadata = {"tool_calls": tool_calls} if tool_calls else {}
        return ChatMessage.objects.create(
            thread=self.thread,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            metadata=metadata,
            is_hidden_from_user=hidden,
            token_count=count_tokens(content),
        )

    async def test_tool_messages_excluded_from_summary(self):
        # A large narrative pushes older messages out of the budget window so
        # they become summarisation candidates; the tool noise among them must
        # be filtered out of the returned list.
        big = "word " * 6000
        await self._msg("narrative one " + big, role="user")
        await self._msg(
            "", role="assistant", hidden=True,
            tool_calls=[{"id": "c1", "name": "web_search", "arguments": {"query": "x"}}],
        )
        await self._msg("TOOL_NOISE_RESULT", role="tool", tool_call_id="c1")
        await self._msg("narrative two " + big, role="assistant")
        for i in range(4):
            await self._msg(f"filler {i} " + big, role="user")

        to_summarise = await self.consumer._get_messages_to_summarise(
            self.thread, model="gpt-5.4",
        )
        roles = [m.role for m in to_summarise]
        self.assertNotIn("tool", roles)
        # The hidden tool-call assistant message is excluded too.
        for m in to_summarise:
            if m.role == "assistant":
                self.assertFalse(
                    m.is_hidden_from_user and (m.metadata or {}).get("tool_calls")
                )
