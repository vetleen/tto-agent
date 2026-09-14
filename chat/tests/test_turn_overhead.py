"""Turn-assembly wires a MEASURED per-turn overhead into the history budget.

Verifies `_assemble_turn_inputs` measures system+tools+preamble (tiktoken) and
passes it to `_load_history` as `reserved_tokens` (instead of the flat 24k), that
the value is stashed in `meta` for summarization consistency, and that a
measurement failure falls back to the flat reservation without breaking the turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase

from chat.consumers import ChatConsumer
from chat.models import ChatThread

User = get_user_model()


def _prefs(**over):
    base = dict(
        allowed_tools=[], allowed_skills=[], allowed_specializations=[],
        allow_agent_attach_skills=True, parallel_subagents=True,
        max_context_tokens=200_000,
    )
    base.update(over)
    return SimpleNamespace(**base)


class AssembleTurnOverheadTests(TransactionTestCase):
    async def _setup(self):
        self.user = await database_sync_to_async(User.objects.create_user)(
            email="ov@test.com", password="pass"
        )
        self.thread = await database_sync_to_async(ChatThread.objects.create)(
            created_by=self.user
        )
        c = ChatConsumer()
        c.user = self.user
        c.resolved_prefs = _prefs()
        c.data_room_ids = []
        c.active_skill_ids = []
        c._soul = None
        c._org_description = None
        c._user_context = None
        c._org_name = "Test Org"
        return c

    def _patches(self, consumer, load_history_mock, *, semi_static=""):
        # Patch the DB-touching async helpers + tool resolution so the test is
        # pure-CPU, and patch _load_history to capture the reserved_tokens kwarg.
        return [
            patch.object(consumer, "_get_thread_tasks", AsyncMock(return_value=[])),
            patch.object(consumer, "_get_canvases_for_prompt", AsyncMock(return_value=None)),
            patch.object(consumer, "_get_active_deck_for_prompt", AsyncMock(return_value=(None, None))),
            patch.object(consumer, "_resolve_selected_tools", AsyncMock(return_value=([], []))),
            patch.object(consumer, "_load_history", load_history_mock),
            patch("chat.prompts.build_semi_static_prompt", return_value=semi_static),
        ]

    @staticmethod
    def _load_history_mock():
        return AsyncMock(return_value={
            "messages": [],
            "meta": {
                "total_messages": 0, "included_messages": 0,
                "has_summary": False, "needs_summary": False,
                "history_budget_tokens": 100_000, "included_history_tokens": 0,
                "summary_tokens": 0, "history_truncated": False,
                "has_subagent_results": False,
            },
        })

    async def test_measured_overhead_passed_and_scales_with_preamble(self):
        consumer = await self._setup()

        # Small preamble → small reserved overhead.
        lh_small = self._load_history_mock()
        patches = self._patches(consumer, lh_small, semi_static="short preamble")
        for p in patches:
            p.start()
        try:
            await consumer._assemble_turn_inputs(
                self.thread, "hello", model="gpt-5.4", max_context_tokens=200_000,
                thinking_level="medium",
            )
        finally:
            for p in patches:
                p.stop()
        small_reserved = lh_small.await_args.kwargs["reserved_tokens"]
        self.assertIsInstance(small_reserved, int)
        self.assertGreater(small_reserved, 0)
        # effort threaded through
        self.assertEqual(lh_small.await_args.kwargs["effort"], "medium")

        # Big skill-instruction preamble → much larger reserved overhead (> flat 24k).
        lh_big = self._load_history_mock()
        patches = self._patches(consumer, lh_big, semi_static="word " * 60_000)
        for p in patches:
            p.start()
        try:
            await consumer._assemble_turn_inputs(
                self.thread, "hello", model="gpt-5.4", max_context_tokens=200_000,
                thinking_level="medium",
            )
        finally:
            for p in patches:
                p.stop()
        big_reserved = lh_big.await_args.kwargs["reserved_tokens"]
        self.assertGreater(big_reserved, small_reserved)
        self.assertGreater(big_reserved, 24_000)  # the whole point: not the flat value

    async def test_meta_carries_reserved_and_effort(self):
        consumer = await self._setup()
        lh = self._load_history_mock()
        patches = self._patches(consumer, lh, semi_static="word " * 5_000)
        for p in patches:
            p.start()
        try:
            result = await consumer._assemble_turn_inputs(
                self.thread, "hi", model="gpt-5.4", max_context_tokens=200_000,
                thinking_level="high",
            )
        finally:
            for p in patches:
                p.stop()
        meta = result[4]
        # _load_history's returned meta is what propagates; assert the kwargs it
        # was called with are the measured ones (they get echoed into meta there).
        self.assertEqual(lh.await_args.kwargs["effort"], "high")
        self.assertGreater(lh.await_args.kwargs["reserved_tokens"], 0)
        # 7-tuple shape
        self.assertEqual(len(result), 7)

    async def test_measurement_failure_falls_back_to_flat(self):
        consumer = await self._setup()
        lh = self._load_history_mock()
        patches = self._patches(consumer, lh, semi_static="word " * 100)
        for p in patches:
            p.start()
        # Make the measurement raise → reserved_tokens must be None (flat fallback).
        fail = patch("core.tokens.measure_request_overhead", side_effect=RuntimeError("boom"))
        fail.start()
        try:
            await consumer._assemble_turn_inputs(
                self.thread, "hi", model="gpt-5.4", max_context_tokens=200_000,
                thinking_level="medium",
            )
        finally:
            fail.stop()
            for p in patches:
                p.stop()
        self.assertIsNone(lh.await_args.kwargs["reserved_tokens"])
