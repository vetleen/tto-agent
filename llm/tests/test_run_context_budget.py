"""Tests for RunContext.try_add_native_asset — the per-run byte budget."""

from concurrent.futures import ThreadPoolExecutor

from django.test import SimpleTestCase

from llm.types.context import NATIVE_ASSET_BUDGET_B64_CHARS, RunContext


class NativeAssetBudgetTests(SimpleTestCase):
    def test_within_budget_appends_and_returns_true(self):
        ctx = RunContext.create()
        item = {"kind": "image", "b64": "A" * 100}
        self.assertTrue(ctx.try_add_native_asset(item))
        self.assertEqual(ctx.pending_native_assets, [item])
        self.assertEqual(
            ctx.native_asset_budget_remaining(), NATIVE_ASSET_BUDGET_B64_CHARS - 100,
        )

    def test_over_budget_rejects_without_append(self):
        ctx = RunContext.create()
        ctx._native_asset_b64_used = NATIVE_ASSET_BUDGET_B64_CHARS - 10
        self.assertFalse(ctx.try_add_native_asset({"b64": "A" * 11}))
        self.assertEqual(ctx.pending_native_assets, [])
        # The failed reserve must not consume budget
        self.assertEqual(ctx.native_asset_budget_remaining(), 10)

    def test_exact_fit_accepted(self):
        ctx = RunContext.create()
        ctx._native_asset_b64_used = NATIVE_ASSET_BUDGET_B64_CHARS - 10
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 10}))
        self.assertEqual(ctx.native_asset_budget_remaining(), 0)

    def test_budget_survives_pipeline_drain(self):
        """The counter is per-run cumulative — clearing the pending list (as the
        pipeline drain does each iteration) must NOT reset the budget."""
        ctx = RunContext.create()
        ctx.try_add_native_asset({"b64": "A" * 500})
        ctx.pending_native_assets.clear()
        self.assertEqual(
            ctx.native_asset_budget_remaining(), NATIVE_ASSET_BUDGET_B64_CHARS - 500,
        )

    def test_missing_b64_counts_as_zero(self):
        ctx = RunContext.create()
        self.assertTrue(ctx.try_add_native_asset({"kind": "image"}))
        self.assertEqual(ctx.native_asset_budget_remaining(), NATIVE_ASSET_BUDGET_B64_CHARS)

    def test_concurrent_reserves_never_overshoot(self):
        """Tools run in a ThreadPoolExecutor; reserves must be atomic."""
        ctx = RunContext.create()
        item_size = NATIVE_ASSET_BUDGET_B64_CHARS // 4
        payload = "A" * item_size

        def attempt(_):
            return ctx.try_add_native_asset({"b64": payload})

        with ThreadPoolExecutor(max_workers=8) as pool:
            wins = sum(pool.map(attempt, range(16)))

        self.assertEqual(wins, 4)  # only budget/item_size can fit
        self.assertEqual(len(ctx.pending_native_assets), 4)
        self.assertEqual(ctx.native_asset_budget_remaining(), 0)
