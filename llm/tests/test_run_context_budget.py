"""Tests for RunContext native-asset budget — pool + per-pathway skill cap."""

from concurrent.futures import ThreadPoolExecutor

from django.test import SimpleTestCase, override_settings

from llm.types.context import (
    PATHWAY_ATTACHMENT,
    PATHWAY_DATAROOM,
    PATHWAY_SKILL,
    RunContext,
    native_asset_pool_bytes,
)


@override_settings(NATIVE_ASSET_BUDGET_B64_BYTES=1000, NATIVE_ASSET_SKILL_FRACTION=0.5)
class NativeAssetBudgetTests(SimpleTestCase):
    def test_within_budget_appends_and_returns_true(self):
        ctx = RunContext.create()
        item = {"kind": "image", "b64": "A" * 100}
        self.assertTrue(ctx.try_add_native_asset(item, pathway=PATHWAY_DATAROOM))
        self.assertEqual(ctx.pending_native_assets, [item])
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 900)

    def test_over_budget_rejects_without_append(self):
        ctx = RunContext.create()
        ctx._b64_used_by_pathway[PATHWAY_DATAROOM] = 990
        self.assertFalse(ctx.try_add_native_asset({"b64": "A" * 11}, pathway=PATHWAY_DATAROOM))
        self.assertEqual(ctx.pending_native_assets, [])
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 10)

    def test_exact_fit_accepted(self):
        ctx = RunContext.create()
        ctx._b64_used_by_pathway[PATHWAY_DATAROOM] = 990
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 10}, pathway=PATHWAY_DATAROOM))
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 0)

    def test_budget_survives_pipeline_drain(self):
        """Clearing the pending list (as the pipeline drain does each iteration)
        must NOT reset the byte counter."""
        ctx = RunContext.create()
        ctx.try_add_native_asset({"b64": "A" * 500}, pathway=PATHWAY_DATAROOM)
        ctx.pending_native_assets.clear()
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 500)

    def test_missing_b64_counts_as_zero(self):
        ctx = RunContext.create()
        self.assertTrue(ctx.try_add_native_asset({"kind": "image"}, pathway=PATHWAY_DATAROOM))
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 1000)

    def test_item_is_tagged_with_pathway(self):
        ctx = RunContext.create()
        item = {"b64": "A" * 10}
        ctx.try_add_native_asset(item, pathway=PATHWAY_SKILL)
        self.assertEqual(item["_pathway"], PATHWAY_SKILL)

    def test_skill_capped_at_fraction_of_pool(self):
        ctx = RunContext.create()
        # 50% of 1000 = 500 is the skill ceiling.
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_SKILL), 500)
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 500}, pathway=PATHWAY_SKILL))
        self.assertFalse(ctx.try_add_native_asset({"b64": "A"}, pathway=PATHWAY_SKILL))
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_SKILL), 0)

    def test_skill_reserve_independent_of_nonskill(self):
        """Non-skill filling the pool does not shrink the skill's 50% ceiling —
        skills draw from their own reserve (send-time pruning trims the total)."""
        ctx = RunContext.create()
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 1000}, pathway=PATHWAY_ATTACHMENT))
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_ATTACHMENT), 0)
        # Skill still has its full 500 even though non-skill filled the pool.
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_SKILL), 500)
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 500}, pathway=PATHWAY_SKILL))

    def test_nonskill_pathways_share_the_pool(self):
        ctx = RunContext.create()
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 600}, pathway=PATHWAY_ATTACHMENT))
        # dataroom sees the pool minus the attachment's 600.
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 400)
        self.assertFalse(ctx.try_add_native_asset({"b64": "A" * 401}, pathway=PATHWAY_DATAROOM))
        self.assertTrue(ctx.try_add_native_asset({"b64": "A" * 400}, pathway=PATHWAY_DATAROOM))

    def test_concurrent_reserves_never_overshoot(self):
        """Tools run in a ThreadPoolExecutor; reserves must be atomic."""
        ctx = RunContext.create()
        item_size = 250  # pool / 4
        payload = "A" * item_size

        def attempt(_):
            return ctx.try_add_native_asset({"b64": payload}, pathway=PATHWAY_DATAROOM)

        with ThreadPoolExecutor(max_workers=8) as pool:
            wins = sum(pool.map(attempt, range(16)))

        self.assertEqual(wins, 4)
        self.assertEqual(len(ctx.pending_native_assets), 4)
        self.assertEqual(ctx.native_asset_budget_remaining(PATHWAY_DATAROOM), 0)

    def test_pool_reads_from_settings(self):
        self.assertEqual(native_asset_pool_bytes(), 1000)
