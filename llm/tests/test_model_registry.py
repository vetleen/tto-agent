"""Tests for the curated model registry."""

from decimal import Decimal

from django.test import SimpleTestCase

from llm.model_registry import (
    TIER_CHEAP,
    TIER_FLAGSHIP,
    TIER_MID,
    TIER_STANDARD,
    ModelInfo,
    canonical_model_id,
    get_model_info,
    get_model_tier,
    get_models_by_tier,
    get_models_for_slot,
    get_models_with_min_stars,
    get_registered_model_ids,
    is_model_valid_for_slot,
    normalize_model_ids,
)


EXPECTED_IDS = [
    "openai/gpt-6-astra",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.4-nano",
    "anthropic/claude-fable-5-1",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4-5",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.7-flash",
    "gemini/gemini-3.5-flash-lite",
]


class RegistryTests(SimpleTestCase):
    def test_curated_set_is_exact(self):
        self.assertEqual(get_registered_model_ids(), EXPECTED_IDS)

    def test_every_model_has_complete_core_metadata(self):
        for model_id in EXPECTED_IDS:
            with self.subTest(model=model_id):
                info = get_model_info(model_id)
                self.assertIsNotNone(info)
                self.assertIn(
                    info.tier,
                    (TIER_CHEAP, TIER_MID, TIER_STANDARD, TIER_FLAGSHIP),
                )
                # Every curated model must land in at least one category.
                self.assertTrue(info.tiers)
                self.assertGreater(info.context_window, 0)
                self.assertGreater(info.max_output_tokens, 0)
                self.assertIn("text", info.input_modalities)
                self.assertTrue(info.supports_vision)
                self.assertIsNotNone(info.input_price)
                self.assertIsNotNone(info.output_price)
                self.assertIn(info.default_reasoning_level, info.reasoning_levels)

    def test_exact_reasoning_capabilities(self):
        expected = {
            "openai/gpt-6-astra": (("low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.6-sol": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.6-terra": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.6-luna": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.4-nano": (("none", "low", "medium", "high", "xhigh"), "none"),
            "anthropic/claude-fable-5-1": (("low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-fable-5": (("low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-opus-5": (("off", "low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-opus-4-8": (("off", "low", "medium", "high", "max"), "off"),
            "anthropic/claude-opus-4-6": (("off", "low", "medium", "high", "max"), "off"),
            "anthropic/claude-sonnet-5": (("off", "low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-haiku-4-5": (("off", "low", "medium", "high"), "off"),
            "gemini/gemini-3.1-pro-preview": (("low", "medium", "high"), "high"),
            "gemini/gemini-3.8-flash": (("low", "medium", "high"), "medium"),
            "gemini/gemini-3.7-flash": (("low", "medium", "high"), "medium"),
            "gemini/gemini-3.5-flash-lite": (("minimal", "low", "medium", "high"), "minimal"),
        }
        for model_id, (levels, default) in expected.items():
            with self.subTest(model=model_id):
                info = get_model_info(model_id)
                self.assertEqual(info.reasoning_levels, levels)
                self.assertEqual(info.default_reasoning_level, default)

    def test_context_and_output_limits(self):
        self.assertEqual(get_model_info("gpt-5.6-sol").context_window, 1_050_000)
        self.assertEqual(get_model_info("gpt-5.6-sol").max_output_tokens, 128_000)
        self.assertEqual(get_model_info("gpt-5.4-nano").context_window, 400_000)
        self.assertEqual(get_model_info("claude-haiku-4-5").max_output_tokens, 64_000)
        self.assertEqual(get_model_info("gemini-3.7-flash").context_window, 1_048_576)

    def test_flagship_pricing_is_present(self):
        info = get_model_info("openai/gpt-6-astra")
        self.assertEqual(info.input_price, Decimal("10.00"))
        self.assertEqual(info.cache_write_price, Decimal("12.50"))
        self.assertEqual(info.output_price, Decimal("50.00"))
        self.assertEqual(info.long_context_threshold, 272_000)
        self.assertEqual(info.long_context_output_price, Decimal("75.00"))

    def test_fable_5_1_metadata(self):
        info = get_model_info("anthropic/claude-fable-5-1")
        self.assertEqual(info.api_model, "claude-fable-5-1")
        self.assertEqual(info.input_price, Decimal("10.00"))
        self.assertEqual(info.output_price, Decimal("50.00"))
        # Fable 5.1 cache reads are 0.025x the input price, not the usual 0.1x.
        self.assertEqual(info.cached_input_price, Decimal("0.25"))
        self.assertEqual(info.cache_write_price, Decimal("12.50"))
        self.assertEqual(info.cache_write_1h_price, Decimal("20.00"))
        self.assertEqual(info.context_window, 1_000_000)
        self.assertEqual(info.max_output_tokens, 128_000)
        self.assertEqual(info.thinking_mode, "adaptive")

    def test_gemini_3_8_flash_matches_3_7_pricing(self):
        new = get_model_info("gemini/gemini-3.8-flash")
        old = get_model_info("gemini/gemini-3.7-flash")
        self.assertEqual(new.api_model, "gemini-3.8-flash")
        self.assertEqual(new.input_price, old.input_price)
        self.assertEqual(new.cached_input_price, old.cached_input_price)
        self.assertEqual(new.output_price, old.output_price)
        # Same intro pricing, including the 2027-01-01 increase.
        self.assertEqual(new.price_changes, old.price_changes)
        self.assertEqual(new.context_window, 1_048_576)

    def test_anthropic_thinking_transport_is_explicit(self):
        for model_id in EXPECTED_IDS:
            info = get_model_info(model_id)
            if info.provider == "anthropic":
                self.assertIn(info.thinking_mode, ("adaptive", "extended"))
        self.assertEqual(get_model_info("claude-opus-4-6").thinking_mode, "adaptive")
        self.assertEqual(get_model_info("claude-haiku-4-5").thinking_mode, "extended")

    def test_every_model_has_manual_stars(self):
        # Curated models must not rely on the price-derived fallback, which
        # promo pricing can silently demote (that's how Sol lost a star once).
        for model_id in EXPECTED_IDS:
            with self.subTest(model=model_id):
                self.assertIn(get_model_info(model_id).stars, (1, 2, 3, 4, 5))

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_model_info("unknown/model"))


class ReplacementTests(SimpleTestCase):
    def test_retired_models_map_to_approved_replacements(self):
        expected = {
            "openai/gpt-5.5": "openai/gpt-5.6-sol",
            "openai/gpt-5.4": "openai/gpt-5.6-terra",
            "openai/gpt-5.4-mini": "openai/gpt-5.6-luna",
            "anthropic/claude-opus-4-7": "anthropic/claude-opus-5",
            "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-5",
            "gemini/gemini-3.5-flash": "gemini/gemini-3.7-flash",
            "gemini/gemini-3.1-flash-lite": "gemini/gemini-3.5-flash-lite",
        }
        for retired, replacement in expected.items():
            with self.subTest(model=retired):
                self.assertEqual(canonical_model_id(retired), replacement)
                self.assertEqual(get_model_info(retired), get_model_info(replacement))

    def test_bare_names_are_normalized(self):
        self.assertEqual(canonical_model_id("gpt-5.6-terra"), "openai/gpt-5.6-terra")
        self.assertEqual(canonical_model_id("claude-opus-5"), "anthropic/claude-opus-5")
        self.assertEqual(canonical_model_id("gemini-3.7-flash"), "gemini/gemini-3.7-flash")

    def test_list_normalization_deduplicates_replacements(self):
        self.assertEqual(
            normalize_model_ids(["openai/gpt-5.5", "openai/gpt-5.6-sol", "bogus"]),
            ["openai/gpt-5.6-sol"],
        )


class TierTests(SimpleTestCase):
    def test_capability_stars_price_fallback(self):
        def unstarred(price, flagship=False):
            return ModelInfo(
                display_name="X", provider="openai", api_model="x",
                flagship=flagship,
                input_price=Decimal(price) if price is not None else None,
            )

        self.assertEqual(unstarred("0.50").capability_stars, 1)
        self.assertEqual(unstarred("0.51").capability_stars, 2)
        self.assertEqual(unstarred("1.50").capability_stars, 2)
        self.assertEqual(unstarred("1.51").capability_stars, 3)
        self.assertEqual(unstarred("4.99").capability_stars, 3)
        self.assertEqual(unstarred("5.00").capability_stars, 4)
        self.assertEqual(unstarred(None).capability_stars, 3)
        # Flagship adds a star (a $10 unstarred flagship reads as 5).
        self.assertEqual(unstarred("10.00", flagship=True).capability_stars, 5)

    def test_manual_stars_beat_price(self):
        # Luna's curated 2 stars keep it mid-grade despite its promo price.
        luna = get_model_info("openai/gpt-5.6-luna")
        self.assertEqual(luna.capability_stars, 2)
        self.assertEqual(luna.tiers, frozenset({TIER_CHEAP, TIER_MID}))
        self.assertEqual(get_model_tier("openai/gpt-5.6-luna"), TIER_MID)

    def test_tier_sets(self):
        self.assertEqual(
            get_models_by_tier(TIER_CHEAP),
            ["openai/gpt-5.6-luna", "openai/gpt-5.4-nano", "gemini/gemini-3.5-flash-lite"],
        )
        self.assertEqual(
            get_models_by_tier(TIER_MID),
            [
                "openai/gpt-5.6-terra",
                "openai/gpt-5.6-luna",
                "anthropic/claude-sonnet-5",
                "anthropic/claude-haiku-4-5",
                "gemini/gemini-3.1-pro-preview",
                "gemini/gemini-3.8-flash",
                "gemini/gemini-3.7-flash",
            ],
        )
        self.assertEqual(
            get_models_by_tier(TIER_STANDARD),
            [
                "openai/gpt-5.6-sol",
                "anthropic/claude-opus-5",
                "anthropic/claude-opus-4-8",
                "anthropic/claude-opus-4-6",
            ],
        )
        self.assertEqual(
            get_models_by_tier(TIER_FLAGSHIP),
            [
                "openai/gpt-6-astra",
                "anthropic/claude-fable-5-1",
                "anthropic/claude-fable-5",
            ],
        )

    def test_slot_validation(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4-nano", "cheap"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-luna", "cheap"))
        # Luna's 2 stars now clear the mid slot's floor (overlap is intended).
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-luna", "mid"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4-nano", "mid"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-terra", "mid"))
        # Primary requires 3 stars: Sonnet-class models stay eligible.
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-terra", "primary"))
        self.assertTrue(is_model_valid_for_slot("anthropic/claude-sonnet-5", "primary"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-sol", "primary"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-6-astra", "primary"))
        self.assertTrue(is_model_valid_for_slot("anthropic/claude-fable-5-1", "primary"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.6-luna", "primary"))
        self.assertFalse(is_model_valid_for_slot("anthropic/claude-haiku-4-5", "primary"))

    def test_get_models_for_slot_canonicalizes_filter(self):
        allowed = ["openai/gpt-5.4-nano", "openai/gpt-5.4"]
        self.assertEqual(get_models_for_slot("cheap", allowed), ["openai/gpt-5.4-nano"])
        self.assertEqual(get_models_for_slot("primary", allowed), ["openai/gpt-5.6-terra"])

    def test_models_with_min_stars(self):
        self.assertEqual(len(get_models_with_min_stars(1)), len(EXPECTED_IDS))
        two_up = get_models_with_min_stars(2)
        self.assertNotIn("openai/gpt-5.4-nano", two_up)
        self.assertNotIn("gemini/gemini-3.5-flash-lite", two_up)
        self.assertIn("openai/gpt-5.6-luna", two_up)
        three_up = get_models_with_min_stars(3)
        self.assertIn("anthropic/claude-sonnet-5", three_up)
        self.assertNotIn("anthropic/claude-haiku-4-5", three_up)
        self.assertEqual(
            get_models_with_min_stars(5),
            [
                "openai/gpt-6-astra",
                "anthropic/claude-fable-5-1",
                "anthropic/claude-fable-5",
            ],
        )

    def test_only_astra_and_fables_are_flagships(self):
        flagships = [
            model_id
            for model_id in EXPECTED_IDS
            if get_model_info(model_id).flagship
        ]
        self.assertEqual(
            flagships,
            [
                "openai/gpt-6-astra",
                "anthropic/claude-fable-5-1",
                "anthropic/claude-fable-5",
            ],
        )
