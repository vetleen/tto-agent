"""Tests for the curated model registry."""

from decimal import Decimal

from django.test import SimpleTestCase

from llm.model_registry import (
    TIER_CHEAP,
    TIER_MID,
    TIER_PREMIUM,
    TIER_STANDARD,
    canonical_model_id,
    get_model_info,
    get_model_tier,
    get_models_at_or_above_tier,
    get_models_by_tier,
    get_models_for_slot,
    get_performance_tier,
    get_registered_model_ids,
    is_model_valid_for_slot,
    normalize_model_ids,
)


EXPECTED_IDS = [
    "openai/gpt-5.6-sol",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.4-nano",
    "anthropic/claude-fable-5",
    "anthropic/claude-opus-5",
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4-5",
    "gemini/gemini-3.1-pro-preview",
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
                    (TIER_CHEAP, TIER_MID, TIER_STANDARD, TIER_PREMIUM),
                )
                self.assertGreater(info.context_window, 0)
                self.assertGreater(info.max_output_tokens, 0)
                self.assertIn("text", info.input_modalities)
                self.assertTrue(info.supports_vision)
                self.assertIsNotNone(info.input_price)
                self.assertIsNotNone(info.output_price)
                self.assertIn(info.default_reasoning_level, info.reasoning_levels)

    def test_exact_reasoning_capabilities(self):
        expected = {
            "openai/gpt-5.6-sol": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.6-terra": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.6-luna": (("none", "low", "medium", "high", "xhigh", "max"), "medium"),
            "openai/gpt-5.4-nano": (("none", "low", "medium", "high", "xhigh"), "none"),
            "anthropic/claude-fable-5": (("low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-opus-5": (("off", "low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-opus-4-6": (("off", "low", "medium", "high", "max"), "off"),
            "anthropic/claude-sonnet-5": (("off", "low", "medium", "high", "xhigh", "max"), "high"),
            "anthropic/claude-haiku-4-5": (("off", "low", "medium", "high"), "off"),
            "gemini/gemini-3.1-pro-preview": (("low", "medium", "high"), "high"),
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
        info = get_model_info("openai/gpt-5.6-sol")
        self.assertEqual(info.input_price, Decimal("5.00"))
        self.assertEqual(info.cache_write_price, Decimal("6.25"))
        self.assertEqual(info.output_price, Decimal("30.00"))
        self.assertEqual(info.long_context_threshold, 272_000)
        self.assertEqual(info.long_context_output_price, Decimal("45.00"))

    def test_anthropic_thinking_transport_is_explicit(self):
        for model_id in EXPECTED_IDS:
            info = get_model_info(model_id)
            if info.provider == "anthropic":
                self.assertIn(info.thinking_mode, ("adaptive", "extended"))
        self.assertEqual(get_model_info("claude-opus-4-6").thinking_mode, "adaptive")
        self.assertEqual(get_model_info("claude-haiku-4-5").thinking_mode, "extended")

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_model_info("unknown/model"))


class ReplacementTests(SimpleTestCase):
    def test_retired_models_map_to_approved_replacements(self):
        expected = {
            "openai/gpt-5.5": "openai/gpt-5.6-sol",
            "openai/gpt-5.4": "openai/gpt-5.6-terra",
            "openai/gpt-5.4-mini": "openai/gpt-5.6-luna",
            "anthropic/claude-opus-4-8": "anthropic/claude-opus-5",
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
    def test_input_price_intervals_define_tiers(self):
        self.assertEqual(get_performance_tier(Decimal("0.50")), TIER_CHEAP)
        self.assertEqual(get_performance_tier(Decimal("0.51")), TIER_MID)
        self.assertEqual(get_performance_tier(Decimal("1.50")), TIER_MID)
        self.assertEqual(get_performance_tier(Decimal("1.51")), TIER_STANDARD)
        self.assertEqual(get_performance_tier(Decimal("4.99")), TIER_STANDARD)
        self.assertEqual(get_performance_tier(Decimal("5.00")), TIER_PREMIUM)

    def test_tier_sets(self):
        self.assertEqual(
            get_models_by_tier(TIER_CHEAP),
            ["openai/gpt-5.4-nano", "gemini/gemini-3.5-flash-lite"],
        )
        self.assertEqual(
            get_models_by_tier(TIER_MID),
            ["openai/gpt-5.6-luna", "anthropic/claude-haiku-4-5", "gemini/gemini-3.7-flash"],
        )
        self.assertIn("openai/gpt-5.6-terra", get_models_by_tier(TIER_STANDARD))
        self.assertEqual(
            get_models_by_tier(TIER_PREMIUM),
            [
                "openai/gpt-5.6-sol",
                "anthropic/claude-fable-5",
                "anthropic/claude-opus-5",
                "anthropic/claude-opus-4-6",
            ],
        )

    def test_slot_validation(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4-nano", "cheap"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.6-luna", "cheap"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-luna", "mid"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-terra", "mid"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-terra", "primary"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.6-sol", "primary"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.6-luna", "primary"))

    def test_get_models_for_slot_canonicalizes_filter(self):
        allowed = ["openai/gpt-5.4-nano", "openai/gpt-5.4"]
        self.assertEqual(get_models_for_slot("cheap", allowed), ["openai/gpt-5.4-nano"])
        self.assertEqual(get_models_for_slot("primary", allowed), ["openai/gpt-5.6-terra"])

    def test_at_or_above_tier(self):
        self.assertEqual(len(get_models_at_or_above_tier(TIER_CHEAP)), len(EXPECTED_IDS))
        self.assertNotIn("openai/gpt-5.4-nano", get_models_at_or_above_tier(TIER_MID))
        for model_id in get_models_at_or_above_tier(TIER_STANDARD):
            self.assertIn(get_model_tier(model_id), (TIER_STANDARD, TIER_PREMIUM))

    def test_only_sol_and_fable_are_flagships(self):
        flagships = [
            model_id
            for model_id in EXPECTED_IDS
            if get_model_info(model_id).flagship
        ]
        self.assertEqual(
            flagships,
            ["openai/gpt-5.6-sol", "anthropic/claude-fable-5"],
        )
