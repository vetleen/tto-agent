"""Tests for llm.model_registry — unified model metadata registry."""

from decimal import Decimal

from django.test import TestCase

from llm.model_registry import (
    SLOT_ALLOWED_TIERS,
    TIER_CHEAP,
    TIER_MID,
    TIER_STANDARD,
    ModelInfo,
    get_model_info,
    get_model_tier,
    get_models_at_or_above_tier,
    get_models_by_tier,
    get_models_for_slot,
    is_model_valid_for_slot,
)


class GetModelInfoTests(TestCase):

    def test_known_model_full_id(self):
        info = get_model_info("openai/gpt-5.4")
        self.assertIsNotNone(info)
        self.assertEqual(info.display_name, "GPT-5.4")
        self.assertEqual(info.provider, "openai")
        self.assertEqual(info.api_model, "gpt-5.4")

    def test_known_model_bare_name(self):
        info = get_model_info("claude-sonnet-4-6")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "anthropic")

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_model_info("unknown/model-xyz"))

    def test_anthropic_models_support_thinking(self):
        for model_id in [
            "anthropic/claude-opus-4-7",
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5",
        ]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info, f"{model_id} not found")
            self.assertTrue(info.supports_thinking, f"{model_id} should support thinking")

    def test_gpt5_5_supports_thinking(self):
        info = get_model_info("openai/gpt-5.5")
        self.assertTrue(info.supports_thinking)

    def test_gpt5_4_supports_thinking(self):
        info = get_model_info("openai/gpt-5.4")
        self.assertTrue(info.supports_thinking)

    def test_gpt5_mini_no_thinking(self):
        info = get_model_info("openai/gpt-5.4-mini")
        self.assertFalse(info.supports_thinking)

    def test_gemini_3_supports_thinking(self):
        for model_id in [
            "gemini/gemini-3.1-pro-preview",
            "gemini/gemini-3-flash-preview",
            "gemini/gemini-3.1-flash-lite-preview",
        ]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info, f"{model_id} not found")
            self.assertTrue(info.supports_thinking, f"{model_id} should support thinking")

    def test_gemini_2_no_thinking(self):
        info = get_model_info("gemini/gemini-2.5-flash")
        self.assertFalse(info.supports_thinking)

    def test_all_registered_models_have_vision(self):
        for model_id in [
            "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
            "anthropic/claude-opus-4-7", "anthropic/claude-opus-4-6",
            "gemini/gemini-2.5-flash", "gemini/gemini-3-flash-preview",
        ]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info)
            self.assertTrue(info.supports_vision, f"{model_id} should support vision")

    def test_gpt5_5_lookup_and_pricing(self):
        info = get_model_info("openai/gpt-5.5")
        self.assertIsNotNone(info)
        self.assertEqual(info.display_name, "GPT-5.5")
        self.assertEqual(info.provider, "openai")
        self.assertEqual(info.api_model, "gpt-5.5")
        self.assertEqual(info.context_window, 1_000_000)
        self.assertEqual(info.input_price, Decimal("5.00"))
        self.assertEqual(info.cached_input_price, Decimal("0.50"))
        self.assertEqual(info.output_price, Decimal("30.00"))

    def test_context_windows(self):
        self.assertEqual(get_model_info("openai/gpt-5.5").context_window, 1_000_000)
        self.assertEqual(get_model_info("openai/gpt-5.4").context_window, 1_000_000)
        self.assertEqual(get_model_info("openai/gpt-5.4-nano").context_window, 128_000)
        self.assertEqual(get_model_info("anthropic/claude-opus-4-7").context_window, 1_000_000)
        self.assertEqual(get_model_info("anthropic/claude-opus-4-6").context_window, 1_000_000)
        self.assertEqual(get_model_info("anthropic/claude-sonnet-4-6").context_window, 1_000_000)

    def test_pricing(self):
        info = get_model_info("openai/gpt-5.4")
        self.assertEqual(info.input_price, Decimal("1.75"))
        self.assertEqual(info.cached_input_price, Decimal("0.175"))
        self.assertEqual(info.output_price, Decimal("14.00"))

    def test_opus_47_pricing(self):
        info = get_model_info("anthropic/claude-opus-4-7")
        self.assertIsNotNone(info)
        self.assertEqual(info.input_price, Decimal("5.00"))
        self.assertEqual(info.cached_input_price, Decimal("0.50"))
        self.assertEqual(info.output_price, Decimal("25.00"))

    def test_bare_name_normalisation(self):
        """Bare model names resolve via prefix scanning."""
        info = get_model_info("gpt-5.4")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "openai")

        info = get_model_info("gemini-3-flash-preview")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "google_genai")

        info = get_model_info("claude-opus-4-7")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "anthropic")


class TierClassificationTests(TestCase):

    def test_all_models_have_a_tier(self):
        from llm.model_registry import _MODELS
        for model_id, info in _MODELS.items():
            self.assertIn(info.tier, (TIER_CHEAP, TIER_MID, TIER_STANDARD), f"{model_id} has invalid tier")

    def test_cheap_models(self):
        cheap = get_models_by_tier(TIER_CHEAP)
        self.assertIn("openai/gpt-5.4-nano", cheap)
        self.assertIn("gemini/gemini-2.5-flash-lite", cheap)
        self.assertIn("gemini/gemini-3.1-flash-lite-preview", cheap)
        self.assertNotIn("openai/gpt-5.4", cheap)

    def test_mid_models(self):
        mid = get_models_by_tier(TIER_MID)
        self.assertIn("openai/gpt-5.4-mini", mid)
        self.assertIn("anthropic/claude-haiku-4-5", mid)
        self.assertIn("gemini/gemini-2.5-flash", mid)
        self.assertNotIn("openai/gpt-5.4-nano", mid)

    def test_standard_models(self):
        standard = get_models_by_tier(TIER_STANDARD)
        self.assertIn("openai/gpt-5.5", standard)
        self.assertIn("openai/gpt-5.4", standard)
        self.assertIn("anthropic/claude-opus-4-7", standard)
        self.assertIn("anthropic/claude-opus-4-6", standard)
        self.assertIn("anthropic/claude-sonnet-4-6", standard)
        self.assertIn("gemini/gemini-2.5-pro", standard)
        self.assertNotIn("openai/gpt-5.4-mini", standard)

    def test_get_model_tier(self):
        self.assertEqual(get_model_tier("openai/gpt-5.4-nano"), TIER_CHEAP)
        self.assertEqual(get_model_tier("openai/gpt-5.4-mini"), TIER_MID)
        self.assertEqual(get_model_tier("openai/gpt-5.4"), TIER_STANDARD)
        self.assertIsNone(get_model_tier("unknown/model"))


class SlotValidationTests(TestCase):

    def test_cheap_slot_accepts_only_cheap(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4-nano", "cheap"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4-mini", "cheap"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4", "cheap"))

    def test_mid_slot_accepts_mid_and_standard(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4-mini", "mid"))
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4", "mid"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4-nano", "mid"))

    def test_primary_slot_accepts_only_standard(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4", "primary"))
        self.assertTrue(is_model_valid_for_slot("anthropic/claude-sonnet-4-6", "primary"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4-mini", "primary"))
        self.assertFalse(is_model_valid_for_slot("openai/gpt-5.4-nano", "primary"))

    def test_unknown_model_returns_false(self):
        self.assertFalse(is_model_valid_for_slot("unknown/model", "cheap"))
        self.assertFalse(is_model_valid_for_slot("unknown/model", "primary"))

    def test_unknown_slot_allows_any_model(self):
        self.assertTrue(is_model_valid_for_slot("openai/gpt-5.4", "nonexistent"))


class GetModelsForSlotTests(TestCase):

    def test_cheap_slot_returns_only_cheap(self):
        models = get_models_for_slot("cheap")
        for m in models:
            self.assertEqual(get_model_tier(m), TIER_CHEAP)

    def test_mid_slot_returns_mid_and_standard(self):
        models = get_models_for_slot("mid")
        for m in models:
            self.assertIn(get_model_tier(m), (TIER_MID, TIER_STANDARD))

    def test_primary_slot_returns_only_standard(self):
        models = get_models_for_slot("primary")
        for m in models:
            self.assertEqual(get_model_tier(m), TIER_STANDARD)

    def test_filtered_by_allowed_models(self):
        allowed = ["openai/gpt-5.4-nano", "openai/gpt-5.4"]
        cheap = get_models_for_slot("cheap", allowed)
        self.assertEqual(cheap, ["openai/gpt-5.4-nano"])
        primary = get_models_for_slot("primary", allowed)
        self.assertEqual(primary, ["openai/gpt-5.4"])

    def test_empty_allowed_returns_all_for_slot(self):
        models = get_models_for_slot("cheap", [])
        self.assertTrue(len(models) >= 3)

    def test_no_allowed_returns_all_for_slot(self):
        models = get_models_for_slot("cheap")
        self.assertTrue(len(models) >= 3)


class GetModelsAtOrAboveTierTests(TestCase):

    def test_at_or_above_cheap_returns_all(self):
        from llm.model_registry import _MODELS
        models = get_models_at_or_above_tier(TIER_CHEAP)
        self.assertEqual(len(models), len(_MODELS))

    def test_at_or_above_mid_excludes_cheap(self):
        models = get_models_at_or_above_tier(TIER_MID)
        for m in models:
            self.assertNotEqual(get_model_tier(m), TIER_CHEAP)

    def test_at_or_above_standard_excludes_cheap_and_mid(self):
        models = get_models_at_or_above_tier(TIER_STANDARD)
        for m in models:
            self.assertEqual(get_model_tier(m), TIER_STANDARD)
