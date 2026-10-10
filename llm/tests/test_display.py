"""Tests for model display and picker capabilities."""

from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase

from llm.model_registry import ModelInfo
from llm.display import (
    _PRICE_MIX_CACHE_READ,
    _PRICE_MIX_CACHE_WRITE,
    _PRICE_MIX_OUTPUT,
    _blended_price,
    get_capability_level,
    get_default_thinking_level,
    get_display_name,
    get_model_meta_tooltip,
    get_price_level,
    get_thinking_levels,
    supports_thinking,
    supports_vision,
)


class DisplayNameTests(SimpleTestCase):
    def test_registered_names(self):
        self.assertEqual(get_display_name("openai/gpt-6-astra"), "GPT-6 Astra")
        self.assertEqual(get_display_name("openai/gpt-5.6-sol"), "GPT-5.6 Sol")
        self.assertEqual(get_display_name("anthropic/claude-fable-5"), "Claude Fable 5")
        self.assertEqual(get_display_name("gemini/gemini-3.5-flash-lite"), "Gemini 3.5 Flash-Lite")

    def test_date_suffix_and_fallback(self):
        self.assertEqual(
            get_display_name("anthropic/claude-haiku-4-5-20251001"),
            "Claude Haiku 4.5",
        )
        self.assertEqual(get_display_name("custom/my-cool-model"), "My Cool Model")


class CapabilityTests(SimpleTestCase):
    def test_registered_models_support_thinking_and_vision(self):
        for model_id in (
            "openai/gpt-5.6-terra",
            "openai/gpt-5.4-nano",
            "anthropic/claude-fable-5",
            "anthropic/claude-haiku-5-5",
            "gemini/gemini-3.7-flash",
            "gemini/gemini-3.5-flash-lite",
        ):
            with self.subTest(model=model_id):
                self.assertTrue(supports_thinking(model_id))
                self.assertTrue(supports_vision(model_id))

    def test_unknown_fallbacks(self):
        self.assertTrue(supports_thinking("openai/o3"))
        self.assertTrue(supports_thinking("moonshot/kimi-k2-thinking"))
        self.assertFalse(supports_thinking("custom/plain"))
        self.assertFalse(supports_vision("custom/plain"))


class ReasoningLevelTests(SimpleTestCase):
    def test_openai_levels(self):
        self.assertEqual(
            get_thinking_levels("openai/gpt-5.6-terra"),
            ["none", "low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(get_default_thinking_level("openai/gpt-5.6-terra"), "medium")
        # Astra dropped the "none" effort level.
        self.assertEqual(
            get_thinking_levels("openai/gpt-6-astra"),
            ["low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(get_default_thinking_level("openai/gpt-6-astra"), "medium")

    def test_fable_has_no_off_level(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-fable-5"),
            ["low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(get_default_thinking_level("anthropic/claude-fable-5"), "high")

    def test_anthropic_levels_are_model_specific(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-sonnet-5"),
            ["off", "low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(
            get_thinking_levels("anthropic/claude-haiku-5-5"),
            ["off", "low", "medium", "high", "xhigh", "max"],
        )

    def test_flash_lite_includes_minimal(self):
        self.assertEqual(
            get_thinking_levels("gemini/gemini-3.5-flash-lite"),
            ["minimal", "low", "medium", "high"],
        )
        self.assertEqual(
            get_default_thinking_level("gemini/gemini-3.5-flash-lite"), "minimal"
        )

    def test_unknown_model_has_no_levels(self):
        self.assertEqual(get_thinking_levels("custom/unknown"), [])
        self.assertIsNone(get_default_thinking_level("custom/unknown"))


class PickerRatingTests(SimpleTestCase):
    def test_price_buckets(self):
        # Rated on the blended price at the prod token mix, not output alone.
        expected = {
            "openai/gpt-6-luna": 1,
            "openai/gpt-5.4-nano": 1,
            "gemini/gemini-3.5-flash-lite": 1,
            # Same $2.50 output as flash-lite, but its >100K band input/cache
            # prices put it with Gemini 3.8 Flash.
            "anthropic/claude-haiku-5-5": 2,
            "gemini/gemini-3.8-flash": 2,
            "openai/gpt-6.1-sol": 3,
            "openai/gpt-6-sol": 3,
            "openai/gpt-5.6-terra": 3,
            "gemini/gemini-3.1-pro-preview": 3,
            "anthropic/claude-sonnet-5-5": 3,
            "anthropic/claude-sonnet-5": 3,
            "openai/gpt-5.6-sol": 4,
            "anthropic/claude-opus-5-5": 4,
            "openai/gpt-6-astra": 5,
            "anthropic/claude-fable-5-1": 5,
            "anthropic/claude-fable-5": 5,
        }
        for model_id, level in expected.items():
            with self.subTest(model=model_id):
                self.assertEqual(get_price_level(model_id), level)
        # Retired IDs rate as their replacement.
        self.assertEqual(get_price_level("gemini/gemini-3.7-flash"), 2)
        self.assertEqual(get_price_level("custom/unknown"), 0)

    def test_price_mix_sums_to_one(self):
        self.assertEqual(
            _PRICE_MIX_CACHE_READ + _PRICE_MIX_CACHE_WRITE + _PRICE_MIX_OUTPUT, Decimal("1"),
        )

    def test_blended_price_falls_back_to_input_without_write_price(self):
        # Gemini has no cache-write charge: those tokens bill as plain input.
        prices = (Decimal("1"), Decimal("0.10"), None, Decimal("5"))
        with patch("llm.service.pricing.get_display_pricing", return_value=prices):
            self.assertEqual(
                _blended_price("gemini/x"),
                _PRICE_MIX_CACHE_READ * Decimal("0.10")
                + _PRICE_MIX_CACHE_WRITE * Decimal("1")
                + _PRICE_MIX_OUTPUT * Decimal("5"),
            )

    def test_price_bucket_boundaries(self):
        cases = [
            (Decimal("0.1499"), 1),
            (Decimal("0.15"), 2),
            (Decimal("0.50"), 2),
            (Decimal("0.5001"), 3),
            (Decimal("1.20"), 3),
            (Decimal("3"), 4),
            (Decimal("3.0001"), 5),
        ]
        for price, level in cases:
            with self.subTest(price=price), patch(
                "llm.display._blended_price", return_value=price,
            ):
                self.assertEqual(get_price_level("x/y"), level)

    def test_capability_buckets(self):
        self.assertEqual(get_capability_level("openai/gpt-5.4-nano"), 1)
        # Manually rated 2 to match Haiku (cheap price tier would say 1).
        self.assertEqual(get_capability_level("openai/gpt-5.6-luna"), 2)
        self.assertEqual(get_capability_level("openai/gpt-5.6-terra"), 3)
        self.assertEqual(get_capability_level("anthropic/claude-opus-5"), 4)
        self.assertEqual(get_capability_level("anthropic/claude-opus-4-8"), 4)
        self.assertEqual(get_capability_level("anthropic/claude-opus-4-6"), 4)
        # Manually held at 4 stars through the promo price ($4.00 would
        # otherwise demote Sol to the standard tier's 3).
        self.assertEqual(get_capability_level("openai/gpt-5.6-sol"), 4)
        self.assertEqual(get_capability_level("openai/gpt-6-astra"), 5)
        self.assertEqual(get_capability_level("anthropic/claude-fable-5-1"), 5)
        self.assertEqual(get_capability_level("anthropic/claude-fable-5"), 5)
        self.assertEqual(get_capability_level("gemini/gemini-3.1-pro-preview"), 3)
        self.assertEqual(get_capability_level("gemini/gemini-3.8-flash"), 2)
        self.assertEqual(get_capability_level("openai/gpt-6-sol"), 4)
        self.assertEqual(get_capability_level("openai/gpt-6-luna"), 2)
        # Held at 4 (not flagship) until it proves out against Fable.
        self.assertEqual(get_capability_level("anthropic/claude-opus-5-5"), 4)

    def test_capability_falls_back_to_price_tier_without_manual_stars(self):
        unstarred = ModelInfo(
            "X", "openai", "x", flagship=True, input_price=Decimal("10.00")
        )
        with patch("llm.display.get_model_info", return_value=unstarred):
            self.assertEqual(get_capability_level("openai/x"), 5)

    def test_tooltips(self):
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-6-astra"),
            "Flagship · $50 / 1M output tokens",
        )
        # Label follows the manual 4-star rating, not the promo-price tier.
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.6-sol"),
            "Standard · $20 / 1M output tokens",
        )
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-opus-5-5"),
            "Standard · $20 / 1M output tokens",
        )
        # 3-star models read as "Mid" under the star-based categories.
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-sonnet-5"),
            "Mid · $10 / 1M output tokens",
        )
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.4-nano"),
            "Cheap · $1.25 / 1M output tokens",
        )
        # Shows the >100K band most requests pay, not the $0.50 base.
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-haiku-5-5"),
            "Mid · $2.50 / 1M output tokens",
        )
        self.assertIsNone(get_model_meta_tooltip("custom/unknown"))
