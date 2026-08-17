"""Tests for model display and picker capabilities."""

from django.test import SimpleTestCase

from llm.display import (
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
            "anthropic/claude-haiku-4-5",
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

    def test_fable_has_no_off_level(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-fable-5"),
            ["low", "medium", "high", "xhigh", "max"],
        )
        self.assertEqual(get_default_thinking_level("anthropic/claude-fable-5"), "high")

    def test_anthropic_levels_are_model_specific(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-opus-4-6"),
            ["off", "low", "medium", "high", "max"],
        )
        self.assertEqual(
            get_thinking_levels("anthropic/claude-haiku-4-5"),
            ["off", "low", "medium", "high"],
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
        self.assertEqual(get_price_level("openai/gpt-5.4-nano"), 2)
        self.assertEqual(get_price_level("gemini/gemini-3.7-flash"), 2)
        self.assertEqual(get_price_level("openai/gpt-5.6-terra"), 3)
        self.assertEqual(get_price_level("openai/gpt-5.6-sol"), 4)
        self.assertEqual(get_price_level("custom/unknown"), 0)

    def test_capability_buckets(self):
        self.assertEqual(get_capability_level("openai/gpt-5.4-nano"), 1)
        self.assertEqual(get_capability_level("openai/gpt-5.6-luna"), 2)
        self.assertEqual(get_capability_level("openai/gpt-5.6-terra"), 3)
        self.assertEqual(get_capability_level("anthropic/claude-opus-5"), 4)
        self.assertEqual(get_capability_level("anthropic/claude-opus-4-6"), 4)
        self.assertEqual(get_capability_level("openai/gpt-5.6-sol"), 5)
        self.assertEqual(get_capability_level("anthropic/claude-fable-5"), 5)
        self.assertEqual(get_capability_level("gemini/gemini-3.1-pro-preview"), 3)

    def test_tooltips(self):
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.6-sol"),
            "Flagship · $30 / 1M output tokens",
        )
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-opus-5"),
            "Premium · $25 / 1M output tokens",
        )
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.4-nano"),
            "Cheap · $1.25 / 1M output tokens",
        )
        self.assertIsNone(get_model_meta_tooltip("custom/unknown"))
