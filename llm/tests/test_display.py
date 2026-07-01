"""Tests for llm.display module."""

from django.test import TestCase

from llm.display import (
    get_capability_level,
    get_display_name,
    get_model_meta_tooltip,
    get_price_level,
    get_thinking_levels,
    supports_thinking,
    supports_vision,
)


class GetDisplayNameTests(TestCase):

    def test_anthropic_claude_sonnet_with_date(self):
        self.assertEqual(
            get_display_name("anthropic/claude-sonnet-4-5-20250929"),
            "Claude Sonnet 4.5",
        )

    def test_anthropic_claude_opus(self):
        self.assertEqual(
            get_display_name("anthropic/claude-opus-4-6-20260301"),
            "Claude Opus 4.6",
        )

    def test_anthropic_claude_haiku(self):
        self.assertEqual(
            get_display_name("anthropic/claude-haiku-4-5-20251001"),
            "Claude Haiku 4.5",
        )

    def test_anthropic_claude_single_version(self):
        self.assertEqual(
            get_display_name("anthropic/claude-sonnet-4"),
            "Claude Sonnet 4",
        )

    def test_openai_gpt_5_mini(self):
        self.assertEqual(
            get_display_name("openai/gpt-5-mini"),
            "GPT-5 Mini",
        )

    def test_openai_gpt_5(self):
        self.assertEqual(
            get_display_name("openai/gpt-5"),
            "GPT-5",
        )

    def test_openai_o3(self):
        self.assertEqual(get_display_name("openai/o3"), "O3")

    def test_openai_o4_mini(self):
        self.assertEqual(get_display_name("openai/o4-mini"), "O4 Mini")

    def test_gemini_model(self):
        self.assertEqual(
            get_display_name("gemini/gemini-3.5-flash"),
            "Gemini 3.5 Flash",
        )

    def test_unknown_provider(self):
        self.assertEqual(
            get_display_name("custom/my-cool-model"),
            "My Cool Model",
        )

    def test_no_provider_prefix(self):
        self.assertEqual(
            get_display_name("some-model"),
            "Some Model",
        )


class SupportsThinkingTests(TestCase):

    def test_anthropic_models(self):
        self.assertTrue(supports_thinking("anthropic/claude-sonnet-4-5-20250929"))
        self.assertTrue(supports_thinking("anthropic/claude-opus-4-6"))

    def test_openai_reasoning_o3(self):
        self.assertTrue(supports_thinking("openai/o3"))
        self.assertTrue(supports_thinking("openai/o3-mini"))

    def test_openai_reasoning_o4(self):
        self.assertTrue(supports_thinking("openai/o4-mini"))

    def test_openai_reasoning_o1(self):
        self.assertTrue(supports_thinking("openai/o1"))

    def test_openai_gpt54_thinking(self):
        self.assertTrue(supports_thinking("openai/gpt-5.4"))

    def test_openai_gpt_no_thinking(self):
        self.assertFalse(supports_thinking("openai/gpt-5-mini"))

    def test_gemini_thinking(self):
        self.assertTrue(supports_thinking("gemini/gemini-3.1-pro-preview"))
        self.assertTrue(supports_thinking("gemini/gemini-3.5-flash"))
        self.assertTrue(supports_thinking("gemini/gemini-3.1-flash-lite"))

    def test_thinking_in_name(self):
        self.assertTrue(supports_thinking("moonshot/kimi-k2-thinking"))

    def test_plain_model_no_thinking(self):
        self.assertFalse(supports_thinking("custom/my-model"))


class SupportsVisionTests(TestCase):

    def test_anthropic_claude(self):
        self.assertTrue(supports_vision("anthropic/claude-sonnet-4-5-20250929"))
        self.assertTrue(supports_vision("anthropic/claude-opus-4-6"))
        self.assertTrue(supports_vision("anthropic/claude-haiku-4-5-20251001"))

    def test_openai_gpt4(self):
        self.assertTrue(supports_vision("openai/gpt-4o"))
        self.assertTrue(supports_vision("openai/gpt-4-turbo"))

    def test_openai_gpt5(self):
        self.assertTrue(supports_vision("openai/gpt-5"))
        self.assertTrue(supports_vision("openai/gpt-5-mini"))

    def test_openai_o_series_no_vision(self):
        self.assertFalse(supports_vision("openai/o3"))
        self.assertFalse(supports_vision("openai/o4-mini"))

    def test_gemini(self):
        self.assertTrue(supports_vision("gemini/gemini-3.5-flash"))
        self.assertTrue(supports_vision("gemini/gemini-3.1-pro-preview"))

    def test_unknown_model(self):
        self.assertFalse(supports_vision("custom/my-model"))

    def test_openai_gpt3_no_vision(self):
        self.assertFalse(supports_vision("openai/gpt-3.5-turbo"))


class GetThinkingLevelsTests(TestCase):

    def test_opus_47_has_max(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-opus-4-7"),
            ["low", "medium", "high", "max"],
        )

    def test_opus_48_has_max(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-opus-4-8"),
            ["low", "medium", "high", "max"],
        )

    def test_opus_46_standard_levels(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-opus-4-6"),
            ["low", "medium", "high"],
        )

    def test_sonnet_46_standard_levels(self):
        self.assertEqual(
            get_thinking_levels("anthropic/claude-sonnet-4-6"),
            ["low", "medium", "high"],
        )

    def test_sonnet_5_standard_levels(self):
        # Sonnet 5 supports adaptive thinking but, like Sonnet 4.6, does not
        # expose the "max" level in the UI (reserved for the Opus flagship tier).
        self.assertEqual(
            get_thinking_levels("anthropic/claude-sonnet-5"),
            ["low", "medium", "high"],
        )

    def test_non_thinking_model_empty(self):
        self.assertEqual(get_thinking_levels("openai/gpt-5.4-mini"), [])

    def test_unknown_model_empty(self):
        self.assertEqual(get_thinking_levels("custom/unknown"), [])


class GetPriceLevelTests(TestCase):

    def test_nano_is_one_dollar(self):
        # output $0.40 -> bucket 1
        self.assertEqual(get_price_level("openai/gpt-5.4-nano"), 1)

    def test_mid_priced_models(self):
        # $1.50 and $2.00 and $5.00 all fall in bucket 2 (<= $5)
        self.assertEqual(get_price_level("gemini/gemini-3.1-flash-lite"), 2)
        self.assertEqual(get_price_level("openai/gpt-5.4-mini"), 2)
        self.assertEqual(get_price_level("anthropic/claude-haiku-4-5"), 2)  # $5 upper-inclusive

    def test_standard_models(self):
        # $9, $12, $14, $15 all fall in bucket 3 (<= $15)
        self.assertEqual(get_price_level("gemini/gemini-3.5-flash"), 3)
        self.assertEqual(get_price_level("gemini/gemini-3.1-pro-preview"), 3)
        self.assertEqual(get_price_level("openai/gpt-5.4"), 3)
        self.assertEqual(get_price_level("anthropic/claude-sonnet-4-6"), 3)  # $15 upper-inclusive

    def test_premium_models(self):
        # $25 and $30 fall in bucket 4 (<= $50)
        self.assertEqual(get_price_level("anthropic/claude-opus-4-8"), 4)
        self.assertEqual(get_price_level("openai/gpt-5.5"), 4)

    def test_unknown_model_is_zero(self):
        self.assertEqual(get_price_level("custom/unknown"), 0)


class GetCapabilityLevelTests(TestCase):

    def test_cheap_tier_one_star(self):
        self.assertEqual(get_capability_level("openai/gpt-5.4-nano"), 1)
        self.assertEqual(get_capability_level("gemini/gemini-3.1-flash-lite"), 1)

    def test_mid_tier_two_stars(self):
        self.assertEqual(get_capability_level("openai/gpt-5.4-mini"), 2)
        self.assertEqual(get_capability_level("anthropic/claude-haiku-4-5"), 2)

    def test_standard_tier_three_stars(self):
        # Standard tier WITHOUT the cutting_edge flag = 3 stars.
        self.assertEqual(get_capability_level("openai/gpt-5.4"), 3)
        self.assertEqual(get_capability_level("anthropic/claude-opus-4-6"), 3)
        self.assertEqual(get_capability_level("anthropic/claude-sonnet-4-6"), 3)

    def test_cutting_edge_four_stars(self):
        # The manually-curated cutting_edge flag awards a 4th star.
        self.assertEqual(get_capability_level("openai/gpt-5.5"), 4)
        self.assertEqual(get_capability_level("anthropic/claude-opus-4-8"), 4)

    def test_unknown_model_is_zero(self):
        self.assertEqual(get_capability_level("custom/unknown"), 0)


class GetModelMetaTooltipTests(TestCase):

    def test_whole_dollar_price(self):
        # Standard (non-cutting-edge) model leads with its tier.
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-opus-4-6"),
            "Standard · $25 / 1M output tokens",
        )

    def test_cutting_edge_leads_with_label(self):
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-opus-4-8"),
            "Cutting edge · $25 / 1M output tokens",
        )
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.5"),
            "Cutting edge · $30 / 1M output tokens",
        )

    def test_fractional_price_keeps_cents(self):
        self.assertEqual(
            get_model_meta_tooltip("openai/gpt-5.4-nano"),
            "Cheap · $0.40 / 1M output tokens",
        )

    def test_mid_tier_label(self):
        self.assertEqual(
            get_model_meta_tooltip("anthropic/claude-haiku-4-5"),
            "Mid · $5 / 1M output tokens",
        )

    def test_unknown_model_is_none(self):
        self.assertIsNone(get_model_meta_tooltip("custom/unknown"))
