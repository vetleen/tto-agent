"""Tests for llm.display module."""

from django.test import TestCase

from llm.display import get_display_name, supports_thinking, supports_vision


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
            get_display_name("gemini/gemini-2.5-flash"),
            "Gemini 2.5 Flash",
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

    def test_openai_gpt_no_thinking(self):
        self.assertFalse(supports_thinking("openai/gpt-5"))
        self.assertFalse(supports_thinking("openai/gpt-5-mini"))

    def test_gemini_no_thinking(self):
        self.assertFalse(supports_thinking("gemini/gemini-2.5-flash"))

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
        self.assertTrue(supports_vision("gemini/gemini-2.5-flash"))
        self.assertTrue(supports_vision("gemini/gemini-2.5-pro"))

    def test_unknown_model(self):
        self.assertFalse(supports_vision("custom/my-model"))

    def test_openai_gpt3_no_vision(self):
        self.assertFalse(supports_vision("openai/gpt-3.5-turbo"))
