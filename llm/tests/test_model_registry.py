"""Tests for llm.model_registry — unified model metadata registry."""

from decimal import Decimal

from django.test import TestCase

from llm.model_registry import ModelInfo, get_model_info


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
            "anthropic/claude-opus-4-6",
            "anthropic/claude-sonnet-4-6",
            "anthropic/claude-haiku-4-5-20251001",
        ]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info, f"{model_id} not found")
            self.assertTrue(info.supports_thinking, f"{model_id} should support thinking")

    def test_gpt5_4_supports_thinking(self):
        info = get_model_info("openai/gpt-5.4")
        self.assertTrue(info.supports_thinking)

    def test_gpt5_mini_no_thinking(self):
        info = get_model_info("openai/gpt-5-mini")
        self.assertFalse(info.supports_thinking)

    def test_gemini_3_supports_thinking(self):
        for model_id in ["gemini/gemini-3-flash", "gemini/gemini-3-pro"]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info, f"{model_id} not found")
            self.assertTrue(info.supports_thinking, f"{model_id} should support thinking")

    def test_gemini_2_no_thinking(self):
        info = get_model_info("gemini/gemini-2.5-flash")
        self.assertFalse(info.supports_thinking)

    def test_all_registered_models_have_vision(self):
        for model_id in [
            "openai/gpt-5.4", "openai/gpt-5-mini", "openai/gpt-5-nano",
            "anthropic/claude-opus-4-6",
            "gemini/gemini-2.5-flash", "gemini/gemini-3-flash",
        ]:
            info = get_model_info(model_id)
            self.assertIsNotNone(info)
            self.assertTrue(info.supports_vision, f"{model_id} should support vision")

    def test_context_windows(self):
        self.assertEqual(get_model_info("openai/gpt-5.4").context_window, 1_000_000)
        self.assertEqual(get_model_info("openai/gpt-5-nano").context_window, 128_000)
        self.assertEqual(get_model_info("anthropic/claude-sonnet-4-6").context_window, 200_000)

    def test_pricing(self):
        info = get_model_info("openai/gpt-5.4")
        self.assertEqual(info.input_price, Decimal("1.75"))
        self.assertEqual(info.cached_input_price, Decimal("0.175"))
        self.assertEqual(info.output_price, Decimal("14.00"))

    def test_bare_name_normalisation(self):
        """Bare model names resolve via prefix scanning."""
        info = get_model_info("gpt-5.4")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "openai")

        info = get_model_info("gemini-3-flash")
        self.assertIsNotNone(info)
        self.assertEqual(info.provider, "google_genai")
