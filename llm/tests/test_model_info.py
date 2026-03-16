"""Tests for llm.model_info — context window registry and history budget."""

from django.test import TestCase

from llm.model_info import get_context_window, get_history_budget


class GetContextWindowTests(TestCase):

    def test_known_model(self):
        self.assertEqual(get_context_window("claude-sonnet-4-6"), 200_000)

    def test_known_model_with_prefix(self):
        self.assertEqual(get_context_window("anthropic/claude-sonnet-4-6"), 200_000)

    def test_unknown_model_returns_default(self):
        self.assertEqual(get_context_window("unknown-model-xyz"), 128_000)

    def test_none_returns_default(self):
        self.assertEqual(get_context_window(None), 128_000)

    def test_empty_string_returns_default(self):
        self.assertEqual(get_context_window(""), 128_000)

    def test_openai_model(self):
        self.assertEqual(get_context_window("gpt-5.4"), 1_000_000)

    def test_gemini_model(self):
        self.assertEqual(get_context_window("gemini-2.5-pro"), 1_000_000)


class GetHistoryBudgetTests(TestCase):

    def test_budget_is_75_percent_of_context(self):
        # claude-sonnet-4-6: 200k * 0.75 = 150k (at cap)
        self.assertEqual(get_history_budget("claude-sonnet-4-6"), 150_000)

    def test_budget_capped_at_150k(self):
        # gpt-5.4: 1M * 0.75 = 750k, capped to 150k
        self.assertEqual(get_history_budget("gpt-5.4"), 150_000)

    def test_small_context_not_capped(self):
        # gpt-5-nano: 128k * 0.75 = 96k (under cap)
        self.assertEqual(get_history_budget("gpt-5-nano"), 96_000)

    def test_none_model(self):
        # Default: 128k * 0.75 = 96k
        self.assertEqual(get_history_budget(None), 96_000)

    def test_unknown_model(self):
        self.assertEqual(get_history_budget("unknown"), 96_000)
