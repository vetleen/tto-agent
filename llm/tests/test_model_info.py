"""Tests for llm.model_info — context window registry and history budget."""

from django.test import TestCase, override_settings

from llm.model_info import get_context_window, get_history_budget


class GetContextWindowTests(TestCase):

    def test_known_model(self):
        self.assertEqual(get_context_window("claude-sonnet-4-6"), 1_000_000)

    def test_known_model_with_prefix(self):
        self.assertEqual(get_context_window("anthropic/claude-sonnet-4-6"), 1_000_000)

    def test_unknown_model_returns_default(self):
        self.assertEqual(get_context_window("unknown-model-xyz"), 128_000)

    def test_none_returns_default(self):
        self.assertEqual(get_context_window(None), 128_000)

    def test_empty_string_returns_default(self):
        self.assertEqual(get_context_window(""), 128_000)

    def test_openai_model(self):
        # gpt-5.4 canonicalizes to gpt-5.6-terra (1,050,000).
        self.assertEqual(get_context_window("gpt-5.4"), 1_050_000)

    def test_gemini_model(self):
        self.assertEqual(get_context_window("gemini-3.1-pro-preview"), 1_048_576)


# Measured budget = min(aim, window) − output_reservation − margin − overhead.
# Defaults pinned here: margin 8k, overhead 24k, floor 4k; the default output
# reservation is 32k (capped by each model's max_output_tokens, all ≥64k here).
_OUT = 32_000  # default output reservation (no effort given)


@override_settings(
    CONTEXT_SAFETY_MARGIN_TOKENS=8_000,
    CONTEXT_INPUT_OVERHEAD_TOKENS=24_000,
    MIN_HISTORY_BUDGET_TOKENS=4_000,
)
class GetHistoryBudgetTests(TestCase):

    def test_reserves_output_and_overhead_no_cap(self):
        # No max_context_tokens → aim is the full window (1M): 1M − 32k − 8k − 24k.
        self.assertEqual(
            get_history_budget("claude-sonnet-4-6"), 1_000_000 - _OUT - 8_000 - 24_000
        )

    def test_setting_is_the_aim(self):
        # The 200k default aim: 200k − 32k − 8k − 24k = 136k.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=200_000), 136_000
        )

    def test_no_150k_cap_anymore(self):
        # A 300k aim now yields 236k history (old code capped this at 150k).
        self.assertEqual(
            get_history_budget("claude-sonnet-4-6", max_context_tokens=300_000), 236_000
        )

    def test_small_setting(self):
        self.assertEqual(
            get_history_budget("gpt-5.4-nano", max_context_tokens=120_000), 56_000
        )

    def test_unknown_model_uses_default_window(self):
        # 128k default window − 32k − 8k − 24k = 64k.
        self.assertEqual(get_history_budget("unknown"), 64_000)

    def test_higher_effort_reserves_more_output(self):
        # high effort → 40k output reservation → less history.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=200_000, effort="high"),
            200_000 - 40_000 - 8_000 - 24_000,
        )

    def test_measured_reserved_tokens_override(self):
        # When the caller measures system+tools+current, it replaces the flat overhead.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=200_000, reserved_tokens=50_000),
            200_000 - _OUT - 8_000 - 50_000,
        )

    def test_floor_when_reservations_exceed_aim(self):
        # A 50k aim is smaller than the reservations (~64k) → history floors at 4k.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=50_000), 4_000
        )
