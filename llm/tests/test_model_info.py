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
        # No max_context_tokens → aim is the full window: window − 32k − 8k − 24k.
        window = get_context_window("gemini/gemini-4-argon")
        self.assertEqual(
            get_history_budget("gemini/gemini-4-argon"), window - _OUT - 8_000 - 24_000
        )

    def test_setting_is_the_aim(self):
        # The 200k default aim: 200k − 32k − 8k − 24k = 136k.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=200_000), 136_000
        )

    def test_no_150k_cap_anymore(self):
        # A 300k aim now yields 236k history (old code capped this at 150k).
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=300_000), 236_000
        )

    def test_claude_ceiling_in_estimate_units(self):
        # Claude bills ~1.45x our tiktoken estimate (Sonnet 5.5): the provider
        # room (300k − 32k − 8k = 260k) is divided by the factor BEFORE the
        # measured overhead (already in estimate units) is subtracted.
        from llm.context_budget import request_input_ceiling, token_estimate_factor

        self.assertEqual(token_estimate_factor("anthropic/claude-sonnet-5-5"), 1.45)
        self.assertEqual(token_estimate_factor("openai/gpt-6-luna"), 1.0)
        self.assertEqual(token_estimate_factor("unknown-model-xyz"), 1.0)
        self.assertEqual(token_estimate_factor(None), 1.0)
        ceiling = request_input_ceiling("anthropic/claude-sonnet-5-5", 300_000)
        self.assertEqual(ceiling, int(260_000 / 1.45))
        self.assertEqual(
            get_history_budget("anthropic/claude-sonnet-5-5", max_context_tokens=300_000),
            ceiling - 24_000,
        )

    def test_as_provider_tokens(self):
        from llm.context_budget import as_provider_tokens

        self.assertEqual(as_provider_tokens(100_000, "anthropic/claude-sonnet-5-5"), 145_000)
        self.assertEqual(as_provider_tokens(100_000, "openai/gpt-6-luna"), 100_000)
        self.assertIsNone(as_provider_tokens(None, "anthropic/claude-sonnet-5-5"))

    def test_small_setting(self):
        # 120k aim: the 32k reservation is capped at 25% of the aim (30k).
        self.assertEqual(
            get_history_budget("gpt-5.4-nano", max_context_tokens=120_000), 58_000
        )

    def test_reservation_capped_at_quarter_of_aim(self):
        # The 50k floor with xhigh (48k) used to leave no input room at all
        # (ceiling fell to its 1k floor): now 50k - 12.5k - 8k = 29.5k.
        from llm.context_budget import output_reservation, request_input_ceiling

        self.assertEqual(output_reservation("gpt-5.4", "xhigh", 50_000), 12_500)
        self.assertEqual(request_input_ceiling("gpt-5.4", 50_000, "xhigh"), 29_500)
        # A large aim keeps the effort-based reservation.
        self.assertEqual(output_reservation("gpt-5.4", "xhigh", 1_000_000), 48_000)

    def test_unknown_model_uses_default_window(self):
        # 128k default window − 32k − 8k − 24k = 64k.
        self.assertEqual(get_history_budget("unknown"), 64_000)

    def test_output_token_cap_by_effort(self):
        # The provider-side cap is looser than the input-budget reservation:
        # it is the worst case we let the provider bill, reasoning included.
        from llm.context_budget import output_token_cap

        self.assertEqual(output_token_cap("openai/gpt-6-luna", "low"), 24_000)
        self.assertEqual(output_token_cap("openai/gpt-6-luna", "medium"), 32_000)
        self.assertEqual(output_token_cap("openai/gpt-6-luna", "high"), 48_000)
        self.assertEqual(output_token_cap("openai/gpt-6-luna", "xhigh"), 64_000)
        self.assertEqual(output_token_cap("openai/gpt-6-luna", "max"), 64_000)
        # Unknown / absent effort → the default.
        self.assertEqual(output_token_cap("openai/gpt-6-luna", None), 32_000)
        self.assertEqual(output_token_cap("openai/gpt-6-luna", "none"), 32_000)

    def test_output_token_cap_clamped_by_registry_max_output(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from llm.context_budget import output_token_cap

        with patch(
            "llm.context_budget.get_model_info",
            return_value=SimpleNamespace(max_output_tokens=16_384),
        ):
            self.assertEqual(output_token_cap("small-model", "max"), 16_384)
            self.assertEqual(output_token_cap("small-model", "low"), 16_384)

    def test_output_token_cap_unregistered_model_is_conservative(self):
        from llm.context_budget import output_token_cap

        self.assertEqual(output_token_cap("unknown-model-xyz", "max"), 32_000)
        self.assertEqual(output_token_cap("unknown-model-xyz", "low"), 24_000)
        self.assertEqual(output_token_cap(None, "high"), 32_000)

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

    def test_small_aim_keeps_room_for_history(self):
        # 50k aim: output reservation capped at 12.5k → 50k − 12.5k − 8k − 24k.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=50_000), 5_500
        )

    def test_floor_when_reservations_exceed_aim(self):
        # A measured overhead bigger than what's left → history floors at 4k.
        self.assertEqual(
            get_history_budget("gpt-5.4", max_context_tokens=50_000, reserved_tokens=40_000),
            4_000,
        )
