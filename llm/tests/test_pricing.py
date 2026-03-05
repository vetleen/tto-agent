"""Tests for llm.service.pricing — model pricing and cost calculation."""

from decimal import Decimal

from django.test import SimpleTestCase

from llm.service.pricing import calculate_cost, get_model_pricing


class GetModelPricingTests(SimpleTestCase):
    """Tests for get_model_pricing()."""

    def test_known_model_returns_tuple(self):
        pricing = get_model_pricing("claude-sonnet-4-6")
        self.assertIsNotNone(pricing)
        self.assertEqual(pricing, (Decimal("3.00"), Decimal("0.30"), Decimal("15.00")))

    def test_openai_model(self):
        pricing = get_model_pricing("gpt-5-mini")
        self.assertEqual(pricing, (Decimal("0.25"), Decimal("0.025"), Decimal("2.00")))

    def test_gemini_model(self):
        pricing = get_model_pricing("gemini-2.5-flash")
        self.assertEqual(pricing, (Decimal("0.30"), Decimal("0.03"), Decimal("2.50")))

    def test_strips_openai_prefix(self):
        pricing = get_model_pricing("openai/gpt-5-mini")
        self.assertEqual(pricing, get_model_pricing("gpt-5-mini"))

    def test_strips_anthropic_prefix(self):
        pricing = get_model_pricing("anthropic/claude-opus-4-6")
        self.assertEqual(pricing, get_model_pricing("claude-opus-4-6"))

    def test_strips_gemini_prefix(self):
        pricing = get_model_pricing("gemini/gemini-2.5-pro")
        self.assertEqual(pricing, get_model_pricing("gemini-2.5-pro"))

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_model_pricing("unknown-model-xyz"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(get_model_pricing(""))


class CalculateCostTests(SimpleTestCase):
    """Tests for calculate_cost()."""

    def test_basic_cost(self):
        # 1000 input, 500 output of gpt-5-mini: (1000*0.25 + 500*2.00) / 1M
        cost = calculate_cost("gpt-5-mini", 1000, 500)
        expected = (Decimal("1000") * Decimal("0.25") + Decimal("500") * Decimal("2.00")) / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_cost_with_cached_tokens(self):
        # 1000 input (300 cached), 500 output of claude-sonnet-4-6
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500, cached_input_tokens=300)
        expected = (
            Decimal("700") * Decimal("3.00")  # billable input
            + Decimal("300") * Decimal("0.30")  # cached input
            + Decimal("500") * Decimal("15.00")  # output
        ) / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_output_only_fallback(self):
        """Streaming case: only output tokens known."""
        cost = calculate_cost("gpt-5-mini", None, 1000)
        expected = Decimal("1000") * Decimal("2.00") / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_unknown_model_returns_none(self):
        self.assertIsNone(calculate_cost("unknown-model", 100, 50))

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-5-mini", 0, 0)
        self.assertEqual(cost, Decimal("0"))

    def test_none_tokens_treated_as_zero(self):
        cost = calculate_cost("gpt-5-mini", None, None)
        self.assertEqual(cost, Decimal("0"))

    def test_all_pricing_entries_have_three_values(self):
        """Sanity check: every entry in the pricing table is a 3-tuple of Decimals."""
        from llm.service.pricing import _PRICING
        for model, prices in _PRICING.items():
            self.assertEqual(len(prices), 3, f"{model} pricing should be a 3-tuple")
            for p in prices:
                self.assertIsInstance(p, Decimal, f"{model} prices should be Decimal")
