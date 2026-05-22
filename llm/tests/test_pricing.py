"""Tests for llm.service.pricing — model pricing and cost calculation."""

from decimal import Decimal

from django.test import SimpleTestCase

from llm.service.pricing import calculate_cost, get_model_pricing


class GetModelPricingTests(SimpleTestCase):
    """Tests for get_model_pricing()."""

    def test_known_model_returns_tuple(self):
        pricing = get_model_pricing("claude-sonnet-4-6")
        self.assertIsNotNone(pricing)
        self.assertEqual(pricing, (Decimal("3.00"), Decimal("0.30"), Decimal("3.75"), Decimal("15.00")))

    def test_openai_model(self):
        pricing = get_model_pricing("gpt-5.4-mini")
        self.assertEqual(pricing, (Decimal("0.25"), Decimal("0.025"), Decimal("0.25"), Decimal("2.00")))

    def test_gemini_model(self):
        pricing = get_model_pricing("gemini-3.5-flash")
        self.assertEqual(pricing, (Decimal("1.50"), Decimal("0.15"), Decimal("1.50"), Decimal("9.00")))

    def test_strips_openai_prefix(self):
        pricing = get_model_pricing("openai/gpt-5.4-mini")
        self.assertEqual(pricing, get_model_pricing("gpt-5.4-mini"))

    def test_strips_anthropic_prefix(self):
        pricing = get_model_pricing("anthropic/claude-opus-4-6")
        self.assertEqual(pricing, get_model_pricing("claude-opus-4-6"))

    def test_strips_gemini_prefix(self):
        pricing = get_model_pricing("gemini/gemini-3.1-pro-preview")
        self.assertEqual(pricing, get_model_pricing("gemini-3.1-pro-preview"))

    def test_unknown_model_returns_none(self):
        self.assertIsNone(get_model_pricing("unknown-model-xyz"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(get_model_pricing(""))


class CalculateCostTests(SimpleTestCase):
    """Tests for calculate_cost()."""

    def test_basic_cost(self):
        # 1000 input, 500 output of gpt-5.4-mini: (1000*0.25 + 500*2.00) / 1M
        cost = calculate_cost("gpt-5.4-mini", 1000, 500)
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

    def test_cost_with_cache_write_tokens(self):
        # 1000 input (200 cache write, 300 cache read), 500 output of claude-sonnet-4-6
        cost = calculate_cost(
            "claude-sonnet-4-6", 1000, 500,
            cached_input_tokens=300, cache_write_tokens=200,
        )
        expected = (
            Decimal("500") * Decimal("3.00")   # regular input (1000 - 300 - 200)
            + Decimal("200") * Decimal("3.75")  # cache write (1.25x)
            + Decimal("300") * Decimal("0.30")  # cache read
            + Decimal("500") * Decimal("15.00") # output
        ) / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_cache_write_without_read(self):
        # First request: everything is a cache write, nothing cached yet
        cost = calculate_cost(
            "anthropic/claude-opus-4-7", 1000, 500,
            cached_input_tokens=0, cache_write_tokens=800,
        )
        expected = (
            Decimal("200") * Decimal("5.00")    # regular input
            + Decimal("800") * Decimal("6.25")   # cache write
            + Decimal("500") * Decimal("25.00")  # output
        ) / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_output_only_fallback(self):
        """Streaming case: only output tokens known."""
        cost = calculate_cost("gpt-5.4-mini", None, 1000)
        expected = Decimal("1000") * Decimal("2.00") / Decimal("1000000")
        self.assertEqual(cost, expected)

    def test_unknown_model_returns_none(self):
        self.assertIsNone(calculate_cost("unknown-model", 100, 50))

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-5.4-mini", 0, 0)
        self.assertEqual(cost, Decimal("0"))

    def test_none_tokens_treated_as_zero(self):
        cost = calculate_cost("gpt-5.4-mini", None, None)
        self.assertEqual(cost, Decimal("0"))

    def test_all_registry_entries_have_pricing(self):
        """Sanity check: every model in the registry has valid pricing."""
        from llm.model_registry import _MODELS
        for model_id, info in _MODELS.items():
            self.assertIsNotNone(info.input_price, f"{model_id} missing input_price")
            self.assertIsInstance(info.input_price, Decimal, f"{model_id} input_price should be Decimal")
            self.assertIsInstance(info.output_price, Decimal, f"{model_id} output_price should be Decimal")
