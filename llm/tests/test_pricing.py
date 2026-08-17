"""Tests for effective model pricing and cost calculation."""

from datetime import date
from decimal import Decimal

from django.test import SimpleTestCase

from llm.service.pricing import calculate_cost, get_model_pricing


class PricingLookupTests(SimpleTestCase):
    def test_openai_prices(self):
        self.assertEqual(
            get_model_pricing("openai/gpt-5.6-sol"),
            (Decimal("5.00"), Decimal("0.50"), Decimal("6.25"), Decimal("30.00")),
        )
        self.assertEqual(
            get_model_pricing("gpt-5.4-nano"),
            (Decimal("0.20"), Decimal("0.02"), Decimal("0.25"), Decimal("1.25")),
        )

    def test_long_context_pricing_applies_to_entire_request(self):
        self.assertEqual(
            get_model_pricing("gpt-5.6-terra", input_tokens=272_000),
            (Decimal("2.50"), Decimal("0.25"), Decimal("3.125"), Decimal("15.00")),
        )
        self.assertEqual(
            get_model_pricing("gpt-5.6-terra", input_tokens=272_001),
            (Decimal("5.00"), Decimal("0.50"), Decimal("6.25"), Decimal("22.50")),
        )
        self.assertEqual(
            get_model_pricing("gemini-3.1-pro-preview", input_tokens=200_001),
            (Decimal("4.00"), Decimal("0.40"), Decimal("4.00"), Decimal("18.00")),
        )

    def test_sonnet_5_introductory_price_schedule(self):
        self.assertEqual(
            get_model_pricing("claude-sonnet-5", as_of=date(2026, 8, 31)),
            (Decimal("2.00"), Decimal("0.20"), Decimal("2.50"), Decimal("10.00")),
        )
        self.assertEqual(
            get_model_pricing("claude-sonnet-5", as_of=date(2026, 9, 1)),
            (Decimal("3.00"), Decimal("0.30"), Decimal("3.75"), Decimal("15.00")),
        )

    def test_gemini_flash_introductory_price_schedule(self):
        self.assertEqual(
            get_model_pricing("gemini-3.7-flash", as_of=date(2026, 12, 31)),
            (Decimal("0.75"), Decimal("0.075"), Decimal("0.75"), Decimal("3.75")),
        )
        self.assertEqual(
            get_model_pricing("gemini-3.7-flash", as_of=date(2027, 1, 1)),
            (Decimal("1.50"), Decimal("0.15"), Decimal("1.50"), Decimal("7.50")),
        )

    def test_retired_alias_uses_replacement_price(self):
        self.assertEqual(
            get_model_pricing("openai/gpt-5.5"),
            get_model_pricing("openai/gpt-5.6-sol"),
        )

    def test_unknown_model(self):
        self.assertIsNone(get_model_pricing("unknown/model"))


class CostTests(SimpleTestCase):
    def test_basic_cost(self):
        self.assertEqual(
            calculate_cost("gpt-5.4-nano", 1_000, 500),
            Decimal("0.000825"),
        )

    def test_cache_read_and_both_anthropic_write_ttls(self):
        self.assertEqual(
            calculate_cost(
                "claude-fable-5",
                1_000,
                100,
                cached_input_tokens=300,
                cache_write_tokens=200,
                cache_write_1h_tokens=100,
            ),
            Decimal("0.01355"),
        )

    def test_long_context_cost(self):
        self.assertEqual(
            calculate_cost("gpt-5.6-terra", 300_000, 1_000),
            Decimal("1.5225"),
        )

    def test_dated_cost(self):
        promo = calculate_cost(
            "claude-sonnet-5", 1_000_000, 1_000_000, as_of=date(2026, 8, 31)
        )
        standard = calculate_cost(
            "claude-sonnet-5", 1_000_000, 1_000_000, as_of=date(2026, 9, 1)
        )
        self.assertEqual(promo, Decimal("12"))
        self.assertEqual(standard, Decimal("18"))

    def test_missing_counts_are_zero(self):
        self.assertEqual(calculate_cost("gpt-5.4-nano", None, None), Decimal("0"))

    def test_unknown_model(self):
        self.assertIsNone(calculate_cost("unknown/model", 10, 10))
