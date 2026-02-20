"""Tests for fallback pricing."""
from decimal import Decimal

from django.test import TestCase

from llm_service.pricing import FALLBACK_PRICING, get_fallback_cost_usd


class PricingTestCase(TestCase):
    """Test get_fallback_cost_usd and FALLBACK_PRICING."""

    def test_returns_none_for_unknown_model(self):
        self.assertIsNone(get_fallback_cost_usd("unknown/model", 1000, 500))

    def test_returns_decimal_for_known_model(self):
        cost = get_fallback_cost_usd("openai/gpt-4o", 1_000_000, 500_000)
        self.assertIsNotNone(cost)
        self.assertIsInstance(cost, Decimal)
        # 1M input * 2.50/1M + 0.5M output * 10/1M = 2.50 + 5 = 7.50
        self.assertAlmostEqual(float(cost), 7.50, places=6)

    def test_zero_tokens_returns_zero_cost(self):
        cost = get_fallback_cost_usd("openai/gpt-4o-mini", 0, 0)
        self.assertEqual(cost, Decimal("0"))

    def test_fallback_pricing_has_expected_models(self):
        self.assertIn("openai/gpt-4o", FALLBACK_PRICING)
        self.assertIn("openai/gpt-4o-mini", FALLBACK_PRICING)
        for model, pricing in FALLBACK_PRICING.items():
            self.assertIn("input", pricing)
            self.assertIn("output", pricing)
