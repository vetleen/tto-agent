"""Tests for effective model pricing and cost calculation."""

from datetime import date
from decimal import Decimal

from django.test import SimpleTestCase

from llm.service.pricing import calculate_cost, get_display_pricing, get_model_pricing


class PricingLookupTests(SimpleTestCase):
    def test_openai_prices(self):
        self.assertEqual(
            get_model_pricing("openai/gpt-6-astra"),
            (Decimal("10.00"), Decimal("1.00"), Decimal("12.50"), Decimal("50.00")),
        )
        # Retired GPT-5.4 Nano bills as Luna.
        self.assertEqual(
            get_model_pricing("gpt-5.4-nano"),
            (Decimal("0.10"), Decimal("0.01"), Decimal("0.125"), Decimal("0.50")),
        )
        # Cache reads are 0.05x input on GPT-6.1 Sol (not the usual 0.1x).
        self.assertEqual(
            get_model_pricing("openai/gpt-6.1-sol"),
            (Decimal("2.00"), Decimal("0.10"), Decimal("2.50"), Decimal("10.00")),
        )
        self.assertEqual(
            get_model_pricing("openai/gpt-6-luna"),
            (Decimal("0.10"), Decimal("0.01"), Decimal("0.125"), Decimal("0.50")),
        )
        # Cache reads are 0.05x input on GPT-6.1 Terra too.
        self.assertEqual(
            get_model_pricing("openai/gpt-6.1-terra"),
            (Decimal("1.00"), Decimal("0.05"), Decimal("1.25"), Decimal("5.00")),
        )

    def test_opus_55_prices(self):
        # Cache reads are 0.05x input on Opus 5.5 (not the usual 0.1x).
        self.assertEqual(
            get_model_pricing("anthropic/claude-opus-5-5"),
            (Decimal("4.00"), Decimal("0.20"), Decimal("5.00"), Decimal("20.00")),
        )

    def test_fable_52_prices(self):
        # Cache reads are 0.025x input on Fable 5.2.
        self.assertEqual(
            get_model_pricing("anthropic/claude-fable-5-2"),
            (Decimal("8.00"), Decimal("0.20"), Decimal("10.00"), Decimal("40.00")),
        )

    def test_sonnet_55_prices(self):
        # Cache reads are 0.05x input on Sonnet 5.5 (not the usual 0.1x).
        self.assertEqual(
            get_model_pricing("anthropic/claude-sonnet-5-5"),
            (Decimal("2.00"), Decimal("0.10"), Decimal("2.50"), Decimal("10.00")),
        )

    def test_haiku_55_bands_split_at_100k(self):
        self.assertEqual(
            get_model_pricing("claude-haiku-5-5", input_tokens=100_000),
            (Decimal("0.10"), Decimal("0.01"), Decimal("0.125"), Decimal("0.50")),
        )
        self.assertEqual(
            get_model_pricing("claude-haiku-5-5", input_tokens=100_001),
            (Decimal("0.50"), Decimal("0.05"), Decimal("0.625"), Decimal("2.50")),
        )

    def test_display_pricing(self):
        # Haiku 5.5 displays its >100K band; other models their base band.
        self.assertEqual(
            get_display_pricing("anthropic/claude-haiku-5-5"),
            (Decimal("0.50"), Decimal("0.05"), Decimal("0.625"), Decimal("2.50")),
        )
        self.assertEqual(
            get_display_pricing("openai/gpt-6-luna"),
            get_model_pricing("openai/gpt-6-luna"),
        )
        self.assertIsNone(get_display_pricing("unknown/model"))

    def test_long_context_pricing_applies_to_entire_request(self):
        self.assertEqual(
            get_model_pricing("gpt-6.1-terra", input_tokens=272_000),
            (Decimal("1.00"), Decimal("0.05"), Decimal("1.25"), Decimal("5.00")),
        )
        self.assertEqual(
            get_model_pricing("gpt-6.1-terra", input_tokens=272_001),
            (Decimal("2.00"), Decimal("0.10"), Decimal("2.50"), Decimal("7.50")),
        )
        self.assertEqual(
            get_model_pricing("gpt-6-astra", input_tokens=272_001),
            (Decimal("20.00"), Decimal("2.00"), Decimal("25.00"), Decimal("75.00")),
        )
        self.assertEqual(
            get_model_pricing("gpt-6.1-sol", input_tokens=272_001),
            (Decimal("4.00"), Decimal("0.20"), Decimal("5.00"), Decimal("15.00")),
        )
        # A retired ID is priced on its successor's bands.
        self.assertEqual(
            get_model_pricing("gpt-5.6-terra", input_tokens=272_001),
            (Decimal("2.00"), Decimal("0.10"), Decimal("2.50"), Decimal("7.50")),
        )
        self.assertEqual(
            get_model_pricing("gpt-6-luna", input_tokens=272_001),
            (Decimal("0.20"), Decimal("0.02"), Decimal("0.25"), Decimal("0.75")),
        )
        self.assertEqual(
            get_model_pricing("gemini-3.1-pro-preview", input_tokens=200_001),
            (Decimal("4.00"), Decimal("0.40"), Decimal("4.00"), Decimal("18.00")),
        )

    def test_gemini_4_carbon_introductory_price_schedule(self):
        # Gemini has no cache-write charge: those tokens bill as plain input.
        self.assertEqual(
            get_model_pricing("gemini-4-carbon", as_of=date(2026, 11, 30)),
            (Decimal("2.00"), Decimal("0.10"), Decimal("2.00"), Decimal("10.00")),
        )
        self.assertEqual(
            get_model_pricing("gemini-4-carbon", as_of=date(2026, 12, 1)),
            (Decimal("4.00"), Decimal("0.20"), Decimal("4.00"), Decimal("20.00")),
        )

    def test_gemini_4_hydrogen_price_is_flat(self):
        # No introductory price; the retired Flash generations bill as Hydrogen.
        flat = (Decimal("1.50"), Decimal("0.075"), Decimal("1.50"), Decimal("7.50"))
        for model in ("gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.7-flash"):
            for day in (date(2026, 12, 31), date(2027, 1, 1)):
                with self.subTest(model=model, day=day):
                    self.assertEqual(get_model_pricing(model, as_of=day), flat)

    def test_retired_alias_uses_replacement_price(self):
        self.assertEqual(
            get_model_pricing("openai/gpt-5.5"),
            get_model_pricing("openai/gpt-6.1-sol"),
        )
        self.assertEqual(
            get_model_pricing("anthropic/claude-sonnet-5"),
            get_model_pricing("anthropic/claude-sonnet-5-5"),
        )
        self.assertEqual(
            get_model_pricing("anthropic/claude-fable-5"),
            get_model_pricing("anthropic/claude-fable-5-2"),
        )

    def test_unknown_model(self):
        self.assertIsNone(get_model_pricing("unknown/model"))


class DecisionsCostTests(SimpleTestCase):
    def test_bills_input_only_at_the_registry_rates(self):
        from llm.service.pricing import calculate_cost, calculate_decisions_cost

        # Luna: $0.10/M input, $0.01/M cached, $0.125/M cache write; output free.
        # (100k stays under the 272k long-context band.)
        self.assertEqual(calculate_decisions_cost("openai/gpt-6-luna", 100_000), Decimal("0.01"))
        self.assertEqual(
            calculate_decisions_cost("openai/gpt-6-luna", 100_000, cached_input_tokens=50_000),
            Decimal("0.0055"),
        )
        self.assertEqual(
            calculate_decisions_cost("openai/gpt-6-luna", 100_000, cache_write_tokens=20_000),
            Decimal("0.0105"),
        )
        self.assertEqual(
            calculate_decisions_cost("openai/gpt-6-luna", 300_000),
            calculate_cost("openai/gpt-6-luna", 300_000, 0),  # long-context band applies
        )

    def test_unknown_model_or_missing_usage_is_none(self):
        from llm.service.pricing import calculate_decisions_cost

        self.assertIsNone(calculate_decisions_cost("custom/plain", 1000))
        self.assertIsNone(calculate_decisions_cost("openai/gpt-6-luna", None))


class CostTests(SimpleTestCase):
    def test_basic_cost(self):
        self.assertEqual(
            calculate_cost("gpt-6-luna", 1_000, 500),
            Decimal("0.00035"),
        )
        self.assertEqual(
            calculate_cost("gpt-6-astra", 1_000, 500),
            Decimal("0.035"),
        )

    def test_cache_read_and_both_anthropic_write_ttls(self):
        self.assertEqual(
            calculate_cost(
                "claude-fable-5-2",
                1_000,
                100,
                cached_input_tokens=300,
                cache_write_tokens=200,
                cache_write_1h_tokens=100,
            ),
            # 400 regular at $8 + 300 reads at $0.20 + 200 5m writes at $10
            # + 100 1h writes at $16 + 100 output at $40.
            Decimal("0.01066"),
        )

    def test_haiku_55_long_context_cost_includes_1h_writes(self):
        # 200K total input: 100K regular, 50K cache reads, 30K 5m writes,
        # 20K 1h writes, all at the >100K band, plus 10K output.
        self.assertEqual(
            calculate_cost(
                "claude-haiku-5-5",
                200_000,
                10_000,
                cached_input_tokens=50_000,
                cache_write_tokens=50_000,
                cache_write_1h_tokens=20_000,
            ),
            # 0.05 + 0.0025 + 0.01875 + 0.02 (1h at $1.00) + 0.025
            Decimal("0.11625"),
        )

    def test_long_context_cost(self):
        self.assertEqual(
            calculate_cost("gpt-6.1-terra", 300_000, 1_000),
            Decimal("0.6075"),
        )
        self.assertEqual(
            calculate_cost("gpt-6-astra", 300_000, 1_000),
            Decimal("6.075"),
        )

    def test_dated_cost(self):
        promo = calculate_cost(
            "gemini-4-argon", 1_000_000, 1_000_000, as_of=date(2026, 11, 30)
        )
        standard = calculate_cost(
            "gemini-4-argon", 1_000_000, 1_000_000, as_of=date(2026, 12, 1)
        )
        self.assertEqual(promo, Decimal("12"))
        self.assertEqual(standard, Decimal("24"))

    def test_missing_counts_are_zero(self):
        self.assertEqual(calculate_cost("gpt-6-luna", None, None), Decimal("0"))

    def test_unknown_model(self):
        self.assertIsNone(calculate_cost("unknown/model", 10, 10))
