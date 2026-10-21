"""Tests for llm.context_budget helpers that are pure functions of the aim."""

from django.test import SimpleTestCase

from llm.context_budget import canvas_read_token_cap


class CanvasReadTokenCapTests(SimpleTestCase):
    """30% of the context aim, floored at 12k and capped at 100k."""

    def test_values(self):
        for aim, expected in (
            (None, 60_000),      # aim unknown → DEFAULT_MAX_CONTEXT_TOKENS (200k)
            (0, 60_000),
            (30_000, 12_000),    # floor
            (50_000, 15_000),
            (60_000, 18_000),
            (100_000, 30_000),
            (200_000, 60_000),
            (300_000, 90_000),
            (400_000, 100_000),  # ceiling
            (1_000_000, 100_000),
        ):
            self.assertEqual(canvas_read_token_cap(aim), expected, aim)


class HistoryBudgetsTests(SimpleTestCase):
    """Conversation / summary / stub shares of the history budget (estimate
    units; floors and roofs are in the model's own tokens)."""

    def _b(self, model, aim, reserved=21_000):
        from llm.context_budget import history_budgets

        return history_budgets(model, aim, effort="medium", reserved_tokens=reserved)

    def test_sonnet_table(self):
        from llm.context_budget import as_provider_tokens

        f = "claude-sonnet-5-5"
        for aim, conv, summary in (
            # conversation: 15% (5k floor / 25k roof); summary: 5% of the aim
            # (5k–20k), never more than 60% of the conversation budget.
            (100_000, 6_800, 4_100),
            (200_000, 21_800, 10_000),
            (260_000, 25_000, 13_000),
        ):
            b = self._b(f, aim)
            with self.subTest(aim=aim):
                self.assertAlmostEqual(as_provider_tokens(b.conversation, f), conv, delta=300)
                self.assertAlmostEqual(as_provider_tokens(b.summary_target, f), summary, delta=300)
                self.assertEqual(b.stub_cap, int(0.05 * b.history))

    def test_floors_and_roofs_scale_with_the_estimate_factor(self):
        from llm.context_budget import token_estimate_factor

        tiny = self._b("gpt-5.4", 50_000, reserved=40_000)
        self.assertEqual(tiny.conversation, 5_000)  # floor, factor 1.0
        claude = self._b("claude-sonnet-5-5", 50_000, reserved=40_000)
        self.assertEqual(claude.conversation, int(5_000 / token_estimate_factor("claude-sonnet-5-5")))
        huge = self._b("gpt-5.4", 1_000_000, reserved=20_000)
        self.assertEqual(huge.conversation, 25_000)  # roof
        self.assertLessEqual(huge.summary_target, 20_000)

    def test_summary_never_crowds_out_the_kept_rounds(self):
        b = self._b("gpt-5.4", 60_000)
        self.assertLessEqual(b.summary_target, int(0.6 * b.conversation))
        self.assertEqual(b.as_dict()["conversation"], b.conversation)
