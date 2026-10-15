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
