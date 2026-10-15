"""Tests for the output guard's hard ceiling against the canvas size cap."""

import random

from django.test import SimpleTestCase

from chat.services import CANVAS_MAX_CHARS
from llm.core.output_guard import OutputGuard


def _varied_text(n_chars: int) -> str:
    """Incompressible-enough prose so the repetition check never fires."""
    rng = random.Random(42)
    words = ["patent", "claim", "prior", "art", "assignee", "method", "device", "system",
             "signal", "layer", "matrix", "review", "novelty", "search", "table", "row"]
    out = []
    size = 0
    while size < n_chars:
        w = f"{rng.choice(words)}{rng.randint(0, 99999)} "
        out.append(w)
        size += len(w)
    return "".join(out)[:n_chars]


class OutputCeilingTests(SimpleTestCase):
    def test_ceiling_clears_one_full_canvas_write(self):
        # One full-cap canvas_write plus JSON escaping must not read as a runaway.
        self.assertGreaterEqual(OutputGuard.HARD_CEILING_CHARS, int(CANVAS_MAX_CHARS * 1.05))

    def test_full_canvas_write_passes(self):
        guard = OutputGuard()
        reason = None
        for chunk in _varied_text(CANVAS_MAX_CHARS + 1_000).split("\n"):
            reason = guard.feed(chunk)
        self.assertIsNone(reason)

    def test_well_over_ceiling_trips(self):
        guard = OutputGuard()
        text = _varied_text(OutputGuard.HARD_CEILING_CHARS + 10_000)
        reason = None
        for i in range(0, len(text), 4_000):
            reason = guard.feed(text[i:i + 4_000])
            if reason:
                break
        self.assertEqual(reason, OutputGuard.OUTPUT_CEILING)

    def test_thinking_does_not_count_toward_ceiling(self):
        guard = OutputGuard()
        text = _varied_text(OutputGuard.HARD_CEILING_CHARS + 10_000)
        reason = None
        for i in range(0, len(text), 4_000):
            reason = guard.feed(text[i:i + 4_000], counts_toward_ceiling=False)
        self.assertIsNone(reason)
