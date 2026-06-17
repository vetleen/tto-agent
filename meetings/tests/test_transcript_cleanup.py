"""Tests for meetings.services.transcript_cleanup.collapse_repetitions.

Pure-function tests (no DB) in the style of StitchTranscriptsTests /
chat.tests.test_dedup. The fixtures mirror the real production failure mode
(meeting 331): a single utterance that loops one Norwegian phrase dozens of
times.
"""
from __future__ import annotations

from django.test import SimpleTestCase

from meetings.services.transcript_cleanup import (
    MIN_RUN,
    _collapse_sentence_runs,
    _normalize_unit,
    collapse_repetitions,
)

# The actual phrase gpt-4o-transcribe looped on in production meeting 331.
LOOP = "Ja, det er en del penger til å gjøre det. "


class NormalizeUnitTests(SimpleTestCase):
    def test_case_edge_punctuation_and_space_insensitive(self):
        # Normalization is robust to case, surrounding whitespace, and
        # leading/trailing punctuation (the ways repeats of one looped sentence
        # usually differ). Internal punctuation is deliberately preserved.
        self.assertEqual(
            _normalize_unit("Det er BRA!"),
            _normalize_unit("  det er bra.  "),
        )

    def test_distinct_sentences_differ(self):
        self.assertNotEqual(_normalize_unit("Ja det er bra."), _normalize_unit("Nei det er feil."))


class CollapseSentenceRunsTests(SimpleTestCase):
    """Directly exercises the run logic (no redundancy/length gate)."""

    def test_run_below_min_run_is_preserved(self):
        # Two consecutive equal units < MIN_RUN(3): keep both.
        text = "Ja. Ja. Nei. Takk."
        self.assertEqual(_collapse_sentence_runs(text).count("Ja."), 2)

    def test_run_at_min_run_is_collapsed(self):
        text = "Ja. Ja. Ja. Nei."
        out = _collapse_sentence_runs(text)
        self.assertEqual(out.count("Ja."), 1)
        self.assertIn("Nei.", out)

    def test_normalized_equal_units_collapse(self):
        # Same sentence, different casing/punctuation each time -> one survivor.
        text = "Det er bra. det er bra! DET ER BRA."
        out = _collapse_sentence_runs(text)
        self.assertEqual(out.lower().count("det er bra"), 1)


class CollapseRepetitionsTests(SimpleTestCase):
    def test_empty_and_short_unchanged(self):
        for s in ("", "   ", "Ja.", "Kort setning her."):
            self.assertEqual(collapse_repetitions(s), s)

    def test_normal_prose_unchanged(self):
        text = (
            "Vi diskuterte budsjettet for neste kvartal og ble enige om at "
            "Thomas følger opp med investorene mens Kari ferdigstiller "
            "prototypen før fristen i mars."
        )
        self.assertEqual(collapse_repetitions(text), text)

    def test_sentence_loop_collapsed_to_one(self):
        looped = LOOP * 40
        out = collapse_repetitions(looped)
        self.assertEqual(out.count("en del penger"), 1)
        self.assertIn("Ja, det er en del penger til å gjøre det.", out)
        self.assertLess(len(out), 100)

    def test_punctuationless_tandem_loop_collapsed(self):
        out = collapse_repetitions("blah " * 60)
        self.assertEqual(out.strip(), "blah")

    def test_two_cycle_alternation_preserved(self):
        # "A B A B" — only two cycles, no >=3 run: must NOT be collapsed.
        a = "The budget was approved by everyone present today."
        b = "We agreed to revisit the open questions next quarter."
        text = f"{a} {b} {a} {b}"
        out = collapse_repetitions(text)
        self.assertEqual(out.count(a), 2)
        self.assertEqual(out.count(b), 2)

    def test_three_cycle_repeat_is_collapsed(self):
        # Three+ consecutive cycles of a multi-sentence block are degeneration.
        a = "The budget was approved by everyone present today."
        b = "We agreed to revisit the open questions next quarter."
        text = f"{a} {b} " * 3
        out = collapse_repetitions(text)
        self.assertEqual(out.count(a), 1)
        self.assertEqual(out.count(b), 1)

    def test_loop_in_middle_preserves_surrounding_text(self):
        prefix = "Velkommen til møtet i dag, takk for at dere kom. "
        suffix = "Da avslutter vi her, vi sees neste uke alle sammen."
        out = collapse_repetitions(prefix + LOOP * 30 + suffix)
        self.assertIn("Velkommen til møtet", out)
        self.assertIn("vi sees neste uke", out)
        self.assertEqual(out.count("en del penger"), 1)

    def test_unicode_survivor_intact(self):
        phrase = "Vi måtte få større støtte fra ledelsen. "
        out = collapse_repetitions(phrase * 40)
        self.assertEqual(out.count("støtte"), 1)
        self.assertIn("Vi måtte få større støtte fra ledelsen.", out)

    def test_idempotent(self):
        for s in (LOOP * 40, "blah " * 60, "Velkommen. " + LOOP * 30 + "Slutt her nå."):
            once = collapse_repetitions(s)
            self.assertEqual(collapse_repetitions(once), once)

    def test_min_run_constant_is_conservative(self):
        # Guard the documented default so a future bump is a deliberate choice.
        self.assertEqual(MIN_RUN, 3)
