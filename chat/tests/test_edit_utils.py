"""Unit tests for chat.edit_utils.apply_unique_text_edits.

Shared by document_edit and canvas_edit; resolving edits against the ORIGINAL
snapshot (not the mutated buffer) is the property both rely on.
"""

from django.test import SimpleTestCase

from chat.edit_utils import apply_unique_text_edits


class ApplyUniqueTextEditsTests(SimpleTestCase):
    def test_applies_unique_edit(self):
        text, applied, failed = apply_unique_text_edits("alpha beta gamma", [("beta", "BETA")])
        self.assertEqual(text, "alpha BETA gamma")
        self.assertEqual(applied, 1)
        self.assertEqual(failed, [])

    def test_ambiguous_match_fails_unchanged(self):
        text, applied, failed = apply_unique_text_edits("a a a", [("a", "b")])
        self.assertEqual(applied, 0)
        self.assertEqual(text, "a a a")
        self.assertIn("3 matches", failed[0]["error"])

    def test_missing_text_fails(self):
        text, applied, failed = apply_unique_text_edits("hello", [("world", "x")])
        self.assertEqual(applied, 0)
        self.assertEqual(failed[0]["error"], "Text not found.")

    def test_empty_old_text_fails(self):
        text, applied, failed = apply_unique_text_edits("hello", [("", "x")])
        self.assertEqual(applied, 0)
        self.assertEqual(text, "hello")

    def test_edit_cannot_match_text_inserted_by_earlier_edit(self):
        # Edit 1 inserts "needle"; edit 2 targeting "needle" must resolve against
        # the original (where it's absent) and fail, not match the insertion.
        text, applied, failed = apply_unique_text_edits(
            "start", [("start", "start needle"), ("needle", "REPLACED")]
        )
        self.assertEqual(text, "start needle")
        self.assertEqual(applied, 1)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["error"], "Text not found.")

    def test_overlapping_edits_second_fails(self):
        text, applied, failed = apply_unique_text_edits(
            "hello world", [("hello world", "hi"), ("world", "earth")]
        )
        self.assertEqual(applied, 1)
        self.assertEqual(text, "hi")
        self.assertIn("Overlaps", failed[0]["error"])

    def test_multiple_nonoverlapping_edits_apply(self):
        text, applied, failed = apply_unique_text_edits(
            "one two three", [("three", "3"), ("one", "1")]
        )
        self.assertEqual(text, "1 two 3")
        self.assertEqual(applied, 2)
        self.assertEqual(failed, [])
