"""Tests for chat.services.normalize_heading_levels."""
from django.test import SimpleTestCase

from chat.services import normalize_heading_levels


class NormalizeHeadingLevelsTests(SimpleTestCase):
    def test_promotes_h2_top_to_h1(self):
        out = normalize_heading_levels("## Title\n\nBody.\n\n### Sub")
        self.assertEqual(out, "# Title\n\nBody.\n\n## Sub")

    def test_noop_when_already_h1(self):
        src = "# Title\n\n## Sub\n\nBody."
        self.assertEqual(normalize_heading_levels(src), src)

    def test_noop_without_headings(self):
        src = "Just a paragraph.\n\nAnother one."
        self.assertEqual(normalize_heading_levels(src), src)

    def test_preserves_relative_hierarchy_with_gaps(self):
        # min level is 2, shift by 1 → 2 stays minus 1, 4 becomes 3
        out = normalize_heading_levels("## A\n#### B")
        self.assertEqual(out, "# A\n### B")

    def test_does_not_shift_inside_fenced_code(self):
        src = "## Title\n\n```python\n# a comment, not a heading\n```\n\n### Sub"
        out = normalize_heading_levels(src)
        self.assertEqual(
            out,
            "# Title\n\n```python\n# a comment, not a heading\n```\n\n## Sub",
        )

    def test_tilde_fences_respected(self):
        src = "### Title\n\n~~~\n# not a heading\n~~~"
        out = normalize_heading_levels(src)
        self.assertEqual(out, "# Title\n\n~~~\n# not a heading\n~~~")

    def test_ignores_non_heading_hashes(self):
        # No space after # → not an ATX heading; nothing to promote, no-op.
        src = "#hashtag\n\n###NoSpace"
        self.assertEqual(normalize_heading_levels(src), src)

    def test_preserves_indent_and_text(self):
        out = normalize_heading_levels("  ## Indented Heading")
        self.assertEqual(out, "  # Indented Heading")

    def test_idempotent(self):
        once = normalize_heading_levels("### A\n#### B")
        twice = normalize_heading_levels(once)
        self.assertEqual(once, twice)

    def test_empty_and_none_safe(self):
        self.assertEqual(normalize_heading_levels(""), "")
        self.assertEqual(normalize_heading_levels(None), None)
