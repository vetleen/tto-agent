"""Tests for chat.services.normalize_heading_levels / nest_heading_levels."""
from django.test import SimpleTestCase

from chat.services import nest_heading_levels, normalize_heading_levels


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


class NestHeadingLevelsTests(SimpleTestCase):
    def test_h1_root_nests_under_parent(self):
        self.assertEqual(nest_heading_levels("# A\n## B", 2), "### A\n#### B")

    def test_h2_root_lands_at_same_level(self):
        # The shallowest heading becomes parent + 1 whatever it started at.
        self.assertEqual(nest_heading_levels("## A\n### B", 2), "### A\n#### B")

    def test_deep_root_is_promoted(self):
        self.assertEqual(nest_heading_levels("#### A\n##### B", 1), "## A\n### B")

    def test_noop_when_already_nested(self):
        src = "## A\n\nBody.\n\n### B"
        self.assertEqual(nest_heading_levels(src, 1), src)

    def test_clamps_at_h6(self):
        out = nest_heading_levels("# A\n## B\n#### C\n###### D", 2)
        self.assertEqual(out, "### A\n#### B\n###### C\n###### D")

    def test_skips_fenced_code(self):
        src = "# A\n\n```python\n# comment\n```\n\n~~~\n## also code\n~~~"
        out = nest_heading_levels(src, 2)
        self.assertEqual(out, "### A\n\n```python\n# comment\n```\n\n~~~\n## also code\n~~~")

    def test_ignores_non_heading_hashes(self):
        src = "# A\n#hashtag\n#!/bin/sh\n#123"
        self.assertEqual(nest_heading_levels(src, 1), "## A\n#hashtag\n#!/bin/sh\n#123")

    def test_preserves_indent(self):
        self.assertEqual(nest_heading_levels("   # A", 1), "   ## A")

    def test_noop_without_headings(self):
        src = "Plain text.\n#hashtag"
        self.assertEqual(nest_heading_levels(src, 2), src)

    def test_empty_and_none_safe(self):
        self.assertEqual(nest_heading_levels("", 2), "")
        self.assertIsNone(nest_heading_levels(None, 2))
