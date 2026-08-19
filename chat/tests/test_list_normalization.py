"""Tests for export-time Markdown list normalization."""

from django.test import SimpleTestCase

from chat.services import normalize_list_boundaries


class NormalizeListBoundariesTests(SimpleTestCase):
    def test_inserts_boundary_before_tight_ordered_list(self):
        source = "**Recommended next actions:**\n1. Grant power.\n2. Support grant."
        self.assertEqual(
            normalize_list_boundaries(source),
            "**Recommended next actions:**\n\n1. Grant power.\n2. Support grant.",
        )

    def test_normalizes_parenthesized_ordered_list(self):
        source = "Actions:\n1) First\n2) Second"
        self.assertEqual(normalize_list_boundaries(source), "Actions:\n\n1. First\n2. Second")

    def test_normalizes_non_one_parenthesized_list_after_boundary(self):
        source = "Actions:\n\n3) Third\n4) Fourth"
        self.assertEqual(normalize_list_boundaries(source), "Actions:\n\n3. Third\n4. Fourth")

    def test_inserts_boundary_before_each_bullet_style(self):
        for marker in ("-", "+", "*"):
            with self.subTest(marker=marker):
                source = f"Actions:\n{marker} First\n{marker} Second"
                expected = f"Actions:\n\n{marker} First\n{marker} Second"
                self.assertEqual(normalize_list_boundaries(source), expected)

    def test_does_not_insert_between_items_after_continuation(self):
        source = "Actions:\n- First\n  continuation\n- Second"
        expected = "Actions:\n\n- First\n  continuation\n- Second"
        self.assertEqual(normalize_list_boundaries(source), expected)

    def test_separates_lists_when_marker_type_changes(self):
        cases = (
            ("- First\n1) Second", "- First\n\n1. Second"),
            ("1. First\n2) Second", "1. First\n\n2. Second"),
            ("1) First\n2. Second", "1. First\n\n2. Second"),
            ("- First\n* Second", "- First\n\n* Second"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_list_boundaries(source), expected)

    def test_leaves_already_separated_list_unchanged(self):
        source = "Actions:\n\n1. First\n2. Second"
        self.assertEqual(normalize_list_boundaries(source), source)

    def test_leaves_non_interrupting_numeric_prose_unchanged(self):
        for source in ("Planning:\n2. Later", "Planning:\n2026. Launch", "Planning:\n2) Later"):
            with self.subTest(source=source):
                self.assertEqual(normalize_list_boundaries(source), source)

    def test_skips_backtick_and_tilde_fences(self):
        for fence in ("```", "~~~"):
            with self.subTest(fence=fence):
                source = f"{fence}text\nActions:\n1) First\n{fence}"
                self.assertEqual(normalize_list_boundaries(source), source)

    def test_skips_fenced_code_inside_blockquote(self):
        source = "> ```text\n> Actions:\n> 1) First\n> ```"
        self.assertEqual(normalize_list_boundaries(source), source)

    def test_preserves_blockquote_when_inserting_boundary(self):
        source = "> Actions:\n> 1. First\n> 2. Second"
        expected = "> Actions:\n>\n> 1. First\n> 2. Second"
        self.assertEqual(normalize_list_boundaries(source), expected)

    def test_idempotent(self):
        once = normalize_list_boundaries("Actions:\n1) First\n2) Second")
        self.assertEqual(normalize_list_boundaries(once), once)

    def test_empty_and_none_safe(self):
        self.assertEqual(normalize_list_boundaries(""), "")
        self.assertIsNone(normalize_list_boundaries(None))
