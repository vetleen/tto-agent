"""Tests for export-time Markdown table normalization."""

import markdown as md
from django.test import SimpleTestCase

from chat.services import normalize_table_boundaries

TABLE = "| A | B |\n|---|---|\n| 1 | 2 |"


class NormalizeTableBoundariesTests(SimpleTestCase):
    def test_inserts_boundary_before_table_after_prose(self):
        source = "Published before today, so prior art:\n" + TABLE
        self.assertEqual(
            normalize_table_boundaries(source),
            "Published before today, so prior art:\n\n" + TABLE,
        )

    def test_normalized_table_renders_as_html_table(self):
        html = md.markdown(normalize_table_boundaries("Intro:\n" + TABLE), extensions=["tables"])
        self.assertIn("<p>Intro:</p>", html)
        self.assertIn("<table>", html)
        self.assertNotIn("|", html)

    def test_header_without_outer_pipes(self):
        source = "Intro:\nA | B\n--- | ---\n1 | 2"
        self.assertEqual(normalize_table_boundaries(source), "Intro:\n\nA | B\n--- | ---\n1 | 2")

    def test_aligned_delimiter_row(self):
        source = "Intro:\n| A | B |\n|:--|--:|\n| 1 | 2 |"
        self.assertEqual(normalize_table_boundaries(source), "Intro:\n\n| A | B |\n|:--|--:|\n| 1 | 2 |")

    def test_leaves_separated_table_unchanged(self):
        source = "Intro:\n\n" + TABLE
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_leaves_table_after_heading_unchanged(self):
        source = "### Known documents\n" + TABLE
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_leaves_table_after_list_item_unchanged(self):
        source = "- item\n" + TABLE
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_leaves_fenced_code_unchanged(self):
        source = "Intro:\n```\nprose\n" + TABLE + "\n```"
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_leaves_pipe_prose_without_delimiter_unchanged(self):
        source = "Use a | b for alternation.\nNext line."
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_leaves_cell_count_mismatch_unchanged(self):
        source = "Intro:\n| A | B | C |\n|---|---|\n| 1 | 2 | 3 |"
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_setext_underline_is_not_a_table(self):
        source = "Intro\nTitle | sub\n---"
        self.assertEqual(normalize_table_boundaries(source), source)

    def test_quoted_table_gets_quote_separator(self):
        source = "> Intro:\n> | A | B |\n> |---|---|\n> | 1 | 2 |"
        self.assertEqual(
            normalize_table_boundaries(source),
            "> Intro:\n>\n> | A | B |\n> |---|---|\n> | 1 | 2 |",
        )

    def test_separator_goes_before_header_row_only(self):
        source = "Intro with | a pipe\n" + TABLE
        self.assertEqual(normalize_table_boundaries(source), "Intro with | a pipe\n\n" + TABLE)

    def test_rows_of_running_table_are_not_split(self):
        # A row that itself looks like a header + delimiter stays in the table.
        source = "Intro:\n" + TABLE + "\n| 3 | 4 |\n|---|---|"
        self.assertEqual(normalize_table_boundaries(source), "Intro:\n\n" + TABLE + "\n| 3 | 4 |\n|---|---|")

    def test_second_table_after_prose(self):
        source = TABLE + "\n\nMore:\n" + TABLE
        self.assertEqual(normalize_table_boundaries(source), TABLE + "\n\nMore:\n\n" + TABLE)

    def test_is_idempotent(self):
        once = normalize_table_boundaries("Intro:\n" + TABLE + "\n\n> Q:\n> " + TABLE.replace("\n", "\n> "))
        self.assertEqual(normalize_table_boundaries(once), once)

    def test_empty_and_none(self):
        self.assertEqual(normalize_table_boundaries(""), "")
        self.assertIsNone(normalize_table_boundaries(None))
