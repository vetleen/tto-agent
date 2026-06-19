"""Tests for chat.services.render_citations (footnote-style citations)."""
from django.test import SimpleTestCase

from chat.services import render_citations
from core.styles import FOOTNOTE_MARKER as M


class RenderCitationsTests(SimpleTestCase):
    def test_inline_reference_becomes_superscript(self):
        self.assertEqual(render_citations("Growth.[^1]"), "Growth.<sup>1</sup>")

    def test_consecutive_references(self):
        self.assertEqual(
            render_citations("Slower.[^4][^5][^6]"),
            "Slower.<sup>4</sup><sup>5</sup><sup>6</sup>",
        )

    def test_numeric_definition_becomes_ordered_list_item(self):
        self.assertEqual(
            render_citations("[^1]: BCC Research — https://x.test/a"),
            f"1. {M}BCC Research — https://x.test/a",
        )

    def test_non_numeric_definition_keeps_bracket_marker(self):
        self.assertEqual(
            render_citations("[^note]: A source"),
            "\\[note\\] A source",
        )

    def test_reused_labels_per_section_do_not_collide(self):
        # The whole reason we don't use the footnotes extension: section 2 reuses
        # [^1]. Each transforms independently and stays in place.
        src = (
            "A.[^1]\n\n### Sources\n[^1]: First — https://x.test/1\n\n---\n\n"
            "B.[^1]\n\n### Sources\n[^1]: Second — https://x.test/2"
        )
        out = render_citations(src)
        self.assertEqual(
            out,
            f"A.<sup>1</sup>\n\n### Sources\n1. {M}First — https://x.test/1\n\n---\n\n"
            f"B.<sup>1</sup>\n\n### Sources\n1. {M}Second — https://x.test/2",
        )

    def test_skips_fenced_code(self):
        src = "Use.[^1]\n\n```\narr[^1]\n```"
        out = render_citations(src)
        self.assertEqual(out, "Use.<sup>1</sup>\n\n```\narr[^1]\n```")

    def test_noop_without_footnotes(self):
        src = "Just text with [a link](https://x.test) and no footnotes."
        self.assertEqual(render_citations(src), src)

    def test_empty_and_none_safe(self):
        self.assertEqual(render_citations(""), "")
        self.assertEqual(render_citations(None), None)
