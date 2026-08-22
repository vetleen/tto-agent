"""Tests for chat.slides.schema and chat.slides.theme (pure, no DB)."""

from __future__ import annotations

import copy
import json

from django.test import SimpleTestCase

from chat.slides import schema, theme


def _valid_deck():
    return {
        "version": 1,
        "size": {"w": 960, "h": 540},
        "slides": [
            {
                "id": "s1",
                "name": "Title",
                "skip_footer": True,
                "elements": [
                    {
                        "id": "e1", "type": "text", "x": 60, "y": 200, "w": 840, "h": 120,
                        "class": "headline",
                        "paragraphs": [{"align": "center", "runs": [{"t": "Hello"}]}],
                    }
                ],
            },
            {
                "id": "s2", "name": "Body",
                "elements": [
                    {"id": "e1", "type": "table", "x": 40, "y": 80, "w": 880, "h": 300,
                     "rows": [[{"t": "A"}, {"t": "B"}], [{"t": "1"}, {"t": "2"}]]},
                    {"id": "e2", "type": "line", "x1": 40, "y1": 400, "x2": 920, "y2": 400,
                     "arrow": "end"},
                ],
            },
        ],
    }


class ValidateDeckTests(SimpleTestCase):
    def test_valid(self):
        self.assertEqual(schema.validate_deck(_valid_deck()), [])

    def test_not_an_object(self):
        self.assertTrue(schema.validate_deck([]))

    def test_missing_geometry(self):
        deck = {"slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 1}]}]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("h" in i["path"] for i in issues))

    def test_unknown_element_type(self):
        deck = {"slides": [{"elements": [{"type": "widget", "x": 1, "y": 1, "w": 1, "h": 1}]}]}
        self.assertTrue(schema.validate_deck(deck))

    def test_unknown_top_level_key_forbidden(self):
        deck = _valid_deck()
        deck["bogus"] = 1
        self.assertTrue(schema.validate_deck(deck))

    def test_duplicate_slide_ids(self):
        deck = {"slides": [{"id": "s1"}, {"id": "s1"}]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("Duplicate slide id" in i["message"] for i in issues))

    def test_bad_id_format(self):
        deck = {"slides": [{"id": "S1 bad"}]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("Invalid id" in i["message"] for i in issues))

    def test_too_many_slides(self):
        deck = {"slides": [{} for _ in range(schema.MAX_SLIDES_PER_DECK + 1)]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("Too many slides" in i["message"] for i in issues))

    def test_table_span_rejected(self):
        deck = {"slides": [{"elements": [
            {"type": "table", "x": 1, "y": 1, "w": 1, "h": 1, "rows": [[{"t": "a", "span": 2}]]},
        ]}]}
        self.assertTrue(schema.validate_deck(deck))

    def test_valid_chart(self):
        deck = {"version": 1, "slides": [{"elements": [
            {"type": "chart", "x": 1, "y": 1, "w": 400, "h": 300, "chart": "column",
             "categories": ["A", "B"], "series": [{"name": "S", "values": [1.0, 2.0]}]},
        ]}]}
        self.assertEqual(schema.validate_deck(deck), [])

    def test_chart_needs_a_series(self):
        deck = {"version": 1, "slides": [{"elements": [
            {"type": "chart", "x": 1, "y": 1, "w": 400, "h": 300, "chart": "pie", "series": []},
        ]}]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("at least one series" in i["message"] for i in issues))

    def test_chart_too_many_series(self):
        deck = {"version": 1, "slides": [{"elements": [
            {"type": "chart", "x": 1, "y": 1, "w": 400, "h": 300, "chart": "column",
             "series": [{"name": str(i), "values": [1]} for i in range(schema.MAX_CHART_SERIES + 1)]},
        ]}]}
        issues = schema.validate_deck(deck)
        self.assertTrue(any("Too many chart series" in i["message"] for i in issues))


class MintAndHashTests(SimpleTestCase):
    def test_mint_fills_missing_ids(self):
        deck = {"slides": [{"elements": [{"type": "text", "x": 1, "y": 1, "w": 1, "h": 1}]}, {}]}
        schema.mint_ids(deck)
        self.assertEqual([s["id"] for s in deck["slides"]], ["s1", "s2"])
        self.assertEqual(deck["slides"][0]["elements"][0]["id"], "e1")

    def test_mint_preserves_and_avoids_collision(self):
        deck = {"slides": [{"id": "s2"}, {}]}
        schema.mint_ids(deck)
        ids = [s["id"] for s in deck["slides"]]
        self.assertIn("s2", ids)
        self.assertEqual(len(set(ids)), 2)

    def test_canonical_determinism_under_key_shuffle(self):
        deck = _valid_deck()
        shuffled = json.loads(json.dumps(deck))
        shuffled = {k: shuffled[k] for k in sorted(shuffled, reverse=True)}
        self.assertEqual(
            schema.canonical_deck_text(deck), schema.canonical_deck_text(shuffled)
        )

    def test_hash_is_position_sensitive(self):
        deck = _valid_deck()
        h0 = schema.slide_content_hash(deck, 0)
        moved = copy.deepcopy(deck)
        moved["slides"].append({"id": "s3"})
        # s1 is still index 0 but total changed -> page-number-affecting hash differs
        self.assertNotEqual(h0, schema.slide_content_hash(moved, 0))

    def test_changed_slide_ids(self):
        deck = _valid_deck()
        edited = copy.deepcopy(deck)
        edited["slides"][1]["elements"][0]["rows"][0][0]["t"] = "CHANGED"
        self.assertEqual(schema.changed_slide_ids(deck, edited), ["s2"])

    def test_deck_char_limit(self):
        deck = _valid_deck()
        deck["slides"][0]["name"] = "x" * (schema.DECK_MAX_CHARS + 10)
        issues = schema.validate_deck(deck)
        self.assertTrue(any("too large" in i["message"] for i in issues))


class ThemeTests(SimpleTestCase):
    def test_override_merges(self):
        t = theme.resolve_theme({"theme": {"colors": {"accent1": "#111111"}}})
        self.assertEqual(t["colors"]["accent1"], "#111111")
        self.assertEqual(t["colors"]["dk1"], theme.WILFRED_BASE_THEME["colors"]["dk1"])

    def test_color_resolution(self):
        t = theme.resolve_theme({})
        self.assertEqual(theme.resolve_color(t, "accent2"), ("theme", "accent2"))
        self.assertEqual(theme.resolve_color(t, "#abcdef"), ("rgb", "ABCDEF"))
        self.assertEqual(theme.resolve_color(t, "success")[0], "rgb")
        self.assertIsNone(theme.resolve_color(t, None))
        self.assertIsNone(theme.resolve_color(t, "nonexistent"))

    def test_font_role_and_literal(self):
        t = theme.resolve_theme({})
        self.assertEqual(theme.font_family(t, "headline"), t["fonts"]["headline"])
        self.assertEqual(theme.font_family(t, "Arial"), "Arial")

    def test_run_cascade(self):
        t = theme.resolve_theme({})
        base = theme.base_text_style(t, "headline")
        merged = theme.merge_run(base, {"b": False, "size": 40, "color": "accent1"})
        self.assertFalse(merged["bold"])
        self.assertEqual(merged["size"], 40)
        self.assertEqual(merged["color"], "accent1")
