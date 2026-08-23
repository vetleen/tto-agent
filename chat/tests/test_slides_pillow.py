"""Tests for the Pillow slide renderer (pure — no LibreOffice, no DB)."""

from __future__ import annotations

from io import BytesIO

from django.test import SimpleTestCase
from PIL import Image

from chat.slides import pillow_render
from chat.slides.schema import mint_ids


def _open(png_bytes):
    return Image.open(BytesIO(png_bytes)).convert("RGB")


def _colours(img):
    return {c for _n, c in img.getcolors(maxcolors=1 << 20)}


class PillowRenderTests(SimpleTestCase):
    def _deck(self, slides):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": slides}
        mint_ids(deck)
        return deck

    def test_renders_png_at_expected_size(self):
        deck = self._deck([{"elements": [
            {"type": "text", "x": 48, "y": 48, "w": 800, "h": 80, "class": "headline",
             "paragraphs": [{"runs": [{"t": "Hello deck"}]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=120)
        self.assertEqual((w, h), (1600, 900))
        img = _open(png)
        self.assertEqual(img.size, (1600, 900))
        # More than just the background colour was drawn (the headline text).
        self.assertGreater(len(_colours(img)), 1)

    def test_dark_background_and_light_text(self):
        deck = self._deck([{"bg": "dk2", "elements": [
            {"type": "text", "x": 48, "y": 220, "w": 800, "h": 80, "class": "headline",
             "paragraphs": [{"runs": [{"t": "Section", "color": "lt1"}]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        img = _open(png)
        # top-left corner is the dark background
        self.assertLess(sum(img.getpixel((5, 5))), 200)

    def test_all_element_types_render_without_error(self):
        deck = self._deck([{"elements": [
            {"type": "text", "x": 40, "y": 30, "w": 880, "h": 60, "class": "headline",
             "paragraphs": [{"runs": [{"t": "Everything"}]}]},
            {"type": "shape", "x": 40, "y": 110, "w": 200, "h": 80, "shape": "rounded_rect",
             "fill": "accent1", "text": {"paragraphs": [{"align": "center", "runs": [{"t": "Box"}]}]}},
            {"type": "line", "x1": 40, "y1": 210, "x2": 300, "y2": 210, "color": "accent1",
             "w": 3, "arrow": "end"},
            {"type": "table", "x": 40, "y": 240, "w": 880, "h": 160, "header": True, "banding": True,
             "rows": [[{"t": "A"}, {"t": "B"}], [{"t": "1"}, {"t": "2"}], [{"t": "3"}, {"t": "4"}]]},
            {"type": "image", "x": 700, "y": 110, "w": 200, "h": 120, "token": ""},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(png), 1000)
        self.assertGreater(len(_colours(_open(png))), 3)

    def test_bulleted_paragraph_draws_bullet_char(self):
        # The en-dash bullet + text should render (more ink than an empty slide).
        empty = self._deck([{"elements": []}])
        bulleted = self._deck([{"elements": [
            {"type": "text", "x": 48, "y": 120, "w": 800, "h": 300, "class": "body",
             "paragraphs": [
                 {"bullet": True, "runs": [{"t": "First"}]},
                 {"bullet": True, "runs": [{"t": "Second"}]},
             ]},
        ]}])
        e_png, *_ = pillow_render.render_slide_png(empty, 0, dpi=96)
        b_png, *_ = pillow_render.render_slide_png(bulleted, 0, dpi=96)
        self.assertGreater(len(_colours(_open(b_png))), len(_colours(_open(e_png))))

    def test_render_deck_pngs_respects_only_filter(self):
        deck = self._deck([
            {"id": "s1", "elements": []},
            {"id": "s2", "elements": []},
            {"id": "s3", "elements": []},
        ])
        out = pillow_render.render_deck_pngs(deck, dpi=72, only_slide_ids=["s2"])
        self.assertEqual([sid for sid, *_ in out], ["s2"])

    def test_polygon_shapes_and_dashes_render(self):
        deck = self._deck([{"elements": [
            {"type": "shape", "x": 40, "y": 40, "w": 100, "h": 100, "shape": "star", "fill": "accent5"},
            {"type": "shape", "x": 160, "y": 40, "w": 100, "h": 100, "shape": "hexagon", "fill": "accent2"},
            {"type": "shape", "x": 280, "y": 40, "w": 100, "h": 100, "shape": "pentagon", "fill": "accent4"},
            {"type": "shape", "x": 400, "y": 40, "w": 160, "h": 90, "shape": "rect", "fill": "lt2",
             "line": {"color": "accent1", "w": 2, "dash": "dash"}},
            {"type": "line", "x1": 40, "y1": 200, "x2": 900, "y2": 200, "color": "dk2", "w": 2, "dash": "dashdot"},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 3)

    def test_shape_gradient_produces_many_colours(self):
        # A gradient-filled shape must yield a smooth ramp (many distinct colours),
        # unlike a flat fill which adds just one.
        flat = self._deck([{"elements": [
            {"type": "shape", "x": 100, "y": 100, "w": 400, "h": 200, "shape": "rect", "fill": "accent1"},
        ]}])
        grad = self._deck([{"elements": [
            {"type": "shape", "x": 100, "y": 100, "w": 400, "h": 200, "shape": "rect",
             "gradient": {"from": "accent1", "to": "accent2", "angle": 90}},
        ]}])
        flat_png, *_ = pillow_render.render_slide_png(flat, 0, dpi=96)
        grad_png, *_ = pillow_render.render_slide_png(grad, 0, dpi=96)
        self.assertGreater(len(_colours(_open(grad_png))), len(_colours(_open(flat_png))) + 20)

    def test_bg_gradient_fills_background(self):
        # A full-bleed bg_gradient must repaint the flat background with a ramp.
        # skip_footer so the page-number stamp doesn't add ink to the flat case.
        plain = self._deck([{"id": "s1", "bg": "lt1", "skip_footer": True, "elements": []}])
        grad = self._deck([{"id": "s1", "bg": "lt1", "skip_footer": True,
                            "bg_gradient": {"from": "accent1", "to": "dk1", "angle": 120},
                            "elements": []}])
        p_png, *_ = pillow_render.render_slide_png(plain, 0, dpi=72)
        g_png, *_ = pillow_render.render_slide_png(grad, 0, dpi=72)
        self.assertEqual(len(_colours(_open(p_png))), 1)  # flat bg = 1 colour
        self.assertGreater(len(_colours(_open(g_png))), 50)

    def test_gradient_masks_to_poly_shape(self):
        # A gradient on a polygon (chevron) must be clipped to the shape, leaving
        # the surrounding background untouched (top-left corner stays bg colour).
        deck = self._deck([{"id": "s1", "bg": "lt1", "elements": [
            {"type": "shape", "x": 100, "y": 100, "w": 300, "h": 120, "shape": "chevron",
             "gradient": {"from": "accent1", "to": "accent4", "angle": 0}},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=96)
        img = _open(png)
        bg = _open(pillow_render.render_slide_png(
            self._deck([{"id": "s1", "bg": "lt1", "elements": []}]), 0, dpi=96)[0]).getpixel((5, 5))
        self.assertEqual(img.getpixel((5, 5)), bg)  # corner outside the chevron

    def test_network_renders_nodes_edges_labels(self):
        deck = self._deck([{"id": "s1", "bg": "lt1", "skip_footer": True, "elements": [
            {"type": "network", "x": 100, "y": 100, "w": 500, "h": 300,
             "node_color": "accent2", "edge_color": "accent2",
             "nodes": [
                 {"x": 0, "y": 0, "label": "Hub", "label_pos": "c", "emphasis": True, "r": 0},
                 {"x": 200, "y": 40, "label": "Right", "label_pos": "r"},
                 {"x": 40, "y": 200, "label": "Below", "label_pos": "b"},
             ],
             "edges": [{"a": 0, "b": 1}, {"a": 0, "b": 2}]},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=96)
        # Nodes (teal), edges (teal), and dk1 label text = more than the bg colour.
        self.assertGreater(len(_colours(_open(png))), 3)

    def test_network_bad_edge_index_is_skipped(self):
        # An out-of-range edge must be ignored, not crash the render.
        deck = self._deck([{"id": "s1", "elements": [
            {"type": "network", "x": 50, "y": 50, "w": 400, "h": 300,
             "nodes": [{"x": 0, "y": 0}, {"x": 100, "y": 100}],
             "edges": [{"a": 0, "b": 9}, {"a": 0, "b": 1}]},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=72)
        self.assertGreater(len(_colours(_open(png))), 1)

    def test_justified_paragraph_renders(self):
        deck = self._deck([{"elements": [
            {"type": "text", "x": 48, "y": 48, "w": 400, "h": 300, "class": "body",
             "paragraphs": [{"align": "justify", "runs": [{"t": " ".join(["word"] * 40)}]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 1)

    def test_chart_types_render(self):
        for kind in ("column", "bar", "line", "area", "pie"):
            deck = self._deck([{"elements": [
                {"type": "chart", "x": 48, "y": 80, "w": 500, "h": 300, "chart": kind,
                 "title": kind.title(), "categories": ["A", "B", "C"],
                 "series": [{"name": "S1", "values": [3, 5, 4]}, {"name": "S2", "values": [2, 1, 4]}]},
            ]}])
            png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
            self.assertGreater(len(_colours(_open(png))), 3, f"{kind} chart drew nothing")

    def test_glyph_fallback_splits_symbols(self):
        # A symbol the brand font lacks (▲) routes to the Arimo fallback, while
        # ASCII stays on the primary font — so it never renders as a tofu box.
        segs = pillow_render._coverage_segments("A▲B", "Carlito")
        self.assertEqual([s for s, _ in segs], ["A", "▲", "B"])
        self.assertEqual([need for _, need in segs], [False, True, False])
        # a slide using inline momentum triangles renders (more than 1 colour).
        deck = self._deck([{"elements": [
            {"type": "text", "x": 60, "y": 100, "w": 400, "h": 40, "class": "data",
             "paragraphs": [{"runs": [{"t": "▲ ", "color": "#2E75B6"}, {"t": "High"}]}]},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 1)

    def test_every_icon_renders(self):
        from chat.slides import icons
        for name in icons.ICON_NAMES:
            deck = self._deck([{"elements": [
                {"type": "icon", "x": 100, "y": 100, "w": 60, "h": 60,
                 "name": name, "color": "dk2"},
            ]}])
            png, *_ = pillow_render.render_slide_png(deck, 0, dpi=96)
            # each icon draws at least a few dark pixels beyond the empty bg
            self.assertGreater(len(_colours(_open(png))), 1, f"icon {name} drew nothing")

    def test_icon_render_png_and_unknown(self):
        from chat.slides import icons
        png = icons.render_icon_png("check", 48, (0, 0, 0))
        self.assertTrue(png and png[:8] == b"\x89PNG\r\n\x1a\n")
        self.assertIsNone(icons.render_icon_png("nope-not-real", 48, (0, 0, 0)))

    def test_unknown_icon_is_silent(self):
        deck = self._deck([{"elements": [
            {"type": "icon", "x": 100, "y": 100, "w": 60, "h": 60, "name": "does-not-exist"},
            {"type": "text", "x": 40, "y": 300, "w": 400, "h": 40, "class": "headline",
             "paragraphs": [{"runs": [{"t": "ok"}]}]},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 1)  # the text still rendered

    def test_combo_chart_renders(self):
        # Revenue bars (primary) + margin % line (secondary axis) must both draw.
        deck = self._deck([{"elements": [
            {"type": "chart", "x": 60, "y": 80, "w": 620, "h": 320, "chart": "combo",
             "categories": ["Q1", "Q2", "Q3", "Q4"],
             "series": [
                 {"name": "Revenue ($B)", "values": [3.9, 4.0, 4.1, 4.2], "kind": "bar"},
                 {"name": "Op margin %", "values": [10.8, 11.0, 11.2, 11.4],
                  "kind": "line", "axis": "secondary"},
             ]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 4)

    def test_render_chart_png_standalone(self):
        from chat.slides import theme as theme_mod
        el = {"type": "chart", "x": 0, "y": 0, "w": 400, "h": 240, "chart": "combo",
              "categories": ["A", "B"],
              "series": [{"name": "R", "values": [3, 4], "kind": "bar"},
                         {"name": "M", "values": [10, 12], "kind": "line", "axis": "secondary"}]}
        png = pillow_render.render_chart_png(theme_mod.resolve_theme({}), el, k=2)
        self.assertTrue(png[:8] == b"\x89PNG\r\n\x1a\n")

    def test_funnel_renders(self):
        deck = self._deck([{"elements": [
            {"type": "chart", "x": 60, "y": 80, "w": 500, "h": 340, "chart": "funnel",
             "value_labels": True, "categories": ["Visitors", "Signups", "Activated", "Paid"],
             "series": [{"name": "Users", "values": [10000, 3200, 1400, 520]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 3)

    def test_marimekko_columns_math(self):
        series = [{"name": "A", "values": [30, 20]}, {"name": "B", "values": [10, 20]}]
        # explicit widths win
        ws, totals = pillow_render._marimekko_columns(series, ["X", "Y"], [70, 30])
        self.assertEqual(ws, [70.0, 30.0])
        self.assertEqual(totals, [40.0, 40.0])
        # no widths -> width = column total
        ws2, _ = pillow_render._marimekko_columns(series, ["X", "Y"], None)
        self.assertEqual(ws2, [40.0, 40.0])

    def test_marimekko_renders(self):
        deck = self._deck([{"elements": [
            {"type": "chart", "x": 48, "y": 80, "w": 700, "h": 320, "chart": "marimekko",
             "value_labels": True, "categories": ["NA", "EU", "APAC"], "widths": [50, 30, 20],
             "series": [{"name": "HW", "values": [30, 15, 10]},
                        {"name": "SW", "values": [15, 10, 8]},
                        {"name": "Svc", "values": [5, 5, 2]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 4)

    def test_doughnut_punches_a_light_hole(self):
        # Legend off + no title -> the pie centres in the element box; the hole is
        # the light slide bg, not an accent slice.
        deck = self._deck([{"bg": "lt1", "elements": [
            {"type": "chart", "x": 60, "y": 80, "w": 300, "h": 300, "chart": "doughnut",
             "legend": False, "categories": ["New", "Rest"],
             "series": [{"name": "Mix", "values": [34, 66]}], "colors": ["accent1", "dk2"]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=120)
        img = _open(png)
        cx, cy = round((60 + 150) * 120 / 72), round((80 + 150) * 120 / 72)
        r, g, b = img.getpixel((cx, cy))
        self.assertGreater(r + g + b, 600)  # near-white hole at centre

    def test_doughnut_with_center_label_renders(self):
        deck = self._deck([{"bg": "lt1", "elements": [
            {"type": "chart", "x": 60, "y": 80, "w": 300, "h": 300, "chart": "doughnut",
             "categories": ["New", "Rest"], "series": [{"name": "Mix", "values": [34, 66]}],
             "colors": ["accent1", "lt2"], "center_label": "34%"},
        ]}])
        png, *_ = pillow_render.render_slide_png(deck, 0, dpi=120)
        self.assertGreater(len(_colours(_open(png))), 3)

    def test_harvey_fraction_quantises(self):
        self.assertEqual(pillow_render._harvey_fraction(None), 0.0)
        self.assertEqual(pillow_render._harvey_fraction(0.6), 0.5)
        self.assertEqual(pillow_render._harvey_fraction(0.7), 0.75)
        self.assertEqual(pillow_render._harvey_fraction(2), 1.0)
        self.assertEqual(pillow_render._harvey_fraction(-1), 0.0)

    def test_harvey_balls_render_at_each_level(self):
        # A row of Harvey balls (0/.25/.5/.75/1) must draw, and a half ball must
        # differ from an empty ring (more ink).
        empty = self._deck([{"elements": [
            {"type": "shape", "shape": "harvey", "x": 100, "y": 100, "w": 40, "h": 40,
             "value": 0.0, "fill": "dk2"},
        ]}])
        half = self._deck([{"elements": [
            {"type": "shape", "shape": "harvey", "x": 100, "y": 100, "w": 40, "h": 40,
             "value": 0.5, "fill": "dk2"},
        ]}])
        e_png, *_ = pillow_render.render_slide_png(empty, 0, dpi=96)
        h_png, *_ = pillow_render.render_slide_png(half, 0, dpi=96)
        # the half-filled ball paints strictly more dk2 pixels than an empty ring
        def _dark(png):
            img = _open(png)
            return sum(1 for _n, c in img.getcolors(1 << 20) if sum(c) < 200)
        self.assertGreater(_dark(h_png), _dark(e_png))

    def test_per_bar_point_colours_render(self):
        # Single-series column with one accented bar (the rest muted) must render
        # with more than one bar colour — the consulting highlight device.
        deck = self._deck([{"elements": [
            {"type": "chart", "x": 48, "y": 80, "w": 520, "h": 300, "chart": "column",
             "title": "Segments", "categories": ["Long", "Reg", "Last", "Spec"],
             "series": [{"name": "Growth", "values": [4, 6, 9, 3]}],
             "point_colors": ["accent3", "accent3", "accent1", "accent3"]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        # both the muted (accent3) and the accent (accent1) colours are present
        cols = _colours(_open(png))
        self.assertGreater(len(cols), 3)

    def test_waterfall_chart_renders(self):
        deck = self._deck([{"elements": [
            {"type": "chart", "x": 48, "y": 80, "w": 840, "h": 320, "chart": "waterfall",
             "title": "Revenue Bridge", "value_labels": True,
             "categories": ["FY25", "New", "Expansion", "Churn", "FY26"],
             "series": [{"name": "Revenue", "values": [103, 24, 18, -3, 142]}], "totals": [0, 4]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        # up (green), down (red), total (dark) + gridlines/text -> several colours
        self.assertGreater(len(_colours(_open(png))), 4)

    def test_waterfall_bars_math(self):
        bars, edges = pillow_render._waterfall_bars([103, 24, 18, -3, 142], [0, 4])
        roles = [b[2] for b in bars]
        self.assertEqual(roles, ["total", "up", "up", "down", "total"])
        # running totals track the bridge and land on the final total
        self.assertEqual(edges, [103, 127, 145, 142, 142])
        # the churn step floats down from 145 to 142
        self.assertEqual(bars[3][:2], (142.0, 145.0))

    def test_waterfall_defaults_first_to_base(self):
        bars, edges = pillow_render._waterfall_bars([50, 10, -5], [])
        self.assertEqual([b[2] for b in bars], ["total", "up", "down"])
        self.assertEqual(edges, [50, 60, 55])

    def test_stacked_bar_charts_render(self):
        # A stacked column/bar (composition of a total) must render; the stacked
        # axis tops out above the largest per-category SUM, so a tall stack of two
        # large series still fits (differs from grouped, where the axis tops the
        # single max). Just assert both orientations draw multi-colour bars.
        for kind in ("column", "bar"):
            deck = self._deck([{"elements": [
                {"type": "chart", "x": 48, "y": 80, "w": 520, "h": 300, "chart": kind,
                 "stacked": True, "title": "Segments", "categories": ["Q1", "Q2", "Q3", "Q4"],
                 "series": [{"name": "HW", "values": [40, 45, 50, 55]},
                            {"name": "SW", "values": [20, 25, 28, 32]},
                            {"name": "Svc", "values": [10, 12, 14, 16]}]},
            ]}])
            png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
            self.assertGreater(len(_colours(_open(png))), 4, f"stacked {kind} drew nothing")

    def _png_bytes(self, color=(40, 90, 140), size=(200, 120)):
        buf = BytesIO()
        Image.new("RGB", size, color).save(buf, "PNG")
        return buf.getvalue()

    def test_real_image_and_opacity_render(self):
        img_bytes = self._png_bytes()
        resolver = lambda t: (img_bytes, "image/png") if "image:" in t else None
        tok = "[[image:11111111-1111-1111-1111-111111111111]]"
        deck = self._deck([{"elements": [
            {"type": "image", "x": 40, "y": 40, "w": 300, "h": 200, "token": tok, "fit": "cover"},
            {"type": "image", "x": 380, "y": 40, "w": 300, "h": 200, "token": tok, "fit": "cover", "opacity": 0.3},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96, image_resolver=resolver)
        img = _open(png)
        # the opaque image contributes its blue; something beyond the background drew
        self.assertGreater(len(_colours(img)), 2)

    def test_background_image_and_scrim(self):
        img_bytes = self._png_bytes(color=(200, 200, 200))
        resolver = lambda t: (img_bytes, "image/png") if "image:" in t else None
        tok = "[[image:11111111-1111-1111-1111-111111111111]]"
        plain = self._deck([{"bg_image": tok, "elements": []}])
        scrimmed = self._deck([{"bg_image": tok, "bg_scrim": {"color": "dk1", "opacity": 0.6}, "elements": []}])
        p_png, *_ = pillow_render.render_slide_png(plain, 0, dpi=72, image_resolver=resolver)
        s_png, *_ = pillow_render.render_slide_png(scrimmed, 0, dpi=72, image_resolver=resolver)
        # the scrim darkens the (light-grey) background image noticeably
        p_corner = sum(_open(p_png).getpixel((5, 5)))
        s_corner = sum(_open(s_png).getpixel((5, 5)))
        self.assertLess(s_corner, p_corner - 100)

    def test_every_seed_layout_validates_and_renders(self):
        from chat.slides import layouts, schema
        for lid in layouts.LAYOUT_IDS:
            seed = layouts.get_layout(lid)
            seed["name"] = lid
            deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [seed]}
            schema.mint_ids(deck)
            self.assertEqual(schema.validate_deck(deck), [], f"layout {lid} invalid")
            png, w, h = pillow_render.render_slide_png(deck, 0, dpi=72)
            self.assertGreater(len(png), 500, f"layout {lid} rendered nothing")

    def test_rotated_element_renders(self):
        deck = self._deck([{"elements": [
            {"type": "shape", "x": 300, "y": 200, "w": 200, "h": 100, "shape": "rect",
             "fill": "accent1", "rotation": 30,
             "text": {"paragraphs": [{"align": "center", "runs": [{"t": "tilt", "color": "lt1"}]}]}},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 1)

    def test_one_bad_element_does_not_fail_the_slide(self):
        deck = self._deck([{"elements": [
            {"type": "table", "x": 40, "y": 40, "w": 800, "h": 100, "rows": "not-a-list"},
            {"type": "text", "x": 40, "y": 200, "w": 800, "h": 60, "class": "headline",
             "paragraphs": [{"runs": [{"t": "Survivor"}]}]},
        ]}])
        png, w, h = pillow_render.render_slide_png(deck, 0, dpi=96)
        self.assertGreater(len(_colours(_open(png))), 1)  # the text still rendered
