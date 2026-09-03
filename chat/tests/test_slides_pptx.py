"""Golden-OOXML tests for the python-pptx builder (no DB, no LibreOffice)."""

from __future__ import annotations

import base64
import io
import zipfile

from django.test import SimpleTestCase

from chat.slides.pptx_build import build_deck_pptx

# 1x1 transparent PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _rich_deck():
    return {
        "version": 1, "size": {"w": 960, "h": 540},
        "theme": {"colors": {"accent1": "#B87333"}},
        "slides": [
            {"id": "s1", "name": "Title", "skip_footer": True, "bg": "lt1", "elements": [
                {"id": "e1", "type": "text", "x": 80, "y": 210, "w": 800, "h": 120, "class": "headline",
                 "paragraphs": [{"align": "center", "runs": [{"t": "Quarterly Review"}]}]},
            ]},
            {"id": "s2", "name": "Content", "elements": [
                {"id": "e1", "type": "text", "x": 48, "y": 40, "w": 864, "h": 60, "class": "headline",
                 "paragraphs": [{"runs": [{"t": "Pipeline"}]}]},
                {"id": "e2", "type": "text", "x": 48, "y": 120, "w": 520, "h": 300, "class": "body",
                 "paragraphs": [{"bullet": True, "runs": [{"t": "One"}]},
                                {"bullet": True, "runs": [{"t": "Two "}, {"t": "bold", "b": True}]}]},
                {"id": "e3", "type": "shape", "x": 600, "y": 120, "w": 300, "h": 80, "box": "callout",
                 "text": {"paragraphs": [{"align": "center", "runs": [{"t": "38%"}]}]}},
                {"id": "e4", "type": "shape", "x": 600, "y": 230, "w": 120, "h": 60, "shape": "right_arrow", "fill": "accent2"},
                {"id": "e5", "type": "line", "x1": 48, "y1": 440, "x2": 912, "y2": 440,
                 "color": "accent3", "w": 2, "dash": "dash", "arrow": "end"},
                {"id": "e6", "type": "image", "x": 740, "y": 300, "w": 160, "h": 120,
                 "token": "[[image:11111111-1111-1111-1111-111111111111]]", "fit": "cover"},
            ]},
            {"id": "s3", "name": "Table", "elements": [
                {"id": "e1", "type": "table", "x": 48, "y": 80, "w": 864, "h": 260,
                 "header": True, "banding": True, "col_widths": [300, 282, 282],
                 "rows": [[{"t": "Stage"}, {"t": "Count"}, {"t": "Value"}],
                          [{"t": "Filed"}, {"t": "7"}, {"t": "$3.4M"}]]},
            ]},
        ],
    }


def _resolver(token):
    return (_PNG, "image/png") if "image:" in token else None


class BuildDeckTests(SimpleTestCase):
    def setUp(self):
        data, self.warnings = build_deck_pptx(
            _rich_deck(), base_template="wilfred_default", image_resolver=_resolver
        )
        self.zip = zipfile.ZipFile(io.BytesIO(data))
        self.s2 = self.zip.read("ppt/slides/slide2.xml").decode()
        self.s3 = self.zip.read("ppt/slides/slide3.xml").decode()

    def test_no_warnings(self):
        self.assertEqual(self.warnings, [])

    def test_background_image_scrim_and_opacity(self):
        tok = "[[image:11111111-1111-1111-1111-111111111111]]"
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1",
            "bg_image": tok, "bg_scrim": {"color": "dk1", "opacity": 0.5}, "elements": [
                {"id": "i1", "type": "image", "x": 40, "y": 40, "w": 200, "h": 120,
                 "token": tok, "opacity": 0.3},
                {"id": "sh1", "type": "shape", "x": 300, "y": 40, "w": 200, "h": 120,
                 "shape": "rect", "fill": "accent1", "opacity": 0.4},
            ]}]}
        data, warns = build_deck_pptx(deck, image_resolver=_resolver)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = "".join(z.read(n).decode("utf8", "ignore") for n in z.namelist() if n.endswith(".xml"))
        self.assertIn("alphaModFix", xml)   # picture opacity (image + bg)
        self.assertIn("a:alpha", xml)       # shape fill opacity (shape + scrim)
        self.assertEqual(warns, [])

    def test_chart_creates_native_chart_part(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "c1", "type": "chart", "x": 48, "y": 80, "w": 500, "h": 300, "chart": "column",
             "title": "T", "categories": ["A", "B"], "series": [{"name": "S", "values": [1, 2]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        charts = [n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml")]
        self.assertTrue(charts, "no native chart part embedded in the .pptx")
        self.assertEqual(warns, [])

    def test_icon_embeds_as_picture(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "ic", "type": "icon", "x": 80, "y": 120, "w": 28, "h": 28,
             "name": "trend_up", "color": "accent1"},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        media = [n for n in z.namelist() if n.startswith("ppt/media/")]
        self.assertTrue(media, "icon PNG not embedded in the .pptx")
        self.assertEqual(warns, [])

    def test_combo_embeds_as_picture(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "cb", "type": "chart", "x": 60, "y": 80, "w": 620, "h": 320, "chart": "combo",
             "categories": ["Q1", "Q2", "Q3"],
             "series": [{"name": "Rev", "values": [3.9, 4.0, 4.2], "kind": "bar"},
                        {"name": "Margin", "values": [10.8, 11.1, 11.4], "kind": "line",
                         "axis": "secondary"}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue([n for n in z.namelist() if n.startswith("ppt/media/")])
        self.assertEqual(warns, [])

    def test_funnel_builds_rectangles(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "fn", "type": "chart", "x": 60, "y": 80, "w": 500, "h": 340, "chart": "funnel",
             "categories": ["Visitors", "Signups", "Paid"],
             "series": [{"name": "Users", "values": [10000, 3000, 500]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertGreaterEqual(xml.count('prst="rect"'), 3)  # one rect per stage
        self.assertIn("Signups", xml)  # stage label
        self.assertEqual(warns, [])

    def test_marimekko_builds_rectangles(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "mk", "type": "chart", "x": 48, "y": 80, "w": 700, "h": 320, "chart": "marimekko",
             "categories": ["NA", "EU"], "widths": [60, 40],
             "series": [{"name": "HW", "values": [30, 20]}, {"name": "SW", "values": [10, 15]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertGreaterEqual(xml.count('prst="rect"'), 4)  # >=4 mosaic segments
        self.assertIn("HW", xml)  # legend series name
        self.assertEqual(warns, [])

    def test_scatter_creates_native_chart_part(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "sc", "type": "chart", "x": 48, "y": 80, "w": 620, "h": 340, "chart": "scatter",
             "series": [{"name": "A", "points": [[3, 120], [5, 90], [8, 200]]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue([n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml")],
                        "no native scatter chart part embedded")
        self.assertEqual(warns, [])

    def test_bubble_scatter_builds(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "bb", "type": "chart", "x": 48, "y": 80, "w": 620, "h": 340, "chart": "scatter",
             "series": [{"name": "A", "points": [[3, 120, 10], [5, 90, 40]]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        self.assertTrue(data and warns == [])

    def test_histogram_creates_native_chart_part(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "hg", "type": "chart", "x": 48, "y": 80, "w": 620, "h": 340, "chart": "histogram",
             "bins": 6, "series": [{"name": "d", "values": [12, 15, 18, 22, 25, 28, 33, 40, 19, 24]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue([n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml")],
                        "no native histogram chart part embedded")
        self.assertEqual(warns, [])

    def test_dot_builds_shapes(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "dt", "type": "chart", "x": 48, "y": 80, "w": 620, "h": 340, "chart": "dot",
             "categories": ["Alpha", "Beta", "Gamma"], "series": [{"name": "S", "values": [72, 55, 88]}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertGreaterEqual(xml.count('prst="ellipse"'), 3)  # one dot per category
        self.assertIn("Alpha", xml)  # category label
        self.assertEqual(warns, [])

    def test_bullet_builds_shapes(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "bl", "type": "chart", "x": 48, "y": 80, "w": 620, "h": 340, "chart": "bullet",
             "categories": ["Revenue", "NPS"], "series": [{"values": [72, 58]}],
             "targets": [80, 60], "bands": [40, 70, 100]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertGreaterEqual(xml.count('prst="rect"'), 4)  # band + measure rects
        self.assertIn("Revenue", xml)
        self.assertEqual(warns, [])

    def test_roadmap_gantt_layout_builds(self):
        from chat.slides.layouts import get_layout
        from chat.slides.schema import mint_ids
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [get_layout("roadmap_gantt")]}
        mint_ids(deck)
        data, warns = build_deck_pptx(deck)
        self.assertTrue(data and warns == [])

    def test_shape_gradient_writes_gradfill(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "g1", "type": "shape", "shape": "rounded_rect", "x": 60, "y": 80, "w": 240, "h": 150,
             "gradient": {"from": "accent1", "to": "accent2", "angle": 45}},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertIn("<a:gradFill", xml)
        self.assertIn("<a:lin ", xml)                 # linear gradient direction
        self.assertIn('ang="2700000"', xml)           # 45° clockwise, in 60000ths
        self.assertNotIn("<a:satMod", xml)            # preset colour mods stripped
        self.assertEqual(warns, [])

    def test_gradient_with_opacity_fades_both_stops(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "g1", "type": "shape", "shape": "rect", "x": 60, "y": 80, "w": 240, "h": 150,
             "opacity": 0.4, "gradient": {"from": "accent1", "to": "accent2", "angle": 90}},
        ]}]}
        data, warns = build_deck_pptx(deck)
        xml = zipfile.ZipFile(io.BytesIO(data)).read("ppt/slides/slide1.xml").decode()
        self.assertIn("<a:gradFill", xml)
        self.assertEqual(xml.count("<a:alpha"), 2)   # both stops faded
        self.assertEqual(warns, [])

    def test_bg_gradient_writes_fullbleed_gradfill(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{
            "id": "s1", "bg_gradient": {"from": "accent1", "to": "dk1", "angle": 90},
            "elements": [],
        }]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertIn("<a:gradFill", xml)
        self.assertEqual(warns, [])

    def test_slide_is_dark_accounts_for_scrim(self):
        from chat.slides import pptx_build, theme as tmod
        theme = tmod.resolve_theme({})
        self.assertFalse(pptx_build._slide_is_dark({}, theme))                      # light default
        self.assertTrue(pptx_build._slide_is_dark({"bg": "dk1"}, theme))            # dark bg
        # A dark scrim over the (light) default backdrop reads as dark:
        self.assertTrue(pptx_build._slide_is_dark({"bg_scrim": {"color": "dk1", "opacity": 0.7}}, theme))
        # A faint scrim leaves it light:
        self.assertFalse(pptx_build._slide_is_dark({"bg_scrim": {"color": "dk1", "opacity": 0.1}}, theme))

    def test_export_bakes_active_theme_into_theme_part(self):
        # A non-base theme must reach the downloaded file: its colours are baked
        # into ppt/theme/theme1.xml while the slide keeps the editable schemeClr
        # reference (so PowerPoint's colour picker still tracks the palette).
        import re
        from chat.slides import theme as tmod

        base_a1 = tmod.WILFRED_BASE_THEME["colors"]["accent1"].lstrip("#").upper()
        ocean = tmod.preset_theme_override("ocean")
        ocean_a1 = ocean["colors"]["accent1"].lstrip("#").upper()

        def theme_accent1(deck):
            data, _ = build_deck_pptx(deck)
            z = zipfile.ZipFile(io.BytesIO(data))
            th = z.read("ppt/theme/theme1.xml").decode()
            slide = z.read("ppt/slides/slide1.xml").decode()
            m = re.search(r"<a:accent1>\s*<a:srgbClr val=\"([0-9A-Fa-f]{6})\"", th)
            return (m.group(1).upper() if m else None), ('schemeClr val="accent1"' in slide)

        el = [{"type": "shape", "x": 40, "y": 40, "w": 200, "h": 100, "shape": "rect", "fill": "accent1"}]
        o_hex, o_ref = theme_accent1({"version": 1, "size": {"w": 960, "h": 540},
                                      "theme": ocean, "slides": [{"id": "s1", "elements": el}]})
        b_hex, b_ref = theme_accent1({"version": 1, "size": {"w": 960, "h": 540},
                                      "slides": [{"id": "s1", "elements": el}]})
        self.assertEqual(o_hex, ocean_a1)   # ocean colour reached the theme part
        self.assertTrue(o_ref)              # slide still references the slot (not inlined)
        self.assertEqual(b_hex, base_a1)    # base deck unchanged (regression guard)
        self.assertTrue(b_ref)

    def test_shorthand_and_malformed_hex_dont_crash_build(self):
        from chat.slides import theme as tmod
        theme = tmod.resolve_theme({})
        self.assertEqual(tmod.resolve_color(theme, "#FFF"), ("rgb", "FFFFFF"))  # shorthand expands
        self.assertIsNone(tmod.resolve_color(theme, "#zzz"))                    # malformed -> no colour
        # bg + fill using shorthand hex must build a valid .pptx, not 500.
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "bg": "#FFF",
            "elements": [{"type": "shape", "x": 10, "y": 10, "w": 100, "h": 50, "shape": "rect", "fill": "#0A0"}]}]}
        data, warns = build_deck_pptx(deck)
        self.assertGreater(len(data), 0)
        self.assertEqual(warns, [])

    def test_string_opacity_does_not_drop_element(self):
        # Lax validation can let "0.5" through; the shape must still build (not
        # be silently dropped by a "0.5" < 1 TypeError).
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "sh", "type": "shape", "x": 10, "y": 10, "w": 100, "h": 50, "shape": "rect",
             "fill": "accent1", "opacity": "0.5"}]}]}
        data, warns = build_deck_pptx(deck)
        xml = zipfile.ZipFile(io.BytesIO(data)).read("ppt/slides/slide1.xml").decode()
        self.assertIn("prstGeom", xml)   # the shape is present in the .pptx
        self.assertEqual(warns, [])

    def test_chart_on_dark_bg_gets_light_font(self):
        # A chart on a dark slide must flip its auto-drawn text to light so it's
        # legible in the download (the native chart has no other colour control).
        from chat.slides import theme as tmod
        lt1_hex = tmod.resolve_theme({})["colors"]["lt1"].lstrip("#").upper()

        def chart_xml(bg):
            deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{
                "id": "s1", "bg": bg, "elements": [
                    {"type": "chart", "x": 60, "y": 80, "w": 600, "h": 340, "chart": "column",
                     "categories": ["A", "B"], "series": [{"name": "S", "values": [1, 2]}]}]}]}
            data, _ = build_deck_pptx(deck)
            z = zipfile.ZipFile(io.BytesIO(data))
            part = next(n for n in z.namelist() if n.startswith("ppt/charts/chart") and n.endswith(".xml"))
            return z.read(part).decode().upper()

        self.assertIn(lt1_hex, chart_xml("dk1"))       # dark slide -> light chart text
        self.assertNotIn(lt1_hex, chart_xml("lt1"))    # light slide -> no light-text override

    def test_bullet_glyphs_and_indents_step_by_level(self):
        import re

        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "b1", "type": "text", "x": 48, "y": 100, "w": 864, "h": 300, "class": "body",
             "paragraphs": [
                 {"bullet": True, "runs": [{"t": "Top"}]},
                 {"bullet": True, "level": 1, "runs": [{"t": "Sub"}]},
                 {"bullet": True, "level": 2, "runs": [{"t": "Fine"}]},
                 {"bullet": True, "level": 3, "runs": [{"t": "Deeper"}]},
             ]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        xml = zipfile.ZipFile(io.BytesIO(data)).read("ppt/slides/slide1.xml").decode()
        # ‣ / – / ◦, then the last glyph repeats for deeper levels.
        self.assertEqual(re.findall(r'buChar char="(.)"', xml), ["‣", "–", "◦", "◦"])
        # The glyphs reference Arial (the run face may lack them, e.g. Cambria ‣).
        self.assertIn('buFont typeface="Arial"', xml)
        # marL steps 20pt per level off the 18pt hang (EMU = pt * 12700).
        self.assertEqual(re.findall(r'marL="(\d+)"', xml),
                         ["228600", "482600", "736600", "990600"])
        self.assertEqual(warns, [])

    def test_rounded_rect_panel_corner_capped_and_insets_roomier(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "p1", "type": "shape", "shape": "rounded_rect", "x": 48, "y": 120,
             "w": 864, "h": 350, "fill": "lt2",
             "text": {"paragraphs": [{"bullet": True, "runs": [{"t": "Key point"}]}]}},
        ]}]}
        data, warns = build_deck_pptx(deck)
        xml = zipfile.ZipFile(io.BytesIO(data)).read("ppt/slides/slide1.xml").decode()
        # 12pt radius cap on a 350pt-tall panel -> adj = 12/350 = 3428/100000
        # (not the stock 16667 pill curve).
        self.assertIn('fmla="val 3428"', xml)
        # Panel text insets widen to 16/12pt (EMU): text stops hugging the edge.
        self.assertIn('lIns="203200"', xml)
        self.assertIn('tIns="152400"', xml)
        self.assertEqual(warns, [])

    def test_curved_line_embeds_picture_straight_stays_connector(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "c1", "type": "line", "x1": 60, "y1": 120, "x2": 400, "y2": 120,
             "curve": 0.3, "arrow": "end", "color": "accent1", "w": 3},
            {"id": "s2", "type": "line", "x1": 60, "y1": 300, "x2": 400, "y2": 300,
             "color": "dk2", "w": 2},
        ]}]}
        data, warns = build_deck_pptx(deck)
        xml = zipfile.ZipFile(io.BytesIO(data)).read("ppt/slides/slide1.xml").decode()
        self.assertEqual(xml.count("<p:pic>"), 1)     # curved line rasterised
        self.assertEqual(xml.count("<p:cxnSp>"), 1)   # straight line stays native
        self.assertEqual(warns, [])

    def test_network_builds_connectors_ovals_and_labels(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "nw", "type": "network", "x": 100, "y": 100, "w": 500, "h": 300,
             "nodes": [
                 {"x": 0, "y": 0, "label": "Core", "label_pos": "c", "r": 0},
                 {"x": 200, "y": 40, "label": "Partner", "label_pos": "r"},
                 {"x": 40, "y": 200, "label": "Vendor", "label_pos": "b"},
             ],
             "edges": [{"a": 0, "b": 1}, {"a": 0, "b": 2}]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("ppt/slides/slide1.xml").decode()
        self.assertEqual(xml.count("<p:cxnSp>"), 2)          # two edge connectors
        self.assertGreaterEqual(xml.count('prst="ellipse"'), 2)  # node dots (hub r=0 has none)
        self.assertIn("Partner", xml)                        # a node label
        self.assertEqual(warns, [])

    def test_harvey_ball_embeds_as_picture(self):
        # Rasterised (PNG) so it pixel-matches the preview — OOXML pie fills the
        # opposite side, so a native pie would mis-render in PowerPoint.
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "hb", "type": "shape", "shape": "harvey", "x": 100, "y": 100,
             "w": 20, "h": 20, "value": 0.75, "fill": "dk2"},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertTrue([n for n in z.namelist() if n.startswith("ppt/media/")])
        self.assertEqual(warns, [])

    def test_waterfall_builds_stacked_column_chart(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": [
            {"id": "wf", "type": "chart", "x": 48, "y": 80, "w": 840, "h": 300, "chart": "waterfall",
             "title": "Bridge", "categories": ["FY25", "New", "Churn", "FY26"],
             "series": [{"name": "Rev", "values": [100, 30, -10, 120]}], "totals": [0, 3]},
        ]}]}
        data, warns = build_deck_pptx(deck)
        z = zipfile.ZipFile(io.BytesIO(data))
        chart_parts = [n for n in z.namelist() if "/charts/chart" in n and n.endswith(".xml")]
        self.assertTrue(chart_parts, "no chart part for the waterfall")
        xml = z.read(chart_parts[0]).decode()
        # rendered as a stacked bar chart (grouping=stacked) with the 4 spacer series
        self.assertIn("stacked", xml)
        self.assertIn("Increase", xml)
        self.assertIn("Decrease", xml)
        self.assertEqual(warns, [])

    def test_three_slides(self):
        names = [n for n in self.zip.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml") and "rels" not in n]
        self.assertEqual(len(names), 3)

    def test_bullets(self):
        self.assertIn("a:buChar", self.s2)

    def test_bullet_char_attr_is_unqualified(self):
        # The buChar `char` attribute MUST be unqualified; `<a:buChar a:char=…>`
        # is tolerated by python-pptx/LibreOffice but makes real PowerPoint refuse
        # to open the .pptx. Regression guard for that bug.
        self.assertIn('buChar char="', self.s2)
        self.assertNotIn("buChar a:char", self.s2)

    def test_arrowhead_and_dash(self):
        self.assertIn("tailEnd", self.s2)
        self.assertIn("prstDash", self.s2)

    def test_theme_color_reference(self):
        self.assertIn("schemeClr", self.s2)

    def test_shapes_present(self):
        self.assertIn("roundRect", self.s2)  # callout box
        self.assertIn("rightArrow", self.s2)

    def test_picture_embedded(self):
        self.assertIn("p:pic", self.s2)

    def test_table_and_no_style_guid(self):
        self.assertIn("a:tbl", self.s3)
        self.assertIn("2D5ABB26-0587-4C30-8999-92F81FD0307C", self.s3)

    def test_page_number_stamped(self):
        # slide 2 is page 2; slide 1 has skip_footer so no number there.
        self.assertIn("<a:t>2</a:t>", self.s2)

    def test_skip_footer_on_title(self):
        s1 = self.zip.read("ppt/slides/slide1.xml").decode()
        self.assertNotIn("<a:t>1</a:t>", s1)


class PartialBuildTests(SimpleTestCase):
    def test_partial_build_numbers_by_full_deck_position(self):
        data, _ = build_deck_pptx(_rich_deck(), only_slide_ids=["s2"], image_resolver=_resolver)
        z = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml") and "rels" not in n]
        self.assertEqual(len(names), 1)
        # The single built slide is s2 -> page 2 in the full deck.
        self.assertIn("<a:t>2</a:t>", z.read("ppt/slides/slide1.xml").decode())


class DeadImageTests(SimpleTestCase):
    def test_missing_image_degrades_to_placeholder(self):
        data, warnings = build_deck_pptx(_rich_deck(), image_resolver=lambda t: None)
        self.assertTrue(any("image unavailable" in w for w in warnings))
        # deck still builds
        self.assertTrue(len(data) > 0)
