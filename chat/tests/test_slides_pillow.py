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
