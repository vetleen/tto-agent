"""Unit tests for the v2 named-theme model in chat.slides.theme."""
from __future__ import annotations

import tempfile
from io import BytesIO
from types import SimpleNamespace

from django.core.files.base import ContentFile
from django.test import TestCase

from chat.slides import theme as T


def _fake_org(prefs):
    return SimpleNamespace(preferences=prefs)


def _pixels(png):
    from PIL import Image
    return Image.open(BytesIO(png)).convert("RGB")


class ThemeDefaultsAndFooter(TestCase):
    def test_forest_default_is_full_theme(self):
        t = T.slide_theme_defaults()
        self.assertEqual(t["id"], "forest")
        self.assertEqual(t["label"], "Forest")
        for k in ("colors", "fonts", "typography", "tables", "footer", "logo_ext"):
            self.assertIn(k, t)
        self.assertEqual(len(t["colors"]), len(T.SLIDE_STYLE_COLOR_KEYS))

    def test_builtin_themes_cover_all_presets_as_full_themes(self):
        themes = T.builtin_slide_themes()
        self.assertEqual([t["id"] for t in themes], list(T.PRESET_THEMES))
        self.assertEqual([t["label"] for t in themes],
                         [spec["label"] for spec in T.PRESET_THEMES.values()])
        for t in themes:
            self.assertTrue(T._is_full_theme_entry(t))
            for k in ("footer", "logo_ext"):
                self.assertIn(k, t)
        # The first builtin is Forest with the base palette.
        self.assertEqual(themes[0], T.slide_theme_defaults())

    def test_default_footer_is_page_number_right(self):
        f = T.default_footer()
        self.assertEqual(len(f["sections"]), T.FOOTER_SECTION_COUNT)
        self.assertEqual(f["bg_color"], "")
        self.assertEqual(f["sections"][2], {"colspan": 4, "align": "right", "content": "page"})
        self.assertTrue(all(s["content"] == "none" for s in f["sections"][:2]))


class ValidateSlideTheme(TestCase):
    def test_minimal_payload_fills_forest_defaults(self):
        clean, err = T.validate_slide_theme({"label": "  My Theme  "})
        self.assertIsNone(err)
        self.assertEqual(clean["label"], "My Theme")
        self.assertNotIn("id", clean)  # caller assigns id
        self.assertEqual(clean["footer"], T.default_footer())
        self.assertEqual(clean["colors"], T.slide_style_defaults()["colors"])

    def test_label_required(self):
        _, err = T.validate_slide_theme({"label": "   "})
        self.assertTrue(err)

    def test_footer_sections_validated(self):
        good = {"label": "x", "footer": {"bg_color": "dk2", "text": "Confidential",
                "sections": [{"colspan": 3, "align": "left", "content": "logo"},
                             {"colspan": 6, "align": "center", "content": "text"},
                             {"colspan": 3, "align": "right", "content": "page"}]}}
        clean, err = T.validate_slide_theme(good)
        self.assertIsNone(err)
        self.assertEqual(clean["footer"]["bg_color"], "dk2")
        self.assertEqual(clean["footer"]["sections"][0]["content"], "logo")

    def test_footer_hex_bg_normalized(self):
        clean, err = T.validate_slide_theme({"label": "x", "footer": {"bg_color": "#abc"}})
        self.assertIsNone(err)
        self.assertEqual(clean["footer"]["bg_color"], "#AABBCC")

    def test_footer_bad_content_rejected(self):
        _, err = T.validate_slide_theme({"label": "x", "footer": {
            "sections": [{"colspan": 4, "align": "left", "content": "bogus"},
                         {"colspan": 4, "align": "center", "content": "none"},
                         {"colspan": 4, "align": "right", "content": "page"}]}})
        self.assertTrue(err)

    def test_footer_wrong_section_count_rejected(self):
        _, err = T.validate_slide_theme({"label": "x", "footer": {
            "sections": [{"colspan": 4, "align": "left", "content": "none"}]}})
        self.assertTrue(err)

    def test_colspan_clamped(self):
        clean, _ = T.validate_slide_theme({"label": "x", "footer": {
            "sections": [{"colspan": 99, "align": "left", "content": "none"},
                         {"colspan": 0, "align": "center", "content": "none"},
                         {"colspan": 4, "align": "right", "content": "page"}]}})
        self.assertEqual(clean["footer"]["sections"][0]["colspan"], 12)
        self.assertEqual(clean["footer"]["sections"][1]["colspan"], 1)

    def test_footer_logo_height_default_and_clamped(self):
        # Default when unspecified.
        clean, _ = T.validate_slide_theme({"label": "x"})
        self.assertEqual(clean["footer"]["logo_height"], T.FOOTER_LOGO_H_DEFAULT)
        # Out-of-range values clamp to the allowed band; junk falls back to default.
        hi, _ = T.validate_slide_theme({"label": "x", "footer": {"logo_height": 999}})
        lo, _ = T.validate_slide_theme({"label": "x", "footer": {"logo_height": 1}})
        junk, _ = T.validate_slide_theme({"label": "x", "footer": {"logo_height": "big"}})
        self.assertEqual(hi["footer"]["logo_height"], T.FOOTER_LOGO_H_MAX)
        self.assertEqual(lo["footer"]["logo_height"], T.FOOTER_LOGO_H_MIN)
        self.assertEqual(junk["footer"]["logo_height"], T.FOOTER_LOGO_H_DEFAULT)

    def test_footer_logo_box_centres_and_stays_on_slide(self):
        # Default height centres within the 28pt band (historical y0+4 look).
        h, top = T.footer_logo_box(T.FOOTER_LOGO_H_DEFAULT)
        self.assertEqual(h, T.FOOTER_LOGO_H_DEFAULT)
        self.assertAlmostEqual(top, (540 - T.FOOTER_BAND_H) + (T.FOOTER_BAND_H - h) / 2.0)
        # A tall logo never spills past the slide's bottom edge.
        h2, top2 = T.footer_logo_box(T.FOOTER_LOGO_H_MAX)
        self.assertLessEqual(top2 + h2, 540 - 1)


class DeckOverride(TestCase):
    def test_override_carries_footer_and_provenance(self):
        t = T.slide_theme_defaults()
        t["id"], t["label"] = "t123", "Brand"
        ov = T.theme_to_deck_override(t)
        self.assertIn("colors", ov)
        self.assertIn("footer", ov)
        self.assertEqual(ov["_theme_id"], "t123")
        self.assertEqual(ov["_theme_label"], "Brand")

    def test_override_merges_via_resolve_theme(self):
        t = T.slide_theme_defaults()
        t["colors"]["accent1"] = "#123456"
        deck = {"theme": T.theme_to_deck_override(t)}
        resolved = T.resolve_theme(deck)
        self.assertEqual(resolved["colors"]["accent1"], "#123456")
        self.assertIn("sections", resolved["footer"])  # footer carried through


class LegacyUpgrade(TestCase):
    def test_color_only_user_theme_upgrades(self):
        legacy = {"id": "cabc", "label": "Mine", "base": "forest",
                  "colors": {"accent1": "#FF0000", "dk2": "#111111", "lt1": "#FFFFFF"}}
        up = T._upgrade_legacy_theme(legacy)
        self.assertEqual(up["id"], "cabc")
        self.assertEqual(up["label"], "Mine")
        self.assertEqual(up["colors"]["accent1"], "#FF0000")
        self.assertIn("footer", up)
        self.assertIn("fonts", up)

    def test_full_theme_passes_through_with_id(self):
        full = dict(T.slide_theme_defaults(), id="tzzz", label="Keep")
        up = T._upgrade_legacy_theme(full)
        self.assertEqual(up["id"], "tzzz")
        self.assertEqual(up["label"], "Keep")


class ScopeResolution(TestCase):
    def test_org_themes_from_new_list(self):
        org = _fake_org({"slide_themes": [
            dict(T.slide_theme_defaults(), id="torg", label="Org One"),
        ]})
        themes = T.org_slide_themes(org)
        self.assertEqual([t["id"] for t in themes], ["torg"])

    def test_org_themes_synthesized_from_legacy_single(self):
        org = _fake_org({"slide_theme": {"name": "forest"}})
        themes = T.org_slide_themes(org)
        self.assertEqual(len(themes), 1)
        self.assertEqual(themes[0]["id"], "org-legacy")
        self.assertIn("footer", themes[0])

    def test_list_available_tags_scope(self):
        org = _fake_org({"slide_themes": [dict(T.slide_theme_defaults(), id="torg", label="O")]})
        avail = T.list_available_themes(user=None, org=org)
        scopes = {t["scope"] for t in avail}
        self.assertEqual(scopes, {"builtin", "org"})
        self.assertEqual(avail[0]["id"], "forest")

    def test_list_includes_all_builtin_presets(self):
        avail = T.list_available_themes(user=None, org=_fake_org({}))
        builtin_ids = [t["id"] for t in avail if t["scope"] == "builtin"]
        self.assertEqual(builtin_ids, ["forest", "slate", "warm", "mono", "ocean"])

    def test_resolve_builtin_preset_by_id(self):
        slate = T.resolve_theme_by_id("slate", None, None)
        self.assertEqual(slate["label"], "Slate")
        self.assertEqual(slate["scope"], "builtin")
        self.assertEqual(slate["colors"]["accent1"], "#2563EB")
        # A builtin resolves into a deck override with provenance intact.
        ov = T.theme_to_deck_override(slate)
        self.assertEqual(ov["_theme_id"], "slate")
        self.assertEqual(ov["colors"]["accent1"], "#2563EB")

    def test_default_precedence_org_then_forest(self):
        org = _fake_org({
            "slide_themes": [dict(T.slide_theme_defaults(), id="torg", label="O")],
            "slide_theme_default": "torg",
        })
        self.assertEqual(T.default_theme_for(user=None, org=org)["id"], "torg")
        # No explicit default but the org has themes -> its first theme is the default.
        org2 = _fake_org({"slide_themes": [dict(T.slide_theme_defaults(), id="torg", label="O")]})
        self.assertEqual(T.default_theme_for(user=None, org=org2)["id"], "torg")
        # An org with no themes at all -> Forest.
        self.assertEqual(T.default_theme_for(user=None, org=_fake_org({}))["id"], "forest")

    def test_resolve_by_id(self):
        org = _fake_org({"slide_themes": [dict(T.slide_theme_defaults(), id="torg", label="O")]})
        self.assertEqual(T.resolve_theme_by_id("torg", None, org)["label"], "O")
        self.assertIsNone(T.resolve_theme_by_id("nope", None, org))


class FooterRender(TestCase):
    """The section-aware footer stamper (Pillow) — band, sections, and logo."""

    def _deck(self, footer, theme_id="", logo_ext=""):
        th = T.slide_theme_defaults()
        th["footer"] = footer
        if theme_id:
            th["id"] = theme_id
        th["logo_ext"] = logo_ext
        return {"version": 1, "size": {"w": 960, "h": 540},
                "theme": T.theme_to_deck_override(th),
                "slides": [{"id": "s1", "elements": []}]}

    def _render(self, deck, index=0):
        from chat.slides import pillow_render
        png, _w, _h = pillow_render.render_slide_png(deck, index, dpi=96)
        return _pixels(png)

    def test_default_footer_renders(self):
        deck = {"version": 1, "size": {"w": 960, "h": 540},
                "slides": [{"id": "s1", "elements": []}, {"id": "s2", "elements": []}]}
        im = self._render(deck, index=1)  # page 2 -> page-number section stamps
        self.assertGreater(len(im.getcolors(maxcolors=100000)), 1)  # ink beyond flat bg

    def test_band_bg_fills_bottom_edge_to_edge(self):
        footer = T.default_footer()
        footer["bg_color"] = "dk2"
        im = self._render(self._deck(footer))
        # both bottom corners are the dark band, not the light slide bg
        for x in (2, im.width - 3):
            r, g, b = im.getpixel((x, im.height - 2))
            self.assertLess((r + g + b) / 3, 120, "footer band should reach the slide edge")

    def test_logo_section_paints_logo(self):
        from PIL import Image as _Img
        buf = BytesIO()
        _Img.new("RGB", (200, 60), (220, 20, 20)).save(buf, "PNG")
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            from chat.slides.logos import save_slide_theme_logo
            save_slide_theme_logo("tlogo", ContentFile(buf.getvalue()), "png")
            footer = {"bg_color": "", "text": "", "size": 9, "color": "dk2", "sections": [
                {"colspan": 4, "align": "left", "content": "logo"},
                {"colspan": 4, "align": "center", "content": "none"},
                {"colspan": 4, "align": "right", "content": "page"}]}
            im = self._render(self._deck(footer, theme_id="tlogo", logo_ext="png"))
        reds = [c for c in im.getcolors(maxcolors=100000) if c[1][0] > 150 and c[1][1] < 90 and c[1][2] < 90]
        self.assertTrue(reds, "the footer logo (red block) should be painted in the band")


class SlideCustomFonts(TestCase):
    """Org-uploaded custom fonts in slide themes: validation, resolution, render."""

    def _org_with_font(self, family="Acme Brand"):
        import uuid as _uuid

        from accounts.models import FontAsset, Organization
        from core.fonts import bundled_resolution, normalize_font_name

        org = Organization.objects.create(name="Acme", slug=f"acme-{_uuid.uuid4().hex[:6]}")
        data = bundled_resolution("Caladea").faces[0].data  # real embeddable TTF bytes
        fa = FontAsset(
            organization=org, family=family, family_norm=normalize_font_name(family),
            source=FontAsset.SOURCE_UPLOAD, weight=400, style="normal",
            font_format="truetype", embeddable=True,
        )
        fa.blob.save("f.ttf", ContentFile(data), save=True)
        return org

    def test_save_gate_accepts_uploaded_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            org = self._org_with_font("Acme Brand")
            fonts = {"headline": "Acme Brand", "subhead": "Caladea", "body": "Carlito", "data": "Carlito"}
            clean, err = T.validate_slide_theme({"label": "X", "fonts": fonts}, org=org)
            self.assertIsNone(err)
            self.assertEqual(clean["fonts"]["headline"], "Acme Brand")
            _, err2 = T.validate_slide_theme({"label": "X", "fonts": dict(fonts, headline="Totally Missing")}, org=org)
            self.assertTrue(err2)
        # bundled-only save still works without an org
        _, err3 = T.validate_slide_theme({"label": "X"}, org=None)
        self.assertIsNone(err3)

    def test_override_preserves_uploaded_family(self):
        th = dict(T.slide_theme_defaults())
        th["fonts"] = dict(th["fonts"], headline="Acme Brand")
        self.assertEqual(T.theme_to_deck_override(th)["fonts"]["headline"], "Acme Brand")

    def test_read_path_keeps_theme_with_removed_font(self):
        entry = dict(T.slide_theme_defaults(), id="t1")
        entry["fonts"] = dict(entry["fonts"], headline="Ghost Font")
        out = T._upgrade_legacy_theme(entry)
        self.assertIsNotNone(out)
        self.assertEqual(out["fonts"]["headline"], "Ghost Font")

    def test_resolve_faces_and_fontbook(self):
        from chat.slides import pillow_render
        from chat.slides.font_resolve import is_bundled_family, resolve_deck_font_faces

        with tempfile.TemporaryDirectory() as d, self.settings(MEDIA_ROOT=d):
            org = self._org_with_font("Acme Brand")
            theme = {"fonts": {"headline": "Acme Brand", "subhead": "Caladea", "body": "Carlito", "data": "Carlito"}}
            faces = resolve_deck_font_faces(theme, org)
            self.assertTrue(faces["Acme Brand"]["faces"])
            self.assertFalse(faces["Acme Brand"]["bundled"])
            self.assertTrue(faces["Caladea"]["bundled"])
            self.assertTrue(is_bundled_family("Caladea"))
            self.assertFalse(is_bundled_family("Acme Brand"))
            # the FontBook builds a PIL face from the uploaded bytes
            book = pillow_render._FontBook(faces)
            self.assertIsNotNone(book.pil_font("Acme Brand", 24, False, False))
            # and the render entry accepts the face map without error
            deck = {"version": 1, "size": {"w": 960, "h": 540}, "slides": [{"id": "s1", "elements": []}]}
            self.assertEqual(len(pillow_render.render_deck_pngs(deck, dpi=96, font_faces=faces)), 1)
