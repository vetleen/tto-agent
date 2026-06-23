"""Tests for the PDF font pipeline: resolver tiers, inspection, upload, CSS.

Pure-Python — no WeasyPrint render here (those run in chat/tests/test_canvas_pdf).
"""
import io
import tempfile
from pathlib import Path
from unittest import mock

from django.test import TestCase, override_settings

from accounts.models import FontAsset, Organization
from core import fonts
from core.styles import STYLE_DEFAULTS, build_pdf_css

FONTS_DIR = Path(fonts.__file__).resolve().parent / "assets" / "fonts"


def _carlito_bytes():
    return (FONTS_DIR / "Carlito" / "Carlito-Regular.ttf").read_bytes()


def _font_with(*, fstype=0x0000, family=None):
    """Return Carlito bytes with a tweaked OS/2 fsType and/or family name."""
    from fontTools.ttLib import TTFont

    font = TTFont(io.BytesIO(_carlito_bytes()))
    font["OS/2"].fsType = fstype
    if family:
        for rec in font["name"].names:
            if rec.nameID in (1, 16):
                rec.string = family
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


class NormalizeTests(TestCase):
    def test_normalize(self):
        self.assertEqual(fonts.normalize_font_name("  Times  New   Roman "), "times new roman")
        self.assertEqual(fonts.normalize_font_name("Calibri"), "calibri")
        self.assertEqual(fonts.normalize_font_name(""), "")


class ResolverTierTests(TestCase):
    def test_substitute_metric_is_silent(self):
        res = fonts.resolve_font("Calibri")
        self.assertEqual(res.css_family, "wf-carlito")
        self.assertEqual(res.fidelity, "metric")
        self.assertEqual(res.note, "")
        self.assertEqual(len(res.faces), 4)

    def test_substitute_visual_has_note(self):
        res = fonts.resolve_font("Garamond")
        self.assertEqual(res.css_family, "wf-ebgaramond")
        self.assertEqual(res.fidelity, "visual")
        self.assertIn("Garamond", res.note)

    def test_bundled_by_name_is_exact(self):
        res = fonts.resolve_font("EB Garamond")
        self.assertEqual(res.fidelity, "exact")
        self.assertEqual(res.css_family, "wf-ebgaramond")

    def test_unknown_falls_back_and_warns(self):
        res = fonts.resolve_font("Totally Made Up Font")
        self.assertEqual(res.fidelity, "fallback")
        self.assertEqual(res.css_family, "wf-carlito")
        self.assertIn("Totally Made Up Font", res.note)

    def test_resolve_fonts_dedupes_by_name(self):
        styles = dict(STYLE_DEFAULTS)  # all Calibri
        resolved = fonts.resolve_fonts(styles, None)
        self.assertEqual(set(resolved), {"Calibri"})

    def test_face_css_embeds_data_url(self):
        css = fonts.resolve_font("Calibri").font_face_css()
        self.assertIn("@font-face", css)
        self.assertIn("src:url(data:font/ttf;base64,", css)
        self.assertIn("format('truetype')", css)


class InspectFontTests(TestCase):
    def test_inspect_bundled(self):
        meta = fonts.inspect_font(_carlito_bytes())
        self.assertEqual(meta["family"], "Carlito")
        self.assertEqual(meta["weight"], 400)
        self.assertEqual(meta["style"], "normal")
        self.assertEqual(meta["fmt"], "truetype")
        self.assertTrue(meta["embeddable"])

    def test_restricted_fstype_not_embeddable(self):
        meta = fonts.inspect_font(_font_with(fstype=0x0002))
        self.assertFalse(meta["embeddable"])

    def test_non_font_raises(self):
        with self.assertRaises(ValueError):
            fonts.inspect_font(b"this is not a font file at all")

    def test_format_detection(self):
        self.assertEqual(fonts.font_format(_carlito_bytes()), "truetype")
        self.assertEqual(fonts.font_format(b"wOF2....."), "woff2")
        self.assertEqual(fonts.font_format(b"junk"), "")


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class IngestUploadTests(TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme", slug="acme")

    def test_ingest_stores_asset(self):
        asset = fonts.ingest_uploaded_font(
            self.org, filename="Carlito-Regular.ttf", data=_carlito_bytes()
        )
        self.assertEqual(asset.organization_id, self.org.id)
        self.assertEqual(asset.family, "Carlito")
        self.assertEqual(asset.family_norm, "carlito")
        self.assertEqual(asset.source, FontAsset.SOURCE_UPLOAD)
        self.assertTrue(asset.embeddable)
        self.assertTrue(asset.blob)

    def test_ingest_dedupes_by_hash(self):
        a = fonts.ingest_uploaded_font(self.org, filename="x.ttf", data=_carlito_bytes())
        b = fonts.ingest_uploaded_font(self.org, filename="x.ttf", data=_carlito_bytes())
        self.assertEqual(a.id, b.id)
        self.assertEqual(FontAsset.objects.filter(organization=self.org).count(), 1)

    def test_restricted_font_rejected(self):
        with self.assertRaises(ValueError):
            fonts.ingest_uploaded_font(self.org, filename="x.ttf", data=_font_with(fstype=0x0002))

    def test_oversize_rejected(self):
        with self.assertRaises(ValueError):
            fonts.ingest_uploaded_font(
                self.org, filename="x.ttf", data=_carlito_bytes(), max_bytes=1000
            )

    def test_non_font_rejected(self):
        with self.assertRaises(ValueError):
            fonts.ingest_uploaded_font(self.org, filename="x.ttf", data=b"nope")

    def test_uploaded_font_resolves_tier1(self):
        # A custom-named brand font that wouldn't resolve any other way.
        data = _font_with(family="Acme Brand Sans")
        fonts.ingest_uploaded_font(self.org, filename="acme.ttf", data=data)
        res = fonts.resolve_font("Acme Brand Sans", org=self.org)
        self.assertEqual(res.fidelity, "exact")
        self.assertTrue(res.css_family.startswith("wf-org-"))
        self.assertEqual(len(res.faces), 1)

    def test_org_font_families_grouping(self):
        fonts.ingest_uploaded_font(self.org, filename="a.ttf", data=_font_with(family="Acme Brand Sans"))
        families = fonts.org_font_families(self.org)
        self.assertEqual(len(families), 1)
        self.assertEqual(families[0]["family"], "Acme Brand Sans")
        self.assertEqual(len(families[0]["faces"]), 1)


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class GoogleFetchTests(TestCase):
    GMAP = {
        "roboto slab": {
            "family": "Roboto Slab",
            "category": "serif",
            "files": {"regular": "http://x/r.ttf", "700": "http://x/b.ttf"},
        }
    }

    def test_google_fetch_caches_durably(self):
        def fake_download(url):
            # Distinct bytes per face so sha256 dedup keeps both rows.
            name = "Carlito-Bold.ttf" if "b.ttf" in url else "Carlito-Regular.ttf"
            return (FONTS_DIR / "Carlito" / name).read_bytes()

        with mock.patch.object(fonts, "_google_family_map", return_value=self.GMAP), \
             mock.patch.object(fonts, "_download", side_effect=fake_download) as dl:
            res = fonts.resolve_font("Roboto Slab", org=None)
            self.assertEqual(res.fidelity, "exact")
            self.assertEqual(res.css_family, "wf-g-roboto-slab")
            self.assertEqual(res.generic, "serif")
            self.assertEqual(dl.call_count, 2)  # regular + 700
            # Second resolve hits the durable FontAsset cache — no new downloads.
            dl.reset_mock()
            res2 = fonts.resolve_font("Roboto Slab", org=None)
            self.assertEqual(res2.fidelity, "exact")
            self.assertEqual(dl.call_count, 0)
        self.assertEqual(
            FontAsset.objects.filter(source=FontAsset.SOURCE_GOOGLE, family_norm="roboto slab").count(),
            2,
        )

    def test_google_miss_falls_back(self):
        with mock.patch.object(fonts, "_google_family_map", return_value={}):
            res = fonts.resolve_font("Roboto Slab", org=None)
        self.assertEqual(res.fidelity, "fallback")


class BuildPdfCssTests(TestCase):
    def _css(self, **overrides):
        styles = dict(STYLE_DEFAULTS)
        styles.update(overrides)
        return build_pdf_css(styles, fonts.resolve_fonts(styles, None))

    def test_defaults_embed_fonts_and_page(self):
        css = self._css()
        self.assertIn("@font-face", css)
        self.assertIn("wf-carlito", css)
        self.assertIn("wf-cousine", css)  # always present for code blocks
        self.assertIn("@page", css)
        self.assertIn("size: A4", css)

    def test_no_generic_keyword_without_embedded_backstop(self):
        # Every font-family declaration must reference an embedded wf- family.
        css = self._css(heading_font="Georgia")
        self.assertIn("'wf-gelasio'", css)

    def test_table_border_modes(self):
        self.assertIn("border:0.5pt solid", self._css(table_border_style="all"))
        self.assertIn("thead th{border-bottom", self._css(table_border_style="header"))
        css_none = self._css(table_border_style="none")
        self.assertNotIn("border:0.5pt solid", css_none)

    def test_accent_shading_and_contrast(self):
        css = self._css(accent_color="#2563EB")
        self.assertIn("background:#2563EB", css)
        self.assertIn("color:#FFFFFF", css)  # dark accent -> white text

    def test_banding(self):
        self.assertIn("nth-child(even)", self._css(table_banded=True))
        self.assertNotIn("nth-child(even)", self._css(table_banded=False))

    def test_header_footer_boxes(self):
        css = self._css(header_text="Confidential", footer_text="Draft", footer_page_numbers=True)
        self.assertIn("@top-left", css)
        self.assertIn("Confidential", css)
        self.assertIn("@bottom-left", css)
        self.assertIn("@bottom-right", css)
        self.assertIn("counter(page)", css)

    def test_page_numbers_can_be_disabled(self):
        css = self._css(header_text="", footer_text="", footer_page_numbers=False)
        self.assertNotIn("@bottom-right", css)
        self.assertNotIn("@top-left", css)
