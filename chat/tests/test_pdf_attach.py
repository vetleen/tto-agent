"""Tests for chat.pdf_attach — lossless compression and page rendering."""

import io

from django.test import SimpleTestCase

from chat.pdf_attach import (
    compress_pdf_lossless,
    extract_pdf_pages,
    parse_page_ranges,
    render_pdf_pages_to_jpegs,
)


def _make_pdf(pages: int = 3) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class CompressPdfLosslessTests(SimpleTestCase):
    def test_returns_valid_pdf_no_larger_than_input(self):
        from pypdf import PdfReader

        original = _make_pdf(pages=2)
        out = compress_pdf_lossless(original)
        self.assertLessEqual(len(out), len(original))
        reader = PdfReader(io.BytesIO(out))
        self.assertEqual(len(reader.pages), 2)

    def test_garbage_input_returned_unchanged(self):
        garbage = b"not a pdf at all"
        self.assertEqual(compress_pdf_lossless(garbage), garbage)


class RenderPdfPagesTests(SimpleTestCase):
    def test_renders_pages_as_jpeg(self):
        from PIL import Image

        pages, total = render_pdf_pages_to_jpegs(_make_pdf(pages=3))
        self.assertEqual(total, 3)
        self.assertEqual(len(pages), 3)
        img = Image.open(io.BytesIO(pages[0]))
        self.assertEqual(img.format, "JPEG")

    def test_page_indices_renders_only_those(self):
        pages, total = render_pdf_pages_to_jpegs(_make_pdf(pages=5), page_indices=[3, 0])
        self.assertEqual(total, 5)
        self.assertEqual(len(pages), 2)

    def test_page_indices_clamped_and_capped(self):
        # Out-of-range indices dropped; result never exceeds max_pages.
        pages, total = render_pdf_pages_to_jpegs(
            _make_pdf(pages=3), page_indices=[0, 1, 99], max_pages=1
        )
        self.assertEqual(len(pages), 1)


class ParsePageRangesTests(SimpleTestCase):
    def test_range_and_singletons(self):
        self.assertEqual(parse_page_ranges("3-5,12", 20), [2, 3, 4, 11])

    def test_dedupe_preserves_order(self):
        self.assertEqual(parse_page_ranges("2,2,1", 5), [1, 0])

    def test_clamps_to_total(self):
        self.assertEqual(parse_page_ranges("4-100", 5), [3, 4])

    def test_out_of_range_and_garbage_yield_empty(self):
        self.assertEqual(parse_page_ranges("100", 5), [])
        self.assertEqual(parse_page_ranges("abc", 5), [])
        self.assertEqual(parse_page_ranges("", 5), [])

    def test_reversed_range_normalized(self):
        self.assertEqual(parse_page_ranges("5-3", 10), [2, 3, 4])


class ExtractPdfPagesTests(SimpleTestCase):
    def test_extracts_subset(self):
        from pypdf import PdfReader

        out = extract_pdf_pages(_make_pdf(pages=5), [0, 2, 4])
        self.assertEqual(len(PdfReader(io.BytesIO(out)).pages), 3)

    def test_empty_indices_returns_input(self):
        original = _make_pdf(pages=3)
        self.assertEqual(extract_pdf_pages(original, []), original)

    def test_garbage_returns_input(self):
        self.assertEqual(extract_pdf_pages(b"not a pdf", [0]), b"not a pdf")

    def test_honors_max_pages_and_reports_total(self):
        pages, total = render_pdf_pages_to_jpegs(_make_pdf(pages=5), max_pages=2)
        self.assertEqual(total, 5)
        self.assertEqual(len(pages), 2)

    def test_honors_b64_budget(self):
        # Render once to learn a page's size, then set a budget for ~1 page.
        all_pages, _ = render_pdf_pages_to_jpegs(_make_pdf(pages=3))
        one_page_b64 = (len(all_pages[0]) * 4) // 3 + 4
        pages, total = render_pdf_pages_to_jpegs(
            _make_pdf(pages=3), b64_budget=one_page_b64 + 10,
        )
        self.assertEqual(total, 3)
        self.assertEqual(len(pages), 1)

    def test_zero_budget_renders_nothing(self):
        pages, total = render_pdf_pages_to_jpegs(_make_pdf(pages=2), b64_budget=0)
        self.assertEqual(pages, [])
        self.assertEqual(total, 2)

    def test_garbage_input_returns_empty(self):
        pages, total = render_pdf_pages_to_jpegs(b"not a pdf")
        self.assertEqual(pages, [])
        self.assertEqual(total, 0)


# ---------------------------------------------------------------------------
# Vector-graphic page detection + rendered page images for models whose PDF
# input only images raster-bearing pages.
# ---------------------------------------------------------------------------

def _html_pdf(body: str, pages: int = 1) -> bytes:
    from weasyprint import HTML

    section = f"<section style='page-break-after:always'>{body}</section>"
    return HTML(string=f"<html><body style='font-family:sans-serif'>{section * pages}</body></html>").write_pdf()


def _png_uri(px: int) -> str:
    import base64

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (px, px), (120, 120, 120)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_BARS = "".join(
    f'<rect x="{40 + i * 60}" y="{200 - h}" width="40" height="{h}" fill="{c}"/>'
    for i, (h, c) in enumerate([(80, "#3060d0"), (150, "#d06030"), (110, "#30a060"), (170, "#a030c0")])
)
_CHART = f'<h1>Sales</h1><svg width="320" height="240"><line x1="30" y1="200" x2="300" y2="200" stroke="black"/>{_BARS}</svg>'
_FLOW = (
    '<svg width="420" height="120"><rect x="10" y="30" width="100" height="50" fill="#dde" stroke="#333"/>'
    '<rect x="160" y="30" width="100" height="50" fill="#ede" stroke="#333"/>'
    '<rect x="310" y="30" width="100" height="50" fill="#eed" stroke="#333"/>'
    '<path d="M110 55 L160 55 M260 55 L310 55" stroke="#333" stroke-width="2"/></svg>'
)
_CIRCLE = '<p>Visual</p><div style="width:300px;height:300px;border-radius:50%;background:#e01010"></div>'


class VectorPageDetectionTests(SimpleTestCase):
    def _flagged(self, html, **kw):
        from chat.pdf_attach import pages_with_unrendered_vectors

        return pages_with_unrendered_vectors(_html_pdf(html, **kw))

    def test_text_only_page_not_flagged(self):
        self.assertEqual(self._flagged("<h1>Title</h1><p>" + "Plain text. " * 80 + "</p>"), [])

    def test_vector_bar_chart_flagged(self):
        self.assertEqual(self._flagged(_CHART), [0])

    def test_vector_flowchart_flagged(self):
        self.assertEqual(self._flagged(_FLOW), [0])

    def test_circle_drawn_as_clipped_fill_flagged(self):
        self.assertEqual(self._flagged(_CIRCLE), [0])

    def test_heading_band_behind_text_not_flagged(self):
        html = ('<div style="background:#1a3d6b;color:white;padding:12px"><h1>Report</h1></div><p>'
                + "Body text. " * 50 + "</p>")
        self.assertEqual(self._flagged(html), [])

    def test_small_vector_below_area_not_flagged(self):
        html = '<p>Note</p><svg width="40" height="40"><rect width="40" height="40" fill="red"/></svg>'
        self.assertEqual(self._flagged(html), [])

    def test_chart_with_big_raster_not_flagged(self):
        html = f"<img src='{_png_uri(40)}' style='width:250px;height:250px'>{_CHART}"
        self.assertEqual(self._flagged(html), [])

    def test_chart_with_tiny_logo_still_flagged(self):
        html = f"<img src='{_png_uri(40)}' style='width:16px;height:16px'>{_CHART}"
        self.assertEqual(self._flagged(html), [0])

    def test_page_indices_restrict_scan(self):
        from chat.pdf_attach import pages_with_unrendered_vectors

        data = _html_pdf(_CHART, pages=3)
        self.assertEqual(pages_with_unrendered_vectors(data), [0, 1, 2])
        self.assertEqual(pages_with_unrendered_vectors(data, page_indices=[2]), [2])

    def test_garbage_bytes_return_empty(self):
        from chat.pdf_attach import pages_with_unrendered_vectors

        self.assertEqual(pages_with_unrendered_vectors(b"not a pdf"), [])


class PdfInputMissesVectorsTests(SimpleTestCase):
    def test_only_openai_vision_models(self):
        from chat.pdf_attach import pdf_input_misses_vectors

        self.assertTrue(pdf_input_misses_vectors("openai/gpt-6-luna"))
        self.assertFalse(pdf_input_misses_vectors("anthropic/claude-haiku-4-5"))
        self.assertFalse(pdf_input_misses_vectors("gemini/gemini-3.5-flash-lite"))
        self.assertFalse(pdf_input_misses_vectors(None))


class AttachPdfVectorPageImagesTests(SimpleTestCase):
    """attach_pdf_to_context adds rendered images of vector pages for OpenAI models."""

    def _ctx(self, model_id):
        from llm.types.context import RunContext

        ctx = RunContext.create()
        ctx.model_id = model_id
        return ctx

    def _attach(self, ctx, data, **kw):
        from chat.pdf_attach import attach_pdf_to_context

        return attach_pdf_to_context(
            ctx, data, pathway="attachment", filename="deck.pdf",
            description="", extracted_text="", **kw,
        )

    def _deck(self):
        # Page 1 text, page 2 chart, page 3 text.
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        for html in ("<h1>Intro</h1><p>Text only.</p>", _CHART, "<h1>End</h1><p>Text only.</p>"):
            for page in PdfReader(io.BytesIO(_html_pdf(html))).pages:
                writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def test_openai_model_gets_pdf_plus_vector_page_image(self):
        ctx = self._ctx("openai/gpt-6-luna")
        outcome = self._attach(ctx, self._deck())
        self.assertEqual(outcome.representation, "native")
        self.assertEqual(outcome.rendered_pages, [2])
        kinds = [(i["kind"], i.get("media_type")) for i in ctx.pending_native_assets]
        self.assertEqual(kinds, [("pdf", None), ("image", "image/jpeg")])
        self.assertIn("page 2 (rendered)", ctx.pending_native_assets[1]["description"])
        self.assertEqual(ctx.pending_native_assets[1]["_pathway"], "attachment")

    def test_anthropic_model_gets_pdf_only(self):
        ctx = self._ctx("anthropic/claude-haiku-4-5")
        outcome = self._attach(ctx, self._deck())
        self.assertEqual(outcome.rendered_pages, [])
        self.assertEqual([i["kind"] for i in ctx.pending_native_assets], ["pdf"])

    def test_unknown_model_gets_pdf_only(self):
        ctx = self._ctx(None)
        self.assertEqual(self._attach(ctx, self._deck()).rendered_pages, [])

    def test_page_slice_keeps_document_numbering(self):
        ctx = self._ctx("openai/gpt-6-luna")
        outcome = self._attach(ctx, self._deck(), pages="2-3")
        self.assertEqual(outcome.rendered_pages, [2])

    def test_budget_refusing_images_leaves_pdf_only(self):
        from unittest.mock import patch

        ctx = self._ctx("openai/gpt-6-luna")
        real_try = ctx.try_add_native_asset

        def pdf_only(item, pathway="dataroom"):
            return real_try(item, pathway) if item.get("kind") == "pdf" else False

        with patch.object(type(ctx), "try_add_native_asset", side_effect=lambda item, pathway="dataroom": pdf_only(item, pathway)):
            outcome = self._attach(ctx, self._deck())
        self.assertEqual(outcome.representation, "native")
        self.assertEqual(outcome.rendered_pages, [])
        self.assertEqual([i["kind"] for i in ctx.pending_native_assets], ["pdf"])
