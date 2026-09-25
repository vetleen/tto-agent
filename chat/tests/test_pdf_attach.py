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
