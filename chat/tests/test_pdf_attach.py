"""Tests for chat.pdf_attach — lossless compression and page rendering."""

import io

from django.test import SimpleTestCase

from chat.pdf_attach import compress_pdf_lossless, render_pdf_pages_to_jpegs


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
