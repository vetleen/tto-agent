"""Tests for core.pdf — the shared PDF text+image extractor.

Fixtures are built programmatically: image PDFs via Pillow, text PDFs via a
hand-rolled minimal-but-valid PDF (avoids a reportlab dependency), and mixed
PDFs by merging the two with pypdf's writer.
"""
from __future__ import annotations

import io

from django.test import SimpleTestCase, override_settings

from core.pdf import pdf_to_text


def _text_pdf(text: str) -> bytes:
    """A minimal valid single-page PDF whose page renders *text* as selectable text."""
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        None,  # contents stream (filled below)
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    stream = b"BT /F1 18 Tf 20 100 Td (" + text.encode("latin-1") + b") Tj ET"
    objs[3] = b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj" + body + b"endobj\n"
    xref_pos = len(out)
    n = len(objs) + 1
    out += b"xref\n0 " + str(n).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode()
    out += b"trailer<</Size " + str(n).encode() + b"/Root 1 0 R>>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    return bytes(out)


def _image_pdf(*images) -> bytes:
    """A PDF with one page per PIL image (each image embedded as a raster)."""
    buf = io.BytesIO()
    first, rest = images[0], list(images[1:])
    first.save(buf, format="PDF", save_all=True, append_images=rest)
    return buf.getvalue()


def _merge(*pdf_bytes) -> bytes:
    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter()
    for b in pdf_bytes:
        writer.append(PdfReader(io.BytesIO(b)))
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _img(w, h, color=(200, 30, 30)):
    from PIL import Image

    return Image.new("RGB", (w, h), color)


def _recording_sink():
    calls = []

    def sink(image, idx):
        with image.open() as f:
            data = f.read()
        calls.append({"idx": idx, "content_type": image.content_type, "bytes": len(data)})
        return f"[[image:fake-{idx}|Image {idx}: desc]]"

    return sink, calls


class PdfToTextTests(SimpleTestCase):
    def test_extracts_page_text(self):
        sink, calls = _recording_sink()
        out = pdf_to_text(_text_pdf("Hello PDF body text"), image_sink=sink)
        self.assertIn("Hello PDF body text", out)
        self.assertEqual(calls, [])
        self.assertNotIn("[[image:", out)

    def test_image_only_pdf_yields_token(self):
        sink, calls = _recording_sink()
        out = pdf_to_text(_image_pdf(_img(120, 80)), image_sink=sink)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["idx"], 1)
        self.assertIn("[[image:fake-1|", out)

    def test_content_type_derived(self):
        sink, calls = _recording_sink()
        pdf_to_text(_image_pdf(_img(120, 80)), image_sink=sink)
        # Pillow embeds an RGB raster as JPEG inside the PDF.
        self.assertEqual(calls[0]["content_type"], "image/jpeg")

    def test_repeated_image_deduped(self):
        """The same image on two pages → sink invoked once, token reused."""
        same = _img(120, 80)
        sink, calls = _recording_sink()
        out = pdf_to_text(_image_pdf(same, same), image_sink=sink)
        self.assertEqual(len(calls), 1, "identical image should hit the sink once")
        self.assertEqual(out.count("[[image:fake-1|"), 2, "deduped token reused on both pages")

    def test_distinct_images_get_sequential_idx(self):
        sink, calls = _recording_sink()
        pdf_to_text(_image_pdf(_img(120, 80, (200, 0, 0)), _img(90, 90, (0, 0, 200))), image_sink=sink)
        self.assertEqual([c["idx"] for c in calls], [1, 2])

    def test_tiny_image_filtered(self):
        sink, calls = _recording_sink()
        out = pdf_to_text(_image_pdf(_img(10, 10)), image_sink=sink)
        self.assertEqual(calls, [], "sub-threshold image should be skipped")
        self.assertNotIn("[[image:", out)

    @override_settings(PDF_MAX_EMBEDDED_IMAGES=1)
    def test_stored_cap_respected(self):
        sink, calls = _recording_sink()
        with self.assertLogs("core.pdf", level="WARNING") as cm:
            pdf_to_text(_image_pdf(_img(120, 80, (200, 0, 0)), _img(90, 90, (0, 0, 200))), image_sink=sink)
        self.assertEqual(len(calls), 1, "cap should stop after the first distinct image")
        self.assertTrue(any("distinct embedded images" in line for line in cm.output))

    def test_multipage_text_and_image(self):
        sink, calls = _recording_sink()
        out = pdf_to_text(_merge(_text_pdf("Page one words"), _image_pdf(_img(120, 80))), image_sink=sink)
        self.assertIn("Page one words", out)
        self.assertIn("[[image:fake-1|", out)
        self.assertEqual(len(calls), 1)

    def test_sink_failure_skipped(self):
        def boom(image, idx):
            raise RuntimeError("describe blew up")

        # One bad image must not abort extraction; surrounding text survives.
        out = pdf_to_text(_merge(_text_pdf("Survives"), _image_pdf(_img(120, 80))), image_sink=boom)
        self.assertIn("Survives", out)
        self.assertNotIn("[[image:", out)

    def test_corrupt_pdf_raises_value_error(self):
        sink, _ = _recording_sink()
        with self.assertRaises(ValueError):
            pdf_to_text(b"this is not a pdf", image_sink=sink)
