"""Tests for core.pdf — the shared PDF text+image extractor.

Fixtures are built programmatically: image PDFs via Pillow, text PDFs via a
hand-rolled minimal-but-valid PDF (avoids a reportlab dependency), and mixed
PDFs by merging the two with pypdf's writer.
"""
from __future__ import annotations

import io

from django.test import SimpleTestCase, override_settings

from core.pdf import PAGE_MARK_END, PAGE_MARK_START, page_marker, pdf_to_text


def _blank_pdf() -> bytes:
    """A single-page PDF with no text and no images (pypdf blank page)."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


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


class PdfPageMarkerTests(SimpleTestCase):
    """``page_markers=True`` prefixes every page with ``page_marker(n)``; the
    default output never contains one."""

    def _three_pages(self) -> bytes:
        # text / blank / text — the blank page is what the default path drops.
        return _merge(_text_pdf("Page one words"), _blank_pdf(), _text_pdf("Page three words"))

    def test_default_output_has_no_markers_and_drops_empty_pages(self):
        sink, _ = _recording_sink()
        out = pdf_to_text(self._three_pages(), image_sink=sink)
        self.assertNotIn(PAGE_MARK_START, out)
        self.assertNotIn(PAGE_MARK_END, out)
        self.assertIn("Page one words", out)
        self.assertIn("Page three words", out)

    def test_markers_for_every_page_including_empty_ones(self):
        sink, _ = _recording_sink()
        out = pdf_to_text(self._three_pages(), image_sink=sink, page_markers=True)
        self.assertTrue(out.startswith(page_marker(1)))
        self.assertEqual(out.count(PAGE_MARK_START), 3)
        # Markers appear in page order; the empty page 2 is marker-only.
        self.assertLess(out.index(page_marker(1)), out.index(page_marker(2)))
        self.assertLess(out.index(page_marker(2)), out.index(page_marker(3)))
        between = out[out.index(page_marker(2)) + len(page_marker(2)):out.index(page_marker(3))]
        self.assertEqual(between.strip(), "")
        # Page text follows its own marker.
        self.assertLess(out.index(page_marker(1)), out.index("Page one words"))
        self.assertLess(out.index(page_marker(3)), out.index("Page three words"))
        self.assertLess(out.index("Page one words"), out.index(page_marker(2)))

    def test_image_page_marker_precedes_its_token(self):
        sink, calls = _recording_sink()
        out = pdf_to_text(_merge(_text_pdf("Words"), _image_pdf(_img(120, 80))), image_sink=sink, page_markers=True)
        self.assertEqual(len(calls), 1)
        self.assertLess(out.index(page_marker(2)), out.index("[[image:fake-1|"))


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

    def test_decoded_stream_cache_released_per_page(self):
        """pypdf caches decoded stream bytes on the object for the reader's
        lifetime; the extractor must drop that cache page by page, or every
        decoded raster of a figure-heavy document stays resident until the end."""
        from unittest import mock

        import pypdf

        pdf = _image_pdf(_img(120, 80), _img(90, 90, (20, 120, 200)))

        # Control: reading images the plain way leaves the cache populated.
        control = pypdf.PdfReader(io.BytesIO(pdf))
        for page in control.pages:
            for image_file in page.images:
                self.assertIsNotNone(image_file.indirect_reference.get_object().decoded_self)

        readers = []
        real_reader = pypdf.PdfReader

        class RecordingReader(real_reader):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                readers.append(self)

        sink, calls = _recording_sink()
        with mock.patch.object(pypdf, "PdfReader", RecordingReader):
            out = pdf_to_text(pdf, image_sink=sink)
        self.assertEqual(len(calls), 2)
        self.assertIn("[[image:fake-2|", out)
        self.assertEqual(len(readers), 1)
        for page in readers[0].pages:
            xobjects = page["/Resources"].get("/XObject") or {}
            self.assertTrue(xobjects, "fixture page should carry an image XObject")
            for name in xobjects:
                self.assertIsNone(getattr(xobjects[name].get_object(), "decoded_self", None))
            contents = page.get_contents()
            streams = contents if isinstance(contents, (list, tuple)) else [contents]
            for stream in streams:
                obj = stream.get_object() if hasattr(stream, "get_object") else stream
                self.assertIsNone(getattr(obj, "decoded_self", None))

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
