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


def _form_pdf(*, with_acroform: bool = True, appearances: bool = False) -> bytes:
    """A one-page filled-in form: printed labels in the content stream, answers
    only in AcroForm widgets (what a PDF filled in Acrobat looks like).

    Layout (PDF points, origin bottom-left)::

      Title: [Eira]                        Date of innovation/
                                           invention:
                                           [30.06.2026]
      At what stage?
      [ ] Concept   [x] Prototype
      [Tested in an RCT]                   <- comment box under the checkbox row
      Name:                 Role:
      [Ola Nordmann]        [Supervisor]
      [            ]                       <- empty field, not listed
    """
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import (
        ArrayObject, BooleanObject, DictionaryObject, FloatObject, NameObject, NumberObject, TextStringObject,
    )

    lines = [
        (20, 262, "Title:"), (290, 272, "Date of innovation/"), (290, 260, "invention:"),
        (20, 212, "At what stage?"), (34, 192, "Concept"), (114, 192, "Prototype"),
        (20, 122, "Name:"), (200, 122, "Role:"),
    ]
    stream = b"".join(
        b"BT /F1 10 Tf %d %d Td (%s) Tj ET\n" % (x, y, t.encode("latin-1")) for x, y, t in lines
    )
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 300]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length " + str(len(stream)).encode() + b">>stream\n" + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj" + body + b"endobj\n"
    xref_pos = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(("%010d 00000 n \n" % o).encode() for o in offsets)
    out += b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"

    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(bytes(out))))
    page = writer.pages[0]
    fields = [
        ("title", "/Tx", (70, 257, 270, 274), "Eira", None),
        ("date", "/Tx", (290, 235, 390, 252), "30.06.2026", None),
        ("cb_concept", "/Btn", (20, 190, 30, 200), None, "/Off"),
        ("cb_prototype", "/Btn", (100, 190, 110, 200), None, "/Yes"),
        ("comment", "/Tx", (20, 150, 380, 180), "Tested in an RCT", None),
        ("name1", "/Tx", (20, 97, 180, 114), "Ola Nordmann", None),
        ("role1", "/Tx", (200, 97, 380, 114), "Supervisor", None),
        ("name2", "/Tx", (20, 77, 180, 94), "", None),
    ]
    refs = ArrayObject()
    for name, ft, rect, value, state in fields:
        widget = DictionaryObject({
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Widget"),
            NameObject("/FT"): NameObject(ft),
            NameObject("/T"): TextStringObject(name),
            NameObject("/Rect"): ArrayObject([FloatObject(c) for c in rect]),
            NameObject("/F"): NumberObject(4),
            NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g"),
        })
        if ft == "/Tx":
            widget[NameObject("/V")] = TextStringObject(value)
        else:
            widget[NameObject("/V")] = NameObject(state)
            widget[NameObject("/AS")] = NameObject(state)
        refs.append(writer._add_object(widget))
    page[NameObject("/Annots")] = refs
    if with_acroform:
        writer._root_object[NameObject("/AcroForm")] = writer._add_object(DictionaryObject({
            NameObject("/Fields"): ArrayObject(list(refs)),
            NameObject("/NeedAppearances"): BooleanObject(True),
            NameObject("/DA"): TextStringObject("/Helv 10 Tf 0 g"),
        }))
        if appearances:
            # Generate /AP streams, as a real form filler does.
            writer.update_page_form_field_values(
                page, {f[0]: f[3] for f in fields if f[1] == "/Tx"}, auto_regenerate=False,
            )
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class PdfFormFieldTests(SimpleTestCase):
    """Filled-in AcroForm answers live in widgets, not the page text — they must
    be extracted, labelled from the printed text next to them."""

    def _block(self, out: str) -> str:
        from core.pdf import FORM_BLOCK_HEADER

        self.assertIn(FORM_BLOCK_HEADER, out)
        return out[out.index(FORM_BLOCK_HEADER):]

    def test_plain_extract_text_misses_the_answers(self):
        # Control: this is the bug — pypdf's page text has only the labels.
        from pypdf import PdfReader

        text = PdfReader(io.BytesIO(_form_pdf())).pages[0].extract_text()
        self.assertIn("Title:", text)
        self.assertNotIn("Eira", text)

    def test_values_extracted_with_geometric_labels(self):
        sink, _ = _recording_sink()
        block = self._block(pdf_to_text(_form_pdf(), image_sink=sink))
        self.assertIn("- Title: Eira", block)
        # A label wrapped over two lines is re-joined.
        self.assertIn("- Date of innovation/invention: 30.06.2026", block)
        # Checkbox row: every option with its mark, prefixed with the question.
        self.assertIn("- At what stage?: [ ] Concept · [x] Prototype", block)
        # A comment box under the row belongs to the checked option.
        self.assertIn("- Prototype: Tested in an RCT", block)
        # Column headers label the fields below them.
        self.assertIn("- Name: Ola Nordmann", block)
        self.assertIn("- Role: Supervisor", block)

    def test_reading_order_and_empty_fields_skipped(self):
        sink, _ = _recording_sink()
        block = self._block(pdf_to_text(_form_pdf(), image_sink=sink))
        order = ["Eira", "30.06.2026", "[x] Prototype", "Tested in an RCT", "Ola Nordmann", "Supervisor"]
        positions = [block.index(s) for s in order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(block.count("\n- "), 6)

    def test_block_follows_page_text_and_markers(self):
        sink, _ = _recording_sink()
        out = pdf_to_text(_form_pdf(), image_sink=sink, page_markers=True)
        self.assertTrue(out.startswith(page_marker(1)))
        self.assertLess(out.index("Title:"), out.index("- Title: Eira"))

    def test_label_lookup_failure_falls_back_to_field_names(self):
        from unittest import mock

        sink, _ = _recording_sink()
        with mock.patch("core.pdf._PageLines", side_effect=RuntimeError("boom")):
            block = self._block(pdf_to_text(_form_pdf(), image_sink=sink))
        self.assertIn("- title: Eira", block)
        self.assertIn("- comment: Tested in an RCT", block)

    def test_no_acroform_no_block(self):
        from core.pdf import FORM_BLOCK_HEADER

        sink, _ = _recording_sink()
        self.assertNotIn(FORM_BLOCK_HEADER, pdf_to_text(_text_pdf("Plain text"), image_sink=sink))
        self.assertNotIn(FORM_BLOCK_HEADER, pdf_to_text(_form_pdf(with_acroform=False), image_sink=sink))

    def test_form_field_blocks_for_text_only_paths(self):
        from core.pdf import form_field_blocks

        blocks = form_field_blocks(_form_pdf())
        self.assertEqual(list(blocks), [0])
        self.assertIn("- Title: Eira", blocks[0])
        self.assertEqual(form_field_blocks(_form_pdf(), [5]), {})
        self.assertEqual(form_field_blocks(b"not a pdf"), {})

    def test_text_only_chat_extractors_include_answers(self):
        from chat.pdf_attach import extract_pdf_pages_text
        from chat.services import extract_pdf_text

        self.assertIn("- Title: Eira", extract_pdf_text(_form_pdf()))
        [(page_no, text)] = extract_pdf_pages_text(_form_pdf(), [0])
        self.assertEqual(page_no, 1)
        self.assertIn("- Name: Ola Nordmann", text)


class PdfFormRenderTests(SimpleTestCase):
    """Page images (the non-native PDF fallback) and page slices must show the
    filled-in values, not an empty template."""

    @staticmethod
    def _ink(jpeg: bytes, box) -> int:
        """Dark pixels inside *box* (PDF points on the 400x300 form page)."""
        from PIL import Image

        img = Image.open(io.BytesIO(jpeg)).convert("L")
        sx, sy = img.width / 400, img.height / 300
        x1, y1, x2, y2 = box
        crop = img.crop((int(x1 * sx), int((300 - y2) * sy), int(x2 * sx), int((300 - y1) * sy)))
        return sum(1 for p in crop.getdata() if p < 128)

    def test_rendered_pages_draw_form_fields(self):
        from chat.pdf_attach import render_pdf_pages_to_jpegs

        [jpeg], total = render_pdf_pages_to_jpegs(_form_pdf(appearances=True), max_pages=1)
        self.assertEqual(total, 1)
        self.assertGreater(self._ink(jpeg, (72, 259, 268, 272)), 0)  # "Eira" inside the title field

    def test_page_slice_keeps_the_form(self):
        from pypdf import PdfReader

        from chat.pdf_attach import extract_pdf_pages
        from core.pdf import form_field_blocks

        doc = _merge(_text_pdf("Cover page"), _form_pdf(appearances=True))
        sliced = extract_pdf_pages(doc, [1])
        reader = PdfReader(io.BytesIO(sliced))
        self.assertEqual(len(reader.pages), 1)
        self.assertIn("/AcroForm", reader.trailer["/Root"])
        self.assertIn("- Title: Eira", form_field_blocks(sliced)[0])
