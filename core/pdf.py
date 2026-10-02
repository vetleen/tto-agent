"""Single PDF -> text+image extractor shared across data rooms and chat.

Mirrors :mod:`core.docx`. PDFs are extracted to text page-by-page; each page's
embedded raster images are pulled out (pypdf ``page.images``) and rendered
inline via the same ``image_sink(image, idx) -> str`` callback the docx
converter uses — so the existing sinks (asset-persisting, describe-only,
placeholder) work unchanged. pypdf gives no positional layout for images, so a
page's image tokens are appended after that page's text, keeping each image in
its page context for retrieval.

Embedded images are handled as figures, and a simple *scanned* page's single
full-page image passes through as-is. A **layered scan** is different: MRC
("compact PDF") scanners store a page as a low-res page-sized background plus
many 1-bit stencil masks (one per text block, stamp, signature), so no single
image is the page. Such pages (``_is_layered_scan``) are rendered once with
PDFium and handed to the sink as ONE page picture instead of their fragments —
the describer's transcription-first prompt then effectively OCRs the page.

Filled-in form fields (AcroForm widgets) are not part of the page text; each
page's filled values are appended after its text as a labelled
``Form fields (filled-in values):`` list (see ``_FormFieldReader``).
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Skip images whose smaller side is below this many pixels — spacers, hairline
# rules and mask slivers that carry no information but would burn vision calls.
PDF_MIN_IMAGE_DIMENSION = 32
# Also skip images with fewer raw bytes than this (decode-free guard for images
# whose dimensions can't be determined).
PDF_MIN_IMAGE_BYTES = 1024
# Hard cap on how many distinct embedded images are stored per PDF, so a
# pathological deck can't fan out into thousands of assets. Beyond this, images
# are dropped (logged once) — described-image caps live in the sinks themselves.
PDF_MAX_EMBEDDED_IMAGES = 200

# Layered-scan detection (see ``_is_layered_scan``). A page with fewer extracted
# characters than this counts as having no text.
PDF_SCAN_TEXT_MAX_CHARS = 20
# A background image matches the page when its aspect ratio is within this
# relative tolerance of the mediabox's and its long edge is at least this big.
_SCAN_ASPECT_TOLERANCE = 0.03
_SCAN_BACKGROUND_MIN_LONG_EDGE = 500
# Long edge of a rendered layered-scan page, in pixels — the vision sweet spot
# (mirrors chat.pdf_attach.PDF_RENDER_MAX_DIMENSION; core must not import chat).
PDF_SCAN_RENDER_MAX_DIMENSION = 1568

# Opt-in page boundary markers (``pdf_to_text(..., page_markers=True)``): each
# page's text is prefixed with ``page_marker(n)``, e.g. "3" for page
# 3, on its own line. The delimiters are Unicode private-use characters — not
# whitespace, not ``\w`` — so ``str.strip()``, whitespace collapsing and the
# text cleaners leave them alone until the data-room chunker turns them into
# ``source_page_start``/``source_page_end`` and strips them
# (documents.services.chunking.assign_pdf_page_numbers). Default output is
# unchanged; chat attachments and canvas imports never see a marker.
PAGE_MARK_START = ""
PAGE_MARK_END = ""


def page_marker(page_no: int) -> str:
    """The boundary marker for 1-based page ``page_no``."""
    return f"{PAGE_MARK_START}{page_no}{PAGE_MARK_END}"


# Extension -> MIME fallback when PIL can't report a format.
_EXT_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


class _PdfImage:
    """Adapter giving an extracted PDF image the same shape an ``image_sink``
    expects from a mammoth image: ``.content_type``, ``.alt_text``, ``.open()``."""

    def __init__(self, data: bytes, content_type: str, alt_text: str = ""):
        self._data = data
        self.content_type = content_type
        self.alt_text = alt_text

    @contextmanager
    def open(self):
        bio = io.BytesIO(self._data)
        try:
            yield bio
        finally:
            bio.close()


def _read_bytes(file) -> bytes:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return f.read()
    if isinstance(file, (bytes, bytearray)):
        return bytes(file)
    try:
        file.seek(0)
    except (AttributeError, OSError):
        pass
    return file.read()


def _content_type_for(image_file, pil_image) -> str:
    fmt = getattr(pil_image, "format", None)
    if fmt:
        from PIL import Image

        mime = Image.MIME.get(fmt)
        if mime:
            return mime
    name = getattr(image_file, "name", "") or ""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _EXT_MIME.get(ext, "image/png")


def _too_small(pil_image, data: bytes, min_dim: int) -> bool:
    size = getattr(pil_image, "size", None)
    if size and len(size) == 2:
        return size[0] < min_dim or size[1] < min_dim
    # Dimensions unknown — fall back to a raw-byte guard.
    return len(data) < PDF_MIN_IMAGE_BYTES


def _page_image_xobjects(page) -> list[dict]:
    """``[{"w", "h", "mask"}]`` for the page's top-level image XObjects, read
    from their dictionaries only — no stream is decoded."""
    resources = page.get("/Resources")
    resources = resources.get_object() if resources is not None else None
    xobjects = resources.get("/XObject") if resources is not None else None
    xobjects = xobjects.get_object() if xobjects is not None else None
    out: list[dict] = []
    for name in xobjects or ():
        obj = xobjects[name].get_object()
        if obj.get("/Subtype") != "/Image":
            continue
        out.append({
            "w": int(obj.get("/Width") or 0),
            "h": int(obj.get("/Height") or 0),
            "mask": bool(obj.get("/ImageMask", False)),
        })
    return out


def _is_layered_scan(page, text: str, min_dim: int) -> bool:
    """True when the page is built from image layers that only make sense
    together, so it should be rendered as one picture:

    (a) MRC scan — a page-aspect background image plus at least one 1-bit
        stencil mask (holds even when the scanner added an OCR text layer);
    (b) textless collage — no meaningful text and two or more images.

    A textless page with ONE image (a plain scan) is not layered: that image is
    already the page. Top-level XObjects only; any failure means "not layered".
    """
    try:
        images = [i for i in _page_image_xobjects(page) if min(i["w"], i["h"]) >= min_dim]
        if not images:
            return False
        if len(text.strip()) < PDF_SCAN_TEXT_MAX_CHARS and len(images) >= 2:
            return True
        if not any(i["mask"] for i in images):
            return False
        box = page.mediabox
        page_w, page_h = abs(float(box.width)), abs(float(box.height))
        if not page_w or not page_h:
            return False
        # Either orientation: a /Rotate'd page may carry a sideways background.
        aspects = (page_w / page_h, page_h / page_w)
        for i in images:
            if i["mask"] or max(i["w"], i["h"]) < _SCAN_BACKGROUND_MIN_LONG_EDGE:
                continue
            img_aspect = i["w"] / i["h"]
            if any(abs(img_aspect - a) <= _SCAN_ASPECT_TOLERANCE * a for a in aspects):
                return True
        return False
    except Exception:  # noqa: BLE001 — detection is an optimisation; never fail extraction
        logger.debug("pdf: layered-scan detection failed on a page", exc_info=True)
        return False


def _release_decoded_streams(page, image_refs) -> None:
    """Drop pypdf's per-object decoded-data cache for a page we are done with.

    ``EncodedStreamObject.get_data()`` stores the decoded bytes on the object
    (``decoded_self``) for the reader's lifetime. ``page.images`` decodes every
    embedded raster, so without this each Flate-encoded figure stays resident
    until the whole document is extracted — measured at ~240 MB live for four
    concurrent figure-heavy papers on the worker (2026-09-08), multiplied by
    concurrency straight into R14/R15. Page content streams get the same
    treatment. ``image_refs`` are the images' ``indirect_reference`` objects
    (collected while iterating, so nothing is decoded a second time).
    Best-effort: pypdf internals, so every step is guarded.
    """
    for ref in image_refs or ():
        if ref is None:
            continue
        try:
            obj = ref.get_object()
        except Exception:  # noqa: BLE001 — pypdf internals; never fail extraction
            continue
        if getattr(obj, "decoded_self", None) is not None:
            obj.decoded_self = None
    try:
        contents = page.get_contents()
    except Exception:  # noqa: BLE001
        return
    streams = contents if isinstance(contents, (list, tuple)) else [contents]
    for stream in streams:
        try:
            obj = stream.get_object() if hasattr(stream, "get_object") else stream
        except Exception:  # noqa: BLE001
            continue
        if getattr(obj, "decoded_self", None) is not None:
            obj.decoded_self = None


# ── Filled-in form fields (AcroForm) ─────────────────────────────────────────
#
# A filled PDF form keeps its answers in widget annotations, not in the page
# content stream — ``page.extract_text()`` returns only the printed labels
# ("Name:  E-mail:  Phone:"), so the answers are invisible to search and to the
# model. ``_FormFieldReader`` reads each page's filled widgets (pypdf) and labels
# them from the printed text next to them (PDFium character boxes: the text to
# the left on the same line, else the nearest line above; a checkbox's label is
# the text to its right). Field names are usually generic ("Text Field 14"), so
# geometry is the only reliable label; the tooltip (/TU) or name is the fallback.

FORM_BLOCK_HEADER = "Form fields (filled-in values):"
# Labels further than this above a field are not its label.
_FORM_LABEL_MAX_GAP = 120.0
# A left-hand gap narrower than this (to the previous field on the line) can't
# hold a real label — look above the field instead.
_FORM_LEFT_LABEL_MIN_GAP = 40.0
_FORM_MAX_LABEL_CHARS = 160
# Button field flags (PDF 32000-1 §12.7.4.2): push buttons carry no value.
_FF_PUSHBUTTON = 1 << 16


def _ws(text: str) -> str:
    return " ".join((text or "").split())


def _widget_field(annot):
    """The field dictionary a widget belongs to (itself when merged, else the
    nearest ancestor carrying /FT)."""
    node = annot
    for _ in range(8):
        if node is None:
            return None
        if "/FT" in node:
            return node
        parent = node.get("/Parent")
        node = parent.get_object() if parent is not None else None
    return None


def _inherited(field, key):
    node = field
    for _ in range(8):
        if node is None:
            return None
        if key in node:
            return node[key]
        parent = node.get("/Parent")
        node = parent.get_object() if parent is not None else None
    return None


def _filled_widgets(page) -> list[dict]:
    """Filled (non-empty / checked) form widgets on a pypdf page, with their
    rects. Unfilled widgets are kept too (``filled=False``) — they bound the
    label search of their neighbours."""
    try:
        annots = page.get("/Annots")
        annots = annots.get_object() if annots is not None else None
    except Exception:  # noqa: BLE001 — malformed annots never fail extraction
        return []
    if not annots:
        return []
    out = []
    for ref in annots:
        try:
            annot = ref.get_object()
            if annot.get("/Subtype") != "/Widget":
                continue
            field = _widget_field(annot)
            if field is None:
                continue
            ft = _inherited(field, "/FT")
            rect = [float(c) for c in annot["/Rect"]]
        except Exception:  # noqa: BLE001
            continue
        x1, x2 = sorted((rect[0], rect[2]))
        y1, y2 = sorted((rect[1], rect[3]))
        item = {
            "kind": None, "value": "", "filled": False, "rect": (x1, y1, x2, y2),
            "tooltip": _ws(str(field.get("/TU") or "")), "name": _ws(str(field.get("/T") or "")),
        }
        if ft == "/Btn":
            flags = int(_inherited(field, "/Ff") or 0)
            if flags & _FF_PUSHBUTTON:
                continue
            # Checkbox and radio state is the widget's appearance state.
            state = str(annot.get("/AS") or "/Off")
            item["kind"] = "check"
            item["filled"] = state != "/Off"
        elif ft in ("/Tx", "/Ch"):
            value = field.get("/V")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            value = str(value) if value is not None else ""
            value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()
            item["kind"] = "text"
            item["value"] = value
            item["filled"] = bool(value)
        else:  # signatures and unknown field types carry no readable value
            continue
        out.append(item)
    return out


class _PageLines:
    """A PDFium text page as lines of character boxes, for label lookup."""

    def __init__(self, pdfium_page):
        import pypdfium2.raw as pdfium_c

        textpage = pdfium_page.get_textpage()
        chars: list[tuple] = []  # (ch, left, bottom, right, top, cy, height)
        try:
            for i in range(textpage.count_chars()):
                ch = chr(pdfium_c.FPDFText_GetUnicode(textpage.raw, i))
                # Spaces/line breaks (often PDFium-generated, with no real box)
                # are dropped; word gaps are re-inserted from geometry in _text.
                if not ch.strip():
                    continue
                # Loose boxes span the font's full line height and advance
                # width; tight glyph boxes vary per letter ("o" vs "R") and
                # break both line clustering and word-gap detection.
                left, bottom, right, top = textpage.get_charbox(i, loose=True)
                height = top - bottom
                if height <= 0:
                    continue
                chars.append((ch, left, bottom, right, top, (bottom + top) / 2, height))
        finally:
            textpage.close()
        # Cluster characters into visual lines by vertical centre — the content
        # stream order can interleave runs of one line with another.
        lines: list[dict] = []
        for c in sorted(chars, key=lambda c: -c[5]):
            line = lines[-1] if lines else None
            if line is not None and abs(line["cy"] - c[5]) <= 0.5 * max(line["h"], c[6]):
                line["chars"].append(c)
            else:
                lines.append({"cy": c[5], "h": c[6], "chars": [c]})
        self.lines = []  # (cy, height, chars sorted left to right), top to bottom
        for line in lines:
            row = sorted(line["chars"], key=lambda c: c[1])
            ys = sorted(c[5] for c in row)
            hs = sorted(c[6] for c in row)
            self.lines.append((ys[len(ys) // 2], hs[len(hs) // 2], row))

    @staticmethod
    def _text(chars, xa: float, xb: float) -> str:
        out: list[str] = []
        prev = None
        for c in chars:
            if not xa <= (c[1] + c[3]) / 2 <= xb:
                continue
            if prev is not None and c[1] - prev[3] > 0.2 * c[6]:
                out.append(" ")
            out.append(c[0])
            prev = c
        return _ws("".join(out))

    def on_line(self, ya: float, yb: float, xa: float, xb: float) -> str:
        """Text on lines whose centre lies in [ya, yb], within [xa, xb] (the
        topmost matching line when several do)."""
        for cy, _h, chars in self.lines:
            if ya <= cy <= yb:
                text = self._text(chars, xa, xb)
                if text:
                    return text
        return ""

    def above(self, y: float, xa: float, xb: float, max_gap: float = _FORM_LABEL_MAX_GAP) -> tuple[float, str]:
        """``(cy, text)`` of the label above ``y`` inside [xa, xb]: the nearest
        line, joined upward with the lines it continues (a label wrapped over
        lines — "Date of innovation/" + "invention:" — whose lower lines start
        lowercase). ``(0.0, "")`` when there is none."""
        below_first = [line for line in reversed(self.lines) if y < line[0] <= y + max_gap]
        parts: list[str] = []
        found_cy = 0.0
        last_cy = last_h = None
        for cy, h, chars in below_first:
            text = self._text(chars, xa, xb)
            if not text:
                if parts:
                    break
                continue
            if last_cy is not None and cy - last_cy > 1.8 * max(h, last_h):
                break
            if not parts:
                found_cy = cy
            parts.insert(0, text)
            last_cy, last_h = cy, h
            # Keep climbing only while this line reads as a continuation.
            if not text[0].islower() or len(parts) >= 3:
                break
        label = ""
        for part in parts:
            # "innovation/" + "invention:" re-joins without a space.
            label = part if not label else label + ("" if label.endswith(("/", "-")) else " ") + part
        return found_cy, label


def _clip_label(text: str) -> str:
    text = _ws(text)
    if len(text) > _FORM_MAX_LABEL_CHARS:
        text = text[: _FORM_MAX_LABEL_CHARS - 1].rstrip() + "…"
    return text


def _form_block(widgets: list[dict], lines: _PageLines | None, page_width: float) -> str:
    """Render a page's filled fields as a Markdown list, in reading order.

    Text fields become ``- <label>: <value>``. Checkboxes on one row become one
    line with every option and ``[x]``/``[ ]`` marks (only rows with a checked
    box are listed), prefixed with the question printed above/left of the row.
    """

    def same_row(a, b) -> bool:
        return a["rect"][1] < (b["rect"][1] + b["rect"][3]) / 2 < a["rect"][3]

    def neighbours(w):
        row = [o for o in widgets if o is not w and same_row(o, w)]
        x1, _, x2, _ = w["rect"]
        left = max([o["rect"][2] for o in row if o["rect"][2] <= x1 + 1] + [0.0])
        right = min([o["rect"][0] for o in row if o["rect"][0] >= x2 - 1] + [page_width])
        return left, right

    def check_label(w) -> str:
        x1, y1, x2, y2 = w["rect"]
        label = ""
        if lines is not None:
            _, right = neighbours(w)
            label = lines.on_line(y1 - 3, y2 + 3, x2, right)
        return _clip_label(label or w["tooltip"] or w["name"] or "Option")

    def text_label(w) -> str:
        x1, y1, x2, y2 = w["rect"]
        label = ""
        if lines is not None:
            left, _ = neighbours(w)
            if left == 0.0 or x1 - left >= _FORM_LEFT_LABEL_MIN_GAP:
                label = lines.on_line(y1, y2, left, x1)
            if not label:
                label_cy, label = lines.above(y2, x1 - 2, x2 + 2)
                # A comment box under a checkbox row ("No  [x] Yes (comment):")
                # belongs to the checked option, not to the whole printed line.
                row_above = [
                    c for c in widgets
                    if c["kind"] == "check" and c["rect"][1] >= y2 - 1 and c["rect"][1] - y2 <= _FORM_LABEL_MAX_GAP
                    and x1 - 2 <= c["rect"][0] <= x2 + 2
                ]
                if row_above:
                    nearest = min(c["rect"][1] for c in row_above)
                    if not label or label_cy >= nearest - 3:
                        row = [c for c in row_above if c["rect"][1] <= nearest + 3]
                        checked = [c for c in row if c["filled"]]
                        if checked:
                            label = " / ".join(check_label(c) for c in checked)
        return _clip_label(label or w["tooltip"] or w["name"] or "Field")

    def item(label: str, value: str) -> str:
        label = label.rstrip().rstrip(":").rstrip()
        return f"- {label}: {value}" if label else f"- {value}"

    entries: list[tuple[float, float, str]] = []  # (top, x, markdown)
    checks = [w for w in widgets if w["kind"] == "check"]
    done: set[int] = set()
    for w in sorted(checks, key=lambda c: (-c["rect"][3], c["rect"][0])):
        if id(w) in done:
            continue
        row = sorted([c for c in checks if c is w or same_row(c, w)], key=lambda c: c["rect"][0])
        done.update(id(c) for c in row)
        if not any(c["filled"] for c in row):
            continue
        options = " · ".join(f"[{'x' if c['filled'] else ' '}] {check_label(c)}" for c in row)
        question = ""
        if lines is not None and len(row) > 1:
            first = row[0]["rect"]
            before = lines.on_line(first[1] - 3, first[3] + 3, 0.0, first[0])
            _, prior = lines.above(max(c["rect"][3] for c in row), 0.0, page_width, max_gap=30.0)
            question = _clip_label(" ".join(t for t in (prior, before) if t))
        top = max(c["rect"][3] for c in row)
        entries.append((top, row[0]["rect"][0], item(question, options)))

    for w in widgets:
        if w["kind"] != "text" or not w["filled"]:
            continue
        # Multi-line values stay inside their list item; blank lines dropped.
        value = "\n  ".join(ln.strip() for ln in w["value"].split("\n") if ln.strip())
        entries.append((w["rect"][3], w["rect"][0], item(text_label(w), value)))

    if not entries:
        return ""
    # Reading order: top to bottom (rows within 4pt count as one), then left to right.
    entries.sort(key=lambda e: (-round(e[0] / 4), e[1]))
    return FORM_BLOCK_HEADER + "\n" + "\n".join(e[2] for e in entries)


class _PdfiumDoc:
    """Per-document lazy PDFium handle shared by the form-field labeller and
    the layered-scan renderer. Opened only once a page actually needs it — most
    PDFs never pay for it — and a failed open is not retried."""

    def __init__(self, source):
        # Path str, PDF bytes, or a zero-arg callable returning bytes (so an
        # in-memory PDF is only copied if PDFium is actually opened).
        self._source = source
        self._pdf = None
        self._failed = False
        self._forms_ready = False

    def _doc(self):
        if self._pdf is None and not self._failed:
            try:
                import pypdfium2 as pdfium

                source = self._source() if callable(self._source) else self._source
                self._pdf = pdfium.PdfDocument(source)
            except Exception:  # noqa: BLE001
                logger.info("pdf: PDFium could not open the file", exc_info=True)
                self._failed = True
        return self._pdf

    def page(self, index: int):
        doc = self._doc()
        if doc is None:
            return None
        try:
            return doc[index]
        except Exception:  # noqa: BLE001
            return None

    def render_page_jpeg(self, index: int) -> bytes | None:
        """The page rendered to one JPEG (long edge ≈ ``PDF_SCAN_RENDER_MAX_DIMENSION``),
        or None on any failure."""
        doc = self._doc()
        if doc is None:
            return None
        if not self._forms_ready:
            self._forms_ready = True
            try:
                # Without a form environment PDFium renders filled fields blank.
                doc.init_forms()
            except Exception:  # noqa: BLE001
                logger.info("pdf: could not initialise PDF forms; fields may render blank", exc_info=True)
        page = bitmap = None
        try:
            from django.conf import settings

            page = doc[index]
            width_pt, height_pt = page.get_size()
            long_edge_pt = max(float(width_pt or 0), float(height_pt or 0)) or 842.0
            scale = PDF_SCAN_RENDER_MAX_DIMENSION / long_edge_pt
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            if max(pil.size) > PDF_SCAN_RENDER_MAX_DIMENSION:
                pil.thumbnail((PDF_SCAN_RENDER_MAX_DIMENSION, PDF_SCAN_RENDER_MAX_DIMENSION))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=getattr(settings, "VISION_IMAGE_JPEG_QUALITY", 82))
            return buf.getvalue()
        except Exception:  # noqa: BLE001 — caller falls back to the embedded images
            logger.warning("pdf: failed to render a layered-scan page", exc_info=True)
            return None
        finally:
            for obj in (bitmap, page):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:  # noqa: BLE001
                        pass

    def close(self) -> None:
        if self._pdf is not None:
            try:
                self._pdf.close()
            except Exception:  # noqa: BLE001
                pass
            self._pdf = None


class _FormFieldReader:
    """Per-document helper: ``block(page_index, pypdf_page)`` returns the page's
    filled-field block ("" when none). PDFium (a shared :class:`_PdfiumDoc`) is
    only touched once a page actually has filled fields. Label lookup is
    best-effort: if PDFium can't read the file, fields fall back to their
    tooltip/name."""

    def __init__(self, pdfium_doc: _PdfiumDoc):
        self._pdfium = pdfium_doc

    def block(self, index: int, page) -> str:
        try:
            widgets = _filled_widgets(page)
            if not any(w["filled"] for w in widgets):
                return ""
            lines = None
            pdfium_page = self._pdfium.page(index)
            if pdfium_page is not None:
                try:
                    lines = _PageLines(pdfium_page)
                except Exception:  # noqa: BLE001
                    logger.info("pdf form fields: label lookup failed on a page", exc_info=True)
                finally:
                    pdfium_page.close()
            try:
                page_width = float(page.mediabox.width)
            except Exception:  # noqa: BLE001
                page_width = 612.0
            return _form_block(widgets, lines, page_width)
        except Exception:  # noqa: BLE001 — form fields are additive; never fail extraction
            logger.warning("pdf form fields: failed to read a page's form fields", exc_info=True)
            return ""


def _has_acroform(reader) -> bool:
    try:
        return "/AcroForm" in reader.trailer["/Root"]
    except Exception:  # noqa: BLE001
        return False


def form_field_blocks(pdf_bytes: bytes, indices=None) -> dict[int, str]:
    """``{page_index: block}`` of filled form fields for the (0-based)
    ``indices`` (all pages when None) — for text-only extractors outside
    :func:`pdf_to_text`. Any failure returns ``{}``."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        if not _has_acroform(reader):
            return {}
        pdfium_doc = _PdfiumDoc(pdf_bytes)
        forms = _FormFieldReader(pdfium_doc)
        try:
            n = len(reader.pages)
            out = {}
            for i in (range(n) if indices is None else indices):
                if 0 <= i < n:
                    block = forms.block(i, reader.pages[i])
                    if block:
                        out[i] = block
            return out
        finally:
            pdfium_doc.close()
    except Exception:  # noqa: BLE001
        logger.info("pdf form fields: could not read form fields", exc_info=True)
        return {}


def pdf_to_text(
    file, *, image_sink: Callable[[object, int], str], page_markers: bool = False,
) -> str:
    """Extract a PDF to text. ``file`` may be a path, bytes, or a binary
    file-like. For each page: ``page.extract_text()`` followed by the inline
    tokens ``image_sink`` returns for that page's embedded images.

    Images repeated across pages (logos, headers) are deduplicated by sha256 —
    the sink is invoked once and its token reused, so a recurring logo doesn't
    become N assets or N vision calls.

    ``page_markers=True`` prefixes every page — including pages with no text and
    no images, which are otherwise dropped — with ``page_marker(n)`` so a caller
    can recover page numbers after the text has been cleaned and chunked.
    """
    from django.conf import settings
    from pypdf import PdfReader

    min_dim = getattr(settings, "PDF_MIN_IMAGE_DIMENSION", PDF_MIN_IMAGE_DIMENSION)
    max_images = getattr(settings, "PDF_MAX_EMBEDDED_IMAGES", PDF_MAX_EMBEDDED_IMAGES)

    # A path is handed to pypdf as-is so it reads objects from disk on demand;
    # loading a 13 MB file into bytes plus a BytesIO copy was ~26 MB of the
    # transient peak for nothing. Bytes / file-likes still go through memory.
    if isinstance(file, (str, Path)):
        source = str(file)
    else:
        source = io.BytesIO(_read_bytes(file))
    try:
        reader = PdfReader(source)
    except Exception as exc:
        raise ValueError("This PDF file is corrupt or not a valid PDF document.") from exc

    # Filled form fields are appended after each page's text (the answers of a
    # filled-in form live in widget annotations, not in the page content).
    # One lazy PDFium handle per document, shared by the form-field labeller and
    # the layered-scan renderer; never opened for a PDF that needs neither.
    pdfium_doc = _PdfiumDoc(source if isinstance(source, str) else source.getvalue)
    forms = _FormFieldReader(pdfium_doc) if _has_acroform(reader) else None

    seen: dict[str, str] = {}  # sha256 -> token (dedup within the document)
    idx = 0  # 1-indexed count of images actually handed to the sink
    capped = False
    rendered_pages = 0
    pages_out: list[str] = []

    def emit(img_bytes: bytes, content_type: str, alt_text: str = "") -> str | None:
        """Hand one picture to the sink (sha dedup + stored cap); its token, or
        None when it was dropped."""
        nonlocal idx, capped
        sha = hashlib.sha256(img_bytes).hexdigest()
        existing = seen.get(sha)
        if existing is not None:
            return existing
        if len(seen) >= max_images:
            if not capped:
                logger.warning(
                    "pdf_to_text: more than %d distinct embedded images; "
                    "storing the first %d and dropping the rest",
                    max_images, max_images,
                )
                capped = True
            return None
        idx += 1
        try:
            token = image_sink(_PdfImage(img_bytes, content_type, alt_text), idx)
        except Exception:
            logger.exception("pdf_to_text: image_sink failed for an embedded image; skipping")
            idx -= 1
            return None
        seen[sha] = token
        return token

    try:
        for page_no, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").replace("\x00", "")
            except Exception:
                logger.warning("pdf_to_text: failed to extract text from a page", exc_info=True)
                text = ""
            # Measured before the form block: filled answers are not page text.
            layered = _is_layered_scan(page, text, min_dim)
            if forms is not None:
                block = forms.block(page_no - 1, page)
                if block:
                    text = (text.rstrip() + "\n\n" + block) if text.strip() else block

            tokens: list[str] = []
            image_refs: list = []
            page_jpeg = pdfium_doc.render_page_jpeg(page_no - 1) if layered else None
            if page_jpeg is not None:
                # One picture for the whole page; its fragments are never decoded.
                rendered_pages += 1
                token = emit(page_jpeg, "image/jpeg", f"Scanned page {page_no}")
                if token is not None:
                    tokens.append(token)
                page_jpeg = None
                image_iter = iter(())
            else:
                # Iterate lazily: list(page.images) would decode every raster on
                # the page (PIL buffers included) before the first one is handled;
                # one at a time, the previous image is released as the next is decoded.
                try:
                    image_iter = iter(page.images)
                except Exception:
                    logger.warning("pdf_to_text: failed to enumerate images on a page", exc_info=True)
                    image_iter = iter(())
            while True:
                try:
                    image_file = next(image_iter)
                except StopIteration:
                    break
                except Exception:
                    logger.warning("pdf_to_text: failed to decode an embedded image; skipping", exc_info=True)
                    continue
                image_refs.append(getattr(image_file, "indirect_reference", None))
                try:
                    img_bytes = image_file.data
                    pil_image = image_file.image  # may be None / may raise
                except Exception:
                    logger.warning("pdf_to_text: failed to decode an embedded image; skipping", exc_info=True)
                    continue
                if not img_bytes or _too_small(pil_image, img_bytes, min_dim):
                    continue
                token = emit(img_bytes, _content_type_for(image_file, pil_image))
                if token is not None:
                    tokens.append(token)
            image_file = pil_image = img_bytes = None  # noqa: F841 — drop the last raster

            if tokens:
                page_str = (text + "\n\n" + "\n\n".join(tokens)).strip() if text else "\n\n".join(tokens)
            else:
                page_str = text
            if page_markers:
                # Every page gets a marker, so an empty page still advances the
                # numbering instead of silently shifting later pages.
                pages_out.append(page_marker(page_no) + ("\n\n" + page_str if page_str else ""))
            elif page_str:
                pages_out.append(page_str)

            # Free this page's decoded rasters before moving on — otherwise pypdf
            # keeps every decoded image of the document resident until we return.
            _release_decoded_streams(page, image_refs)
    finally:
        pdfium_doc.close()
        close = getattr(reader, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        # The reader's object graph is cyclic (page <-> reader), so dropping the
        # reference alone leaves tens of MB of parsed objects and raw streams to
        # the cyclic GC's discretion. Collect now, while we still hold the slot.
        del reader
        gc.collect()

    if rendered_pages:
        logger.info("pdf_to_text: rendered %d layered-scan page(s) as one picture each", rendered_pages)
    return "\n\n".join(pages_out).strip()
