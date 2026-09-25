"""Helpers for attaching large PDFs to a run within the native-asset budget.

``document_view_native`` used to base64 whole PDFs straight into
``pending_native_assets`` — a 30MB file becomes a ~40MB string held in the
message history for the entire run. These helpers implement the staged
degrade chain for files that would bust ``RunContext``'s per-run budget:

1. lossless pypdf recompression (cheap; often enough for text-born PDFs), then
2. rendering the first N pages to modest JPEGs via pypdfium2 (deterministic
   byte control; the same rasterizer the slide renderer uses), with the
   document's extracted text carried alongside.

Both fall back gracefully — any failure returns the input / no pages, and the
caller degrades to extracted text only.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PDF_ATTACH_MAX_RENDER_PAGES = 20
PDF_RENDER_SCALE = 1.5  # 72dpi * 1.5 = ~108 DPI, readable for vision models
PDF_RENDER_JPEG_QUALITY = 70
PDF_RENDER_MAX_DIMENSION = 1568  # longest side; matches vision-model sweet spot


def pdf_page_count(pdf_bytes: bytes) -> int:
    """Return the number of pages in a PDF, or 0 if it can't be opened.

    Cheap (opens the document, reads the page count, closes it — no rendering),
    so it's safe to call once per PDF at ingest to enforce a page cap before a
    native attach. Any failure returns 0, which callers treat as "unknown /
    don't block".
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 unavailable; cannot count PDF pages")
        return 0
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception:
        logger.info("Could not open PDF to count pages", exc_info=True)
        return 0
    try:
        return len(doc)
    finally:
        doc.close()


def compress_pdf_lossless(pdf_bytes: bytes) -> bytes:
    """Losslessly recompress a PDF; returns the smaller of input/output.

    Any failure (corrupt file, encryption, pypdf edge case) returns the input
    unchanged — this is an opportunistic size reduction, never a gate.
    """
    try:
        from pypdf import PdfWriter

        writer = PdfWriter(clone_from=io.BytesIO(pdf_bytes))
        for page in writer.pages:
            page.compress_content_streams()
        writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
        buf = io.BytesIO()
        writer.write(buf)
        out = buf.getvalue()
        if out and len(out) < len(pdf_bytes):
            return out
    except Exception:
        logger.info("Lossless PDF compression failed; using original bytes", exc_info=True)
    return pdf_bytes


def render_pdf_pages_to_jpegs(
    pdf_bytes: bytes,
    *,
    max_pages: int = PDF_ATTACH_MAX_RENDER_PAGES,
    scale: float = PDF_RENDER_SCALE,
    quality: int = PDF_RENDER_JPEG_QUALITY,
    b64_budget: int | None = None,
    page_indices: list[int] | None = None,
) -> tuple[list[bytes], int]:
    """Render pages of a PDF to JPEG bytes.

    Renders the first ``max_pages`` pages by default, or the specific 0-based
    ``page_indices`` when given (still capped at ``max_pages``). Stops when the
    cumulative base64 size would exceed ``b64_budget``. Returns
    ``(jpeg_pages, total_pages)``; on any failure returns ``([], 0)`` so the
    caller degrades to text.
    """
    try:
        import pypdfium2 as pdfium
        from PIL import Image  # noqa: F401  (bitmap.to_pil needs Pillow)
    except ImportError:
        logger.warning("pypdfium2/Pillow unavailable; cannot render PDF pages")
        return [], 0

    pages: list[bytes] = []
    b64_used = 0
    total = 0
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception:
        logger.info("Could not open PDF for page rendering", exc_info=True)
        return [], 0
    try:
        total = len(doc)
        if page_indices is None:
            targets = list(range(min(total, max_pages)))
        else:
            targets = [i for i in page_indices if 0 <= i < total][:max_pages]
        for i in targets:
            page = doc[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
            bitmap.close()
            page.close()
            if max(pil.width, pil.height) > PDF_RENDER_MAX_DIMENSION:
                pil.thumbnail((PDF_RENDER_MAX_DIMENSION, PDF_RENDER_MAX_DIMENSION))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=quality)
            jpeg = buf.getvalue()
            # base64 inflates 4/3; +4 covers padding
            b64_size = (len(jpeg) * 4) // 3 + 4
            if b64_budget is not None and b64_used + b64_size > b64_budget:
                break
            b64_used += b64_size
            pages.append(jpeg)
    except Exception:
        logger.info("PDF page rendering failed part-way", exc_info=True)
    finally:
        doc.close()
    return pages, total


def parse_page_ranges(spec: str, total: int) -> list[int]:
    """Parse a 1-based human page spec like ``"3-5,12"`` into 0-based indices.

    Clamps to ``[1, total]``, dedupes preserving first-seen order, and silently
    drops malformed tokens. Returns ``[]`` when nothing valid is found (callers
    then fall back to the default first-N behavior).
    """
    if not spec or total <= 0:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for token in str(spec).replace(" ", "").split(","):
        if not token:
            continue
        try:
            if "-" in token:
                a, b = token.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(token)
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        for human in range(max(1, start), min(total, end) + 1):
            idx = human - 1
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def extract_pdf_pages(pdf_bytes: bytes, indices: list[int]) -> bytes:
    """Return a new PDF containing only ``indices`` (0-based), text preserved.

    Uses pypdf to clone the selected pages into a fresh document — far smaller
    than the original and, unlike rasterizing, keeps selectable text and layout.
    Any failure (or empty/failed selection) returns the input unchanged.
    """
    if not indices:
        return pdf_bytes
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(io.BytesIO(pdf_bytes))
        n = len(reader.pages)
        writer = PdfWriter()
        for i in indices:
            if 0 <= i < n:
                writer.add_page(reader.pages[i])
        if len(writer.pages) == 0:
            return pdf_bytes
        buf = io.BytesIO()
        writer.write(buf)
        out = buf.getvalue()
        return out or pdf_bytes
    except Exception:
        logger.info("PDF page extraction failed; using full document", exc_info=True)
        return pdf_bytes


def extract_pdf_pages_text(pdf_bytes: bytes, indices: list[int]) -> list[tuple[int, str]]:
    """Return ``[(human_page_no, text)]`` for the 0-based ``indices`` of a PDF.

    Page-scoped text for callers whose cached extraction has no page boundaries.
    Plain pypdf text only (no embedded-image descriptions). Any failure returns
    ``[]``; a single unreadable page yields empty text for that page.
    """
    if not indices:
        return []
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        n = len(reader.pages)
    except Exception:
        logger.info("Could not open PDF for page text extraction", exc_info=True)
        return []
    out: list[tuple[int, str]] = []
    for i in indices:
        if not 0 <= i < n:
            continue
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            logger.info("PDF page %d text extraction failed", i + 1, exc_info=True)
            text = ""
        out.append((i + 1, text))
    return out


_THIN_PT = 3.0  # vector objects thinner than this: rules, underlines, borders
_BACKGROUND_FRACTION = 0.85  # one object covering this much of the page: a background


def _is_plain_rect(obj, raw) -> bool:
    """A path made only of ≤4 straight segments (a box: cell fill, band, frame)."""
    n = raw.FPDFPath_CountSegments(obj.raw)
    if n < 0 or n > 6:
        return False
    lines = 0
    for i in range(n):
        seg_type = raw.FPDFPathSegment_GetType(raw.FPDFPath_GetPathSegment(obj.raw, i))
        if seg_type == raw.FPDF_SEGMENT_BEZIERTO:
            return False
        if seg_type == raw.FPDF_SEGMENT_LINETO:
            lines += 1
    return lines <= 4


def pages_with_unrendered_vectors(
    pdf_bytes: bytes, page_indices: list[int] | None = None
) -> list[int]:
    """0-based pages whose vector graphics a provider that only images
    raster-bearing pages would miss.

    A page qualifies when it has no raster image of at least
    ``PDF_RASTER_RENDER_TRIGGER_PT2`` (displayed area) and its vector graphics
    cover at least ``PDF_VECTOR_MIN_AREA_PT2`` (summed bounding boxes). Vector
    graphics exclude thin rules/borders, page-sized backgrounds, and plain
    rectangles sitting behind text (cell fills, heading bands). An object census
    only — nothing is rendered. Any failure returns ``[]``.
    """
    from django.conf import settings as dj_settings

    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except ImportError:
        return []
    raster_trigger = getattr(dj_settings, "PDF_RASTER_RENDER_TRIGGER_PT2", 3_300)
    vector_min = getattr(dj_settings, "PDF_VECTOR_MIN_AREA_PT2", 8_000)
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception:
        return []
    flagged: list[int] = []
    try:
        targets = range(len(doc)) if page_indices is None else page_indices
        for i in targets:
            if not 0 <= i < len(doc):
                continue
            page = doc[i]
            try:
                pw, ph = page.get_size()
                objs = list(page.get_objects(max_depth=15))
                text_centres = []
                for o in objs:
                    if o.type == raw.FPDF_PAGEOBJ_TEXT:
                        left, bottom, right, top = o.get_pos()
                        text_centres.append(((left + right) / 2, (bottom + top) / 2))
                largest_raster = 0.0
                vector_area = 0.0
                for o in objs:
                    if o.type not in (raw.FPDF_PAGEOBJ_IMAGE, raw.FPDF_PAGEOBJ_PATH,
                                      raw.FPDF_PAGEOBJ_SHADING):
                        continue
                    left, bottom, right, top = o.get_pos()
                    w, h = max(0.0, right - left), max(0.0, top - bottom)
                    if o.type == raw.FPDF_PAGEOBJ_IMAGE:
                        largest_raster = max(largest_raster, w * h)
                        continue
                    if w < _THIN_PT or h < _THIN_PT or w * h >= _BACKGROUND_FRACTION * pw * ph:
                        continue
                    if (
                        o.type == raw.FPDF_PAGEOBJ_PATH
                        and _is_plain_rect(o, raw)
                        and any(left <= x <= right and bottom <= y <= top for x, y in text_centres)
                    ):
                        continue
                    vector_area += w * h
                if largest_raster < raster_trigger and vector_area >= vector_min:
                    flagged.append(i)
            finally:
                page.close()
    except Exception:
        logger.info("PDF vector census failed", exc_info=True)
        return []
    finally:
        doc.close()
    return flagged


def pdf_input_misses_vectors(model_id: str | None) -> bool:
    """Whether this model's native PDF input drops vector-only page visuals
    (OpenAI images a page only when it holds a sizeable raster), so pages with
    vector graphics should be rendered and sent alongside the PDF."""
    if not model_id:
        return False
    from llm.core.model_factory import detect_provider
    from llm.display import supports_modality

    return detect_provider(model_id) == "openai" and supports_modality(model_id, "image")


def render_unrendered_vector_pages(
    pdf_bytes: bytes,
    model_id: str | None,
    *,
    b64_budget: int | None = None,
    doc_indices: list[int] | None = None,
) -> list[tuple[int, bytes]]:
    """``[(1-based document page, jpeg)]`` for pages whose vector graphics this
    model's PDF input would miss (see :func:`pdf_input_misses_vectors`), rendered
    whole-page and capped at ``PDF_ATTACH_MAX_RENDER_PAGES`` / ``b64_budget``.

    ``doc_indices`` maps positions in a page-sliced PDF back to the document's
    0-based pages. Returns ``[]`` for models that see vector graphics natively.
    """
    if not pdf_input_misses_vectors(model_id):
        return []
    idx = pages_with_unrendered_vectors(pdf_bytes)[:PDF_ATTACH_MAX_RENDER_PAGES]
    if not idx:
        return []
    jpegs, _ = render_pdf_pages_to_jpegs(pdf_bytes, page_indices=idx, b64_budget=b64_budget)
    out = []
    for i, jpeg in zip(idx, jpegs):
        doc_page = doc_indices[i] if doc_indices and i < len(doc_indices) else i
        out.append((doc_page + 1, jpeg))
    return out


def page_list(pages: list[int]) -> str:
    """'page 3' / 'pages 2, 3' — for result wording."""
    return f"page{'s' if len(pages) > 1 else ''} {', '.join(str(p) for p in pages)}"


def _attach_vector_page_images(context, pdf_bytes, *, pathway, filename, doc_indices) -> list[int]:
    """Queue rendered images of the PDF's vector-graphic pages (models whose PDF
    input misses them only); returns the 1-based document pages queued."""
    rendered = render_unrendered_vector_pages(
        pdf_bytes,
        getattr(context, "model_id", None),
        b64_budget=context.native_asset_budget_remaining(pathway),
        doc_indices=doc_indices,
    )
    queued: list[int] = []
    for page_no, jpeg in rendered:
        if not context.try_add_native_asset({
            "kind": "image",
            "asset_id": "",
            "b64": base64.b64encode(jpeg).decode("ascii"),
            "media_type": "image/jpeg",
            "description": f"'{filename}' page {page_no} (rendered)",
        }, pathway=pathway):
            break
        queued.append(page_no)
    return queued


@dataclass
class PdfAttachOutcome:
    """What :func:`attach_pdf_to_context` managed to queue for the model."""

    representation: str  # "native" | "native_pages_as_images" | "text"
    total_pages: int = 0  # pages in the (page-sliced) document; 0 if unknown
    pages_attached: int = 0  # native: total_pages; images: pages queued; text: 0
    compressed: bool = False  # lossless recompression is what made it fit
    over_page_cap: bool = False
    reason: str = ""  # why it isn't a full native attach ("" when native)
    page_note: str = ""  # " (pages 3-5)" when a page selection applied
    rendered_pages: list = None  # 1-based document pages also sent as rendered images

    def __post_init__(self):
        if self.rendered_pages is None:
            self.rendered_pages = []


def attach_pdf_to_context(
    context,
    pdf_bytes: bytes,
    *,
    pathway: str,
    filename: str,
    description: str,
    extracted_text: str,
    pages: str | None = None,
    allow_render: bool = True,
    always_compress: bool = False,
) -> PdfAttachOutcome:
    """Queue a PDF for the model's next request via the staged degrade chain.

    Shared by every tool that shows a PDF natively (``document_view_native``,
    ``skill_resource_view``, ``chat_attachment_view``):

    1. ``pages`` → slice to a smaller sub-PDF up front, so every later stage
       works on just the requested pages;
    2. ``always_compress`` → lossless recompression before the first attempt;
    3. over ``NATIVE_REQUEST_MAX_PDF_PAGES`` → skip the native attach entirely;
    4. attach as-is, then (unless already compressed) retry losslessly
       compressed;
    5. ``allow_render`` → first-N pages as JPEG images within the remaining
       budget (a truncated view — the caller should inline the extracted text);
    6. otherwise ``representation="text"`` and nothing is queued.

    On a native attach for a model whose PDF input misses vector-only visuals
    (``pdf_input_misses_vectors(context.model_id)``), pages with sizeable vector
    graphics and no big raster are also queued as rendered images
    (``outcome.rendered_pages``).

    All queuing goes through ``context.try_add_native_asset(..., pathway)``.
    Queued PDF items carry ``pages`` so the drain can estimate their real token
    cost. Callers map the outcome onto their own result strings.
    """
    from django.conf import settings as dj_settings

    data = pdf_bytes
    page_note = ""
    doc_indices: list[int] | None = None  # slice position -> 0-based document page
    if pages:
        indices = parse_page_ranges(pages, pdf_page_count(data))
        if indices:
            data = extract_pdf_pages(data, indices)
            page_note = f" (pages {pages})"
            doc_indices = indices

    if always_compress:
        data = compress_pdf_lossless(data)

    page_cap = getattr(dj_settings, "NATIVE_REQUEST_MAX_PDF_PAGES", 100)
    n_pages = pdf_page_count(data)
    over_page_cap = 0 < page_cap < n_pages

    def _try_attach(pdf: bytes) -> bool:
        return context.try_add_native_asset({
            "kind": "pdf",
            "b64": base64.b64encode(pdf).decode("ascii"),
            "filename": filename,
            "description": description,
            "extracted_text": extracted_text,
            "pages": n_pages or 1,
        }, pathway=pathway)

    def _native(compressed_flag: bool) -> PdfAttachOutcome:
        return PdfAttachOutcome(
            "native", total_pages=n_pages, pages_attached=n_pages,
            compressed=compressed_flag, page_note=page_note,
            rendered_pages=_attach_vector_page_images(
                context, data, pathway=pathway, filename=filename, doc_indices=doc_indices,
            ),
        )

    if not over_page_cap:
        if _try_attach(data):
            return _native(always_compress)
        if not always_compress:
            smaller = compress_pdf_lossless(data)
            if len(smaller) < len(data) and _try_attach(smaller):
                return _native(True)

    reason = (
        f"has too many pages ({n_pages}) to attach in full"
        if over_page_cap else "too large to attach in full"
    )
    if allow_render:
        jpeg_pages, total_pages = render_pdf_pages_to_jpegs(
            data, b64_budget=context.native_asset_budget_remaining(pathway),
        )
        attached = 0
        for p, jpeg in enumerate(jpeg_pages, start=1):
            if not context.try_add_native_asset({
                "kind": "image",
                "asset_id": "",
                "b64": base64.b64encode(jpeg).decode("ascii"),
                "media_type": "image/jpeg",
                "description": f"'{filename}'{page_note} page {p} of {total_pages} (truncated view)",
            }, pathway=pathway):
                break
            attached += 1
        if attached:
            if over_page_cap:
                reason = f"has too many pages ({total_pages}) to attach in full"
            return PdfAttachOutcome(
                "native_pages_as_images", total_pages=total_pages, pages_attached=attached,
                over_page_cap=over_page_cap, reason=reason, page_note=page_note,
            )

    return PdfAttachOutcome(
        "text", total_pages=n_pages, over_page_cap=over_page_cap,
        reason="attachment budget exhausted" if not over_page_cap else reason,
        page_note=page_note,
    )


__all__ = [
    "attach_pdf_to_context",
    "pages_with_unrendered_vectors",
    "pdf_input_misses_vectors",
    "render_unrendered_vector_pages",
    "extract_pdf_pages_text",
    "PdfAttachOutcome",
    "pdf_page_count",
    "compress_pdf_lossless",
    "render_pdf_pages_to_jpegs",
    "parse_page_ranges",
    "extract_pdf_pages",
    "PDF_ATTACH_MAX_RENDER_PAGES",
    "PDF_RENDER_SCALE",
    "PDF_RENDER_JPEG_QUALITY",
    "PDF_RENDER_MAX_DIMENSION",
]
