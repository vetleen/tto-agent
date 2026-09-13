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

import io
import logging

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


__all__ = [
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
