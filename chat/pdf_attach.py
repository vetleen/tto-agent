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
) -> tuple[list[bytes], int]:
    """Render the first pages of a PDF to JPEG bytes.

    Stops at ``max_pages`` or when the cumulative base64 size of the rendered
    pages would exceed ``b64_budget``. Returns ``(jpeg_pages, total_pages)``;
    on any failure returns ``([], 0)`` so the caller degrades to text.
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
        for i in range(min(total, max_pages)):
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


__all__ = [
    "compress_pdf_lossless",
    "render_pdf_pages_to_jpegs",
    "PDF_ATTACH_MAX_RENDER_PAGES",
    "PDF_RENDER_SCALE",
    "PDF_RENDER_JPEG_QUALITY",
    "PDF_RENDER_MAX_DIMENSION",
]
