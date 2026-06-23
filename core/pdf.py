"""Single PDF -> text+image extractor shared across data rooms and chat.

Mirrors :mod:`core.docx`. PDFs are extracted to text page-by-page; each page's
embedded raster images are pulled out (pypdf ``page.images``) and rendered
inline via the same ``image_sink(image, idx) -> str`` callback the docx
converter uses — so the existing sinks (asset-persisting, describe-only,
placeholder) work unchanged. pypdf gives no positional layout for images, so a
page's image tokens are appended after that page's text, keeping each image in
its page context for retrieval.

Only embedded images are handled (figures, and the single full-page image a
typical *scanned* page contains). True page rasterization / OCR is out of scope.
"""

from __future__ import annotations

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


def pdf_to_text(file, *, image_sink: Callable[[object, int], str]) -> str:
    """Extract a PDF to text. ``file`` may be a path, bytes, or a binary
    file-like. For each page: ``page.extract_text()`` followed by the inline
    tokens ``image_sink`` returns for that page's embedded images.

    Images repeated across pages (logos, headers) are deduplicated by sha256 —
    the sink is invoked once and its token reused, so a recurring logo doesn't
    become N assets or N vision calls.
    """
    from django.conf import settings
    from pypdf import PdfReader

    min_dim = getattr(settings, "PDF_MIN_IMAGE_DIMENSION", PDF_MIN_IMAGE_DIMENSION)
    max_images = getattr(settings, "PDF_MAX_EMBEDDED_IMAGES", PDF_MAX_EMBEDDED_IMAGES)

    data = _read_bytes(file)
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("This PDF file is corrupt or not a valid PDF document.") from exc

    seen: dict[str, str] = {}  # sha256 -> token (dedup within the document)
    idx = 0  # 1-indexed count of images actually handed to the sink
    capped = False
    pages_out: list[str] = []

    for page in reader.pages:
        try:
            text = (page.extract_text() or "").replace("\x00", "")
        except Exception:
            logger.warning("pdf_to_text: failed to extract text from a page", exc_info=True)
            text = ""

        try:
            page_images = list(page.images)
        except Exception:
            logger.warning("pdf_to_text: failed to enumerate images on a page", exc_info=True)
            page_images = []

        tokens: list[str] = []
        for image_file in page_images:
            try:
                img_bytes = image_file.data
                pil_image = image_file.image  # may be None / may raise
            except Exception:
                logger.warning("pdf_to_text: failed to decode an embedded image; skipping", exc_info=True)
                continue
            if not img_bytes or _too_small(pil_image, img_bytes, min_dim):
                continue

            sha = hashlib.sha256(img_bytes).hexdigest()
            existing = seen.get(sha)
            if existing is not None:
                tokens.append(existing)
                continue
            if len(seen) >= max_images:
                if not capped:
                    logger.warning(
                        "pdf_to_text: more than %d distinct embedded images; "
                        "storing the first %d and dropping the rest",
                        max_images, max_images,
                    )
                    capped = True
                continue

            idx += 1
            content_type = _content_type_for(image_file, pil_image)
            try:
                token = image_sink(_PdfImage(img_bytes, content_type), idx)
            except Exception:
                logger.exception("pdf_to_text: image_sink failed for an embedded image; skipping")
                idx -= 1
                continue
            seen[sha] = token
            tokens.append(token)

        if tokens:
            page_str = (text + "\n\n" + "\n\n".join(tokens)).strip() if text else "\n\n".join(tokens)
        else:
            page_str = text
        if page_str:
            pages_out.append(page_str)

    return "\n\n".join(pages_out).strip()
