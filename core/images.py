"""Sanitize externally-sourced raster image bytes before we store or show them.

Pure helper with no request coupling. The hardening mirrors
``feedback.validation.reencode_screenshot`` and ``accounts.avatars``: Pillow's
``Image.open`` validates the magic bytes, a pixel-count guard rejects
decompression bombs *before* the full decode allocates RAM, and re-encoding
(rather than passing the original bytes through) strips EXIF, embedded payloads,
and polyglot content.

Used by ``web_image_view``, whose bytes come from arbitrary web URLs and are
both persisted as browser-served assets *and* sent to the vision model — so an
attacker-supplied ``Content-Type`` must never be trusted, and SVG (XML the
browser executes) must never reach storage.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)

# Raster formats we can safely decode, re-encode, and serve inline. SVG is
# deliberately absent — it is XML that browsers execute (an XSS vector when
# served as an image). Anything not in this set is rejected.
_ALLOWED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}
_FORMAT_CONTENT_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# Image.open() reads only the header, so dimensions are known before any pixel
# decode. Reject pixel bombs here before the decode allocates
# width*height*bytes-per-pixel of RAM. 50M px accepts real high-resolution
# photos while still blocking absurd bombs.
_MAX_IMAGE_PIXELS = 50_000_000


def sanitize_raster_image(raw_bytes: bytes) -> tuple[bytes, str] | None:
    """Validate *raw_bytes* as a supported raster image and return
    ``(safe_bytes, content_type)`` re-encoded with metadata stripped, or
    ``None`` if the bytes are not a usable image in an allowed format.

    Animated images collapse to their first frame (``save`` without
    ``save_all`` writes only the current frame), which is what the vision model
    needs and keeps the stored asset static.
    """
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = img.format
            if fmt not in _ALLOWED_FORMATS:
                return None
            # Dimensions come from the header — reject before the decode below.
            if img.width * img.height > _MAX_IMAGE_PIXELS:
                return None
            # Forces the decode where Pillow raises on truncation / bombs, then
            # re-encode from the clean image so no original metadata carries
            # over. save() defaults save_all=False → first frame for animations.
            img.load()
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            return buffer.getvalue(), _FORMAT_CONTENT_TYPE[fmt]
    except Exception:
        # Garbage bytes, truncated images, decompression bombs, unsupported
        # modes, spoofed MIME (HTML/SVG served as image/*) — all unusable.
        return None
