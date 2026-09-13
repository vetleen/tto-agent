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

from PIL import Image, ImageOps

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


def _has_alpha(img: "Image.Image") -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )


def optimize_for_vision(
    raw_bytes: bytes, *, allow_transcode: bool = True
) -> tuple[bytes, str] | None:
    """Downscale + re-encode an image to the size a vision model actually uses.

    Caps the long edge and total pixel area (settings ``VISION_IMAGE_MAX_EDGE`` /
    ``VISION_IMAGE_MAX_PIXELS``) — every provider downsamples beyond ~1568px /
    ~1.15 MP, so anything larger is wasted bytes. Honors EXIF orientation and
    strips metadata / polyglot content, like :func:`sanitize_raster_image`.

    With ``allow_transcode=True`` (default) it re-encodes opaque images to JPEG
    (``VISION_IMAGE_JPEG_QUALITY``) and images with transparency to optimized PNG
    — the smaller "save for web" output. With ``allow_transcode=False`` it keeps
    the original format (so a caller relying on a stable stored ``media_type``,
    e.g. data-room images read back for viewing, isn't broken by a format change).

    Returns ``(optimized_bytes, media_type)``, or the original bytes + their
    content type when re-encoding wouldn't shrink them (never inflates), or
    ``None`` when the input is not a usable raster in an allowed format.
    """
    try:
        from django.conf import settings

        max_edge = int(getattr(settings, "VISION_IMAGE_MAX_EDGE", 1568))
        max_pixels = int(getattr(settings, "VISION_IMAGE_MAX_PIXELS", 1_150_000))
        quality = int(getattr(settings, "VISION_IMAGE_JPEG_QUALITY", 82))
    except Exception:
        max_edge, max_pixels, quality = 1568, 1_150_000, 82

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            fmt = img.format
            if fmt not in _ALLOWED_FORMATS:
                return None
            if fmt == "JPEG":
                # Downscale-on-decode: the JPEG decoder returns a reduced (≤1/8)
                # image, bounding decode RAM/time — so a very large photo gets
                # SHRUNK here instead of rejected by the pixel guard below.
                img.draft("RGB", (max_edge, max_edge))
            # Header (or post-draft) dims — reject genuine decode bombs before the
            # decode allocates RAM. Non-JPEG formats can't downscale on decode, so
            # this still guards them; a pathological JPEG stays over even post-draft.
            if img.width * img.height > _MAX_IMAGE_PIXELS:
                return None
            ImageOps.exif_transpose(img, in_place=True)
            w, h = img.width, img.height
            # Shrink to satisfy BOTH the long-edge and the pixel-area caps; never
            # upscale (a small image stays its size).
            scale = min(1.0, max_edge / max(w, h), (max_pixels / (w * h)) ** 0.5)
            if scale < 1.0:
                img.thumbnail(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            if not allow_transcode:
                # Preserve the original format — keeps a stored media_type stable.
                media_type = _FORMAT_CONTENT_TYPE.get(fmt, "image/png")
                if fmt == "JPEG":
                    rgb = img if img.mode in ("RGB", "L") else img.convert("RGB")
                    rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
                elif fmt == "WEBP":
                    img.save(buffer, format="WEBP", quality=quality, method=6)
                elif fmt == "GIF":
                    img.save(buffer, format="GIF")
                else:  # PNG
                    img.save(buffer, format="PNG", optimize=True)
            elif _has_alpha(img):
                media_type = "image/png"
                img.save(buffer, format="PNG", optimize=True)
            else:
                media_type = "image/jpeg"
                rgb = img if img.mode in ("RGB", "L") else img.convert("RGB")
                rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
            optimized = buffer.getvalue()
    except Exception:
        return None

    if optimized and len(optimized) < len(raw_bytes):
        return optimized, media_type
    # Re-encoding didn't help (already small/efficient) — keep the original bytes.
    return raw_bytes, _FORMAT_CONTENT_TYPE.get(fmt, media_type)
