"""Validation and resizing for user profile pictures.

Pure helpers with no Django request coupling so they can be unit tested
directly. The hardening mirrors ``feedback.validation.reencode_screenshot``:
``Image.open`` validates the magic bytes, a pixel-count guard rejects
decompression bombs *before* the full decode allocates RAM, and re-encoding
(rather than storing the original bytes) strips EXIF, embedded payloads, and
polyglot content. There is deliberately no PII/guardrails scan here — the image
bytes are never fed to the assistant.
"""

import io
import logging

from django.conf import settings
from django.core.files.base import ContentFile
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Only formats we can safely re-encode and serve. Anything else is rejected.
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

# Image.open() reads only the header, so dimensions are known before any pixel
# decode. Reject pixel bombs here before the decode below allocates
# width*height*bytes-per-pixel of RAM. 50M px accepts real high-resolution
# camera/phone photos (e.g. a 48 MP phone or a 31 MP DSLR) while still blocking
# absurd bombs; decode stays memory-bounded because we transpose and thumbnail
# the image in place (no extra full-size copies).
_MAX_IMAGE_PIXELS = 50_000_000

# Pre-decode upload cap (overridable for ops headroom). 15 MB comfortably covers
# high-quality phone JPEGs; the pixel-count guard above is what bounds decode RAM.
DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

# Longest edge of the resized avatar. 256 stays crisp on retina at the ~32px
# display size while keeping the stored file tiny.
AVATAR_MAX_EDGE = 256


class InvalidProfilePicture(Exception):
    """An uploaded profile picture was missing, too large, or not a usable image."""


def max_upload_bytes() -> int:
    return getattr(settings, "PROFILE_PICTURE_MAX_BYTES", DEFAULT_MAX_UPLOAD_BYTES)


def process_profile_picture(f):
    """Validate, re-encode, and resize an uploaded profile picture.

    Returns ``(ext, original, resized)`` where *ext* is the file extension and
    *original*/*resized* are :class:`~django.core.files.base.ContentFile`\\ s.
    *original* is the full-size upload re-encoded (metadata stripped); *resized*
    is a square-bounded thumbnail for the nav/chat avatar.

    Raises :class:`InvalidProfilePicture` for any problem — an oversize upload,
    an unreadable file, a disallowed format, or a pixel bomb. Its message is
    user-facing (the view returns it verbatim), so each case says what's wrong.
    """
    limit = max_upload_bytes()
    size = getattr(f, "size", None)
    if size is not None and size > limit:
        raise InvalidProfilePicture(f"That image is too large. Please use one under {limit // (1024 * 1024)} MB.")

    try:
        f.seek(0)
        with Image.open(f) as img:
            fmt = img.format
            if fmt not in _ALLOWED_FORMATS:
                raise InvalidProfilePicture("That image type isn't supported. Please use a JPEG, PNG, or WebP.")
            # Header dimensions — reject bombs before the decode in _encode().
            if img.width * img.height > _MAX_IMAGE_PIXELS:
                # Name the image's actual dimensions so the user can see why.
                raise InvalidProfilePicture(
                    f"That image is too large ({img.width} × {img.height} pixels). Please use a smaller one."
                )
            # Honor EXIF orientation so portrait photos aren't shown sideways,
            # in place to avoid a second full-size copy; the rest of the metadata
            # is dropped by re-encoding below. save() in _encode() forces the full
            # decode, which is where Pillow raises on truncation/bombs.
            ImageOps.exif_transpose(img, in_place=True)
            original_bytes = _encode(img, fmt)

            # Shrink the same image in place for the avatar thumbnail — no extra
            # full-size copy, so peak memory stays at a single decode.
            img.thumbnail((AVATAR_MAX_EDGE, AVATAR_MAX_EDGE), Image.Resampling.LANCZOS)
            resized_bytes = _encode(img, fmt)
    except InvalidProfilePicture:
        raise
    except Exception as exc:
        # Garbage bytes, truncated images, decompression bombs, unsupported
        # modes — all map to the same user-facing failure.
        logger.info("Rejected profile picture upload: %s", exc.__class__.__name__)
        raise InvalidProfilePicture("That file couldn't be read as an image. Please use a JPEG, PNG, or WebP.") from exc

    return _FORMAT_EXT[fmt], ContentFile(original_bytes), ContentFile(resized_bytes)


def _encode(img, fmt):
    """Encode *img* to *fmt* bytes. save() forces the full decode, which is where
    Pillow raises on truncation/decompression bombs — caught by the caller."""
    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        # JPEG can't hold alpha or palette modes.
        img = img.convert("RGB")
    buffer = io.BytesIO()
    save_kwargs = {"quality": 88, "optimize": True} if fmt == "JPEG" else {}
    img.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()
