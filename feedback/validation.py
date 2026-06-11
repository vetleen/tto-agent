"""Input validation helpers for feedback submissions.

These are pure functions with no Django request coupling so they can be unit
tested directly. They harden the three pieces of client-controlled input the
submit endpoint trusts: the screenshot bytes, the captured console errors, and
the page URL.
"""

import io
import json
import logging

from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.validators import URLValidator
from PIL import Image

logger = logging.getLogger(__name__)

# Only formats the client widget actually produces (it sends JPEG; PNG/WEBP are
# allowed in case the capture path changes). Anything else is rejected.
_ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
_IMAGE_FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

# Keys the widget sends per console-error entry (static/js/feedback-widget.js).
_CONSOLE_STR_KEYS = ("message", "source", "stack", "timestamp")
_CONSOLE_INT_KEYS = ("lineno", "colno")
_CONSOLE_STR_MAXLEN = 2000

MAX_CONSOLE_ERRORS = 50
# Pre-parse guard so a multi-megabyte body never reaches json.loads.
MAX_CONSOLE_ERRORS_RAW_CHARS = 256 * 1024

_url_validator = URLValidator(schemes=["http", "https"])


def reencode_screenshot(f):
    """Validate and re-encode an uploaded screenshot.

    Returns ``(filename, ContentFile)`` on success, or ``None`` if the upload
    is not a real image in an allowed format. Re-encoding (rather than storing
    the original bytes) strips any embedded payload, EXIF, or polyglot content,
    and forcing the filename discards the client-supplied name. The caller is
    responsible for enforcing the size cap before calling this.
    """
    try:
        f.seek(0)
        with Image.open(f) as img:
            fmt = img.format
            if fmt not in _ALLOWED_IMAGE_FORMATS:
                return None
            # save() forces a full decode here, which is where Pillow raises
            # DecompressionBombError / truncation errors — all caught below.
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
        filename = f"screenshot.{_IMAGE_FORMAT_EXT[fmt]}"
        return filename, ContentFile(buffer.getvalue())
    except Exception:
        # Garbage bytes, truncated images, decompression bombs, unsupported
        # modes — treat every failure the same: it is not a usable screenshot.
        return None


def sanitize_console_errors(raw):
    """Parse and whitelist the client-supplied console_errors payload.

    Returns a list of cleaned dicts (at most ``MAX_CONSOLE_ERRORS``), keeping
    only the keys the widget sends, with string values truncated. Returns ``[]``
    for empty input, oversize input, or any parse failure (including the
    ``RecursionError`` that deeply-nested JSON would otherwise raise uncaught).
    """
    if not raw or len(raw) > MAX_CONSOLE_ERRORS_RAW_CHARS:
        return []

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return []

    if not isinstance(parsed, list):
        return []

    cleaned = []
    for entry in parsed[:MAX_CONSOLE_ERRORS]:
        if not isinstance(entry, dict):
            continue
        clean = {}
        for key in _CONSOLE_STR_KEYS:
            value = entry.get(key)
            if isinstance(value, str):
                clean[key] = value[:_CONSOLE_STR_MAXLEN]
        for key in _CONSOLE_INT_KEYS:
            value = entry.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                clean[key] = value
        if clean:
            cleaned.append(clean)
    return cleaned


def clean_feedback_url(raw, max_length=2000):
    """Return a validated http/https URL, or ``""`` if invalid.

    URLField validators don't run on ``model.save()``, so without this a
    ``javascript:`` or ``data:`` URL would be storable. Mirrors the http/https
    scheme rule enforced in ``llm/tools/web_fetch.py``.
    """
    url = (raw or "").strip()[:max_length]
    if not url:
        return ""
    try:
        _url_validator(url)
    except ValidationError:
        return ""
    return url
