"""Per-theme footer/brand logo blobs, stored in default Django storage keyed by
theme id (``slide_theme_logos/<theme_id>.<ext>``).

A v2 slide theme (chat.slides.theme) is a JSON object; its logo is a separate file
whose presence is recorded by the theme's ``logo_ext``. Both the footer stamper and
the reserved ``[[image:company-logo]]`` token read the bytes back through here, so a
theme is self-contained without embedding image bytes in the preferences JSON.
"""
from __future__ import annotations

import logging

from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

_DIR = "slide_theme_logos"
LOGO_EXTS = ("png", "jpg", "jpeg", "webp")


def slide_theme_logo_path(theme_id: str, ext: str) -> str | None:
    if not theme_id or ext not in LOGO_EXTS:
        return None
    # theme ids are minted tokens (no path separators), but guard anyway.
    if "/" in theme_id or "\\" in theme_id or ".." in theme_id:
        return None
    return f"{_DIR}/{theme_id}.{ext}"


def slide_theme_logo_bytes(theme_id: str, ext: str) -> bytes | None:
    """The logo bytes for a theme, or ``None`` (missing / unreadable)."""
    path = slide_theme_logo_path(theme_id, ext)
    if not path:
        return None
    try:
        if not default_storage.exists(path):
            return None
        with default_storage.open(path, "rb") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001 — a missing/broken logo must not fail a render
        logger.warning("slide theme logo read failed: %s", theme_id, exc_info=True)
        return None


def save_slide_theme_logo(theme_id: str, content, ext: str) -> str | None:
    """Store (overwrite) a theme's logo; returns the stored path or ``None``.

    ``content`` is a Django ``File``/``ContentFile`` (e.g. from ``process_org_logo``)."""
    path = slide_theme_logo_path(theme_id, ext)
    if not path:
        return None
    delete_slide_theme_logo(theme_id)  # clear any prior ext so one logo per theme
    try:
        default_storage.save(path, content)
        return path
    except Exception:  # noqa: BLE001
        logger.warning("slide theme logo save failed: %s", theme_id, exc_info=True)
        return None


def delete_slide_theme_logo(theme_id: str) -> None:
    """Remove a theme's logo blob (all extensions), best-effort."""
    for ext in LOGO_EXTS:
        path = slide_theme_logo_path(theme_id, ext)
        if not path:
            continue
        try:
            if default_storage.exists(path):
                default_storage.delete(path)
        except Exception:  # noqa: BLE001
            logger.debug("slide theme logo delete failed: %s.%s", theme_id, ext, exc_info=True)
