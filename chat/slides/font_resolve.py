"""Resolve a deck theme's fonts to concrete faces (bytes) for the renderers.

The slide renderers (:mod:`chat.slides.pillow_render`, :mod:`chat.slides.pptx_build`)
are deliberately Django-free. This module is the Django-aware seam that resolves each
family a theme uses — bundled *or* org-uploaded — to font faces via :mod:`core.fonts`,
so the Pillow preview can rasterise an uploaded brand font and the ``.pptx`` builder can
embed it. A font uploaded for canvas/PDF or for slides resolves identically (same
``FontAsset`` store), so it is reusable across both.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Font roles a theme carries (mirrors chat.slides.theme.FONT_ROLES).
FONT_ROLES = ("headline", "subhead", "body", "data")


def theme_font_families(theme: dict) -> list[str]:
    """Distinct family names the theme's four font roles resolve to (order-stable)."""
    fonts = (theme or {}).get("fonts") or {}
    seen: set[str] = set()
    out: list[str] = []
    for role in FONT_ROLES:
        fam = fonts.get(role)
        if isinstance(fam, str) and fam and fam not in seen:
            seen.add(fam)
            out.append(fam)
    return out


def is_bundled_family(name: str) -> bool:
    """True if ``name`` is a font we ship on disk (so the ``.pptx`` needn't embed it)."""
    try:
        from core.fonts import _BUNDLED_BY_NAME, normalize_font_name

        return normalize_font_name(name) in _BUNDLED_BY_NAME
    except Exception:  # noqa: BLE001
        return False


def resolve_deck_font_faces(theme: dict, org=None) -> dict:
    """Resolve every family the ``theme`` uses to a record the renderers consume.

    ``{family: {"name": actual, "bundled": bool, "faces": [core.fonts.FontFace, …]}}``.
    Each face carries raw ``data`` bytes + ``weight``/``style``/``fmt``. Never raises —
    a family that fails to resolve is simply omitted (the renderer then falls back to
    its bundled-on-disk lookup). Returns an empty dict when the theme has no fonts.
    """
    from core.fonts import resolve_font

    out: dict[str, dict] = {}
    for fam in theme_font_families(theme):
        try:
            res = resolve_font(fam, org=org)
        except Exception:  # noqa: BLE001
            logger.warning("slide font resolve failed for %r", fam, exc_info=True)
            continue
        if not res.faces:
            continue
        out[fam] = {
            "name": res.actual or fam,
            "bundled": is_bundled_family(fam),
            "faces": list(res.faces),
        }
    return out
