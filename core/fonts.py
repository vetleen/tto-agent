"""Font resolution for PDF export.

DOCX export records only a font *name* (Word substitutes locally). PDF export
renders server-side with WeasyPrint, which needs the actual font *file* embedded
in the document — otherwise it silently substitutes. This module resolves a
requested font name to concrete font faces, in priority order:

1. **Org upload** — a brand typeface the org uploaded (``FontAsset``, embeddable).
2. **Bundled by name** — an OFL font we ship under ``core/assets/fonts`` whose
   own name was requested (e.g. "Carlito", "EB Garamond").
3. **Substitute map** — a proprietary/common name mapped onto a bundled font
   (``core.styles.FONT_SUBSTITUTES``): "metric"-compatible (silent) or "visual"
   (a soft note).
4. **Google Fonts** — auto-fetched and cached globally (``FontAsset``) so a
   custom-typed family renders without a per-export round trip.
5. **Fallback** — a neutral bundled sans, recorded so the export can warn.

The result carries the font bytes; :meth:`FontResolution.font_face_css` emits the
``@font-face`` blocks (bytes as ``data:`` URLs, so S3-stored faces need no temp
files). ``core.styles.build_pdf_css`` consumes the resolutions.

No WeasyPrint import here — this is pure Python and fully unit-testable on any
platform (the native Pango libs WeasyPrint needs are only required at render).
"""
from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from core.styles import BUNDLED_FONTS, FALLBACK_FONT, FONT_SUBSTITUTES

logger = logging.getLogger(__name__)

_FONTS_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# The four faces we bundle/resolve per family.
_FACE_SPECS = (
    ("Regular", 400, "normal"),
    ("Bold", 700, "normal"),
    ("Italic", 400, "italic"),
    ("BoldItalic", 700, "italic"),
)

_CATEGORY_GENERIC = {"sans": "sans-serif", "serif": "serif", "mono": "monospace"}

# Heroku dynos install Pango but no font *files*, so a CSS generic keyword
# (sans-serif/serif/monospace) resolves to nothing — every font-family stack
# must end in an embedded face. These bundled families back each generic.
GENERIC_FALLBACK = {"sans-serif": "Carlito", "serif": "Tinos", "monospace": "Cousine"}

# Google Fonts Developer-API variant key -> (weight, css style).
_GOOGLE_VARIANTS = {
    "regular": (400, "normal"),
    "700": (700, "normal"),
    "italic": (400, "italic"),
    "700italic": (700, "italic"),
}

_GOOGLE_MAP_CACHE_KEY = "google_fonts_map_v1"
_GOOGLE_LIST_URL = "https://www.googleapis.com/webfonts/v1/webfonts"

# data-URL mime + CSS format() keyword per detected face format.
_FMT_MIME = {
    "truetype": "font/ttf",
    "opentype": "font/otf",
    "woff": "font/woff",
    "woff2": "font/woff2",
}


def normalize_font_name(name: str) -> str:
    """Lowercase, trim, and collapse whitespace for case-insensitive lookups."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


# Bundled font name (normalized) -> family key. Covers both the directory key
# ("EBGaramond") and the display label ("EB Garamond").
_BUNDLED_BY_NAME = {}
for _key, _meta in BUNDLED_FONTS.items():
    _BUNDLED_BY_NAME[normalize_font_name(_key)] = _key
    _BUNDLED_BY_NAME[normalize_font_name(_meta["label"])] = _key


@dataclass
class FontFace:
    weight: int
    style: str  # "normal" | "italic"
    data: bytes
    fmt: str  # truetype | opentype | woff | woff2


@dataclass
class FontResolution:
    requested: str
    css_family: str
    generic: str  # sans-serif | serif | monospace
    faces: list[FontFace]
    fidelity: str  # exact | metric | visual | fallback
    actual: str
    note: str = ""

    @property
    def font_family_css(self) -> str:
        return f"'{self.css_family}', {self.generic}"

    def font_face_css(self) -> str:
        """``@font-face`` blocks embedding each face as a ``data:`` URL."""
        out = []
        for face in self.faces:
            mime = _FMT_MIME.get(face.fmt, "font/ttf")
            b64 = base64.b64encode(face.data).decode("ascii")
            out.append(
                "@font-face{"
                f"font-family:'{self.css_family}';"
                f"font-weight:{face.weight};"
                f"font-style:{face.style};"
                f"src:url(data:{mime};base64,{b64}) format('{face.fmt}');"
                "}"
            )
        return "".join(out)


# --- format / embedding inspection (used by the upload endpoint too) --------

def font_format(data: bytes) -> str:
    """Detect a font's container format from its magic bytes."""
    head = data[:4]
    if head == b"wOFF":
        return "woff"
    if head == b"wOF2":
        return "woff2"
    if head == b"OTTO":
        return "opentype"
    if head in (b"\x00\x01\x00\x00", b"true", b"ttcf"):
        return "truetype"
    return ""


def inspect_font(data: bytes) -> dict:
    """Parse a font file's metadata for the upload path.

    Returns ``{family, weight, style, fmt, embeddable}``. Raises ``ValueError``
    if the bytes aren't a usable font. ``embeddable`` is False when the OS/2
    ``fsType`` Restricted-License bit (0x0002) is set — such faces must not be
    embedded in a PDF (embedding = redistribution).
    """
    fmt = font_format(data)
    if not fmt:
        raise ValueError("Unrecognised font format.")
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(io.BytesIO(data), fontNumber=0, lazy=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Could not parse font file.") from exc
    try:
        os2 = font.get("OS/2")
        head = font.get("head")
        fstype = int(getattr(os2, "fsType", 0)) if os2 is not None else 0
        embeddable = not (fstype & 0x0002)
        weight = int(getattr(os2, "usWeightClass", 400)) if os2 is not None else 400
        sel = int(getattr(os2, "fsSelection", 0)) if os2 is not None else 0
        mac = int(getattr(head, "macStyle", 0)) if head is not None else 0
        italic = bool((sel & 0x01) or (mac & 0x02))
        name_tbl = font["name"]
        family = (
            name_tbl.getBestFamilyName()
            or name_tbl.getDebugName(1)
            or ""
        )
    finally:
        font.close()
    return {
        "family": family.strip(),
        "weight": weight,
        "style": "italic" if italic else "normal",
        "fmt": fmt,
        "embeddable": embeddable,
    }


# --- bundled faces ----------------------------------------------------------

@lru_cache(maxsize=None)
def _bundled_faces(family_key: str) -> tuple[FontFace, ...]:
    directory = _FONTS_DIR / family_key
    faces = []
    for suffix, weight, style in _FACE_SPECS:
        path = directory / f"{family_key}-{suffix}.ttf"
        if path.exists():
            faces.append(FontFace(weight, style, path.read_bytes(), "truetype"))
    return tuple(faces)


def _bundled_resolution(requested, family_key, fidelity, *, note=""):
    meta = BUNDLED_FONTS[family_key]
    return FontResolution(
        requested=requested,
        css_family=f"wf-{family_key.lower()}",
        generic=_CATEGORY_GENERIC.get(meta["category"], "sans-serif"),
        faces=list(_bundled_faces(family_key)),
        fidelity=fidelity,
        actual=meta["label"],
        note=note,
    )


def bundled_resolution(family_key: str) -> FontResolution:
    """Public accessor for a bundled family (used as an embedded CSS fallback)."""
    return _bundled_resolution(BUNDLED_FONTS[family_key]["label"], family_key, "exact")


# --- org-uploaded faces -----------------------------------------------------

def _face_from_asset(asset) -> FontFace | None:
    try:
        with asset.blob.open("rb") as fh:
            data = fh.read()
    except Exception:  # noqa: BLE001
        logger.warning("Font asset %s blob unreadable", asset.id)
        return None
    fmt = asset.font_format or font_format(data) or "truetype"
    return FontFace(asset.weight, asset.style, data, fmt)


def _uploaded_resolution(requested, normalized, org):
    from accounts.models import FontAsset

    rows = FontAsset.objects.filter(
        organization=org, family_norm=normalized, embeddable=True
    )
    faces = [f for f in (_face_from_asset(r) for r in rows) if f is not None]
    if not faces:
        return None
    label = rows[0].family or requested
    return FontResolution(
        requested=requested,
        css_family=f"wf-org-{org.id}-{re.sub(r'[^a-z0-9]+', '-', normalized)}",
        generic="sans-serif",
        faces=faces,
        fidelity="exact",
        actual=label,
    )


# --- Google Fonts -----------------------------------------------------------

def _google_family_map() -> dict:
    """Return ``{normalized_family: {family, category, files}}`` from Google.

    Cached in Redis (24h on success, briefly on failure). Empty when no API key
    is configured — the Google tier is then skipped.
    """
    api_key = getattr(settings, "GOOGLE_FONTS_API_KEY", "")
    if not api_key:
        return {}
    cached = cache.get(_GOOGLE_MAP_CACHE_KEY)
    if cached is not None:
        return cached
    try:
        import requests

        resp = requests.get(
            _GOOGLE_LIST_URL, params={"key": api_key, "sort": "popularity"}, timeout=15
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except Exception:  # noqa: BLE001
        logger.warning("Google Fonts list fetch failed", exc_info=True)
        cache.set(_GOOGLE_MAP_CACHE_KEY, {}, 300)
        return {}
    out = {}
    for item in items:
        family = item.get("family")
        if not family:
            continue
        out[normalize_font_name(family)] = {
            "family": family,
            "category": item.get("category", ""),
            "files": item.get("files", {}) or {},
        }
    cache.set(_GOOGLE_MAP_CACHE_KEY, out, 24 * 3600)
    return out


def _download(url: str) -> bytes | None:
    try:
        import requests

        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.content
    except Exception:  # noqa: BLE001
        logger.warning("Google font file download failed: %s", url, exc_info=True)
        return None


def _store_google_face(family, normalized, data, weight, style):
    import hashlib

    from django.core.files.base import ContentFile

    from accounts.models import FontAsset

    sha = hashlib.sha256(data).hexdigest()
    if FontAsset.objects.filter(
        source=FontAsset.SOURCE_GOOGLE, organization__isnull=True, sha256=sha
    ).exists():
        return
    asset = FontAsset(
        organization=None,
        family=family,
        family_norm=normalized,
        source=FontAsset.SOURCE_GOOGLE,
        content_type="font/ttf",
        font_format="truetype",
        weight=weight,
        style=style,
        size_bytes=len(data),
        sha256=sha,
        embeddable=True,
    )
    asset.blob.save(f"{asset.id}.ttf", ContentFile(data), save=True)


def _google_resolution(requested, normalized):
    from accounts.models import FontAsset

    # Durable cache first — already-fetched faces.
    rows = list(
        FontAsset.objects.filter(
            source=FontAsset.SOURCE_GOOGLE, organization__isnull=True, family_norm=normalized
        )
    )
    if not rows:
        entry = _google_family_map().get(normalized)
        if not entry:
            return None
        for variant, (weight, style) in _GOOGLE_VARIANTS.items():
            url = entry["files"].get(variant)
            if not url:
                continue
            data = _download(url)
            if not data:
                continue
            _store_google_face(entry["family"], normalized, data, weight, style)
        rows = list(
            FontAsset.objects.filter(
                source=FontAsset.SOURCE_GOOGLE, organization__isnull=True, family_norm=normalized
            )
        )
        if not rows:
            return None
    faces = [f for f in (_face_from_asset(r) for r in rows) if f is not None]
    if not faces:
        return None
    category = ""
    entry = _google_family_map().get(normalized)
    if entry:
        category = entry.get("category", "")
    generic = "serif" if category == "serif" else (
        "monospace" if category == "monospace" else "sans-serif"
    )
    return FontResolution(
        requested=requested,
        css_family=f"wf-g-{re.sub(r'[^a-z0-9]+', '-', normalized)}",
        generic=generic,
        faces=faces,
        fidelity="exact",
        actual=rows[0].family or requested,
    )


# --- public API -------------------------------------------------------------

def resolve_font(name: str, *, org=None) -> FontResolution:
    """Resolve one font name to a :class:`FontResolution` (never raises)."""
    requested = (name or "").strip() or FALLBACK_FONT
    normalized = normalize_font_name(requested)

    if org is not None:
        try:
            res = _uploaded_resolution(requested, normalized, org)
            if res is not None:
                return res
        except Exception:  # noqa: BLE001
            logger.warning("Uploaded-font lookup failed for %r", requested, exc_info=True)

    if normalized in _BUNDLED_BY_NAME:
        return _bundled_resolution(requested, _BUNDLED_BY_NAME[normalized], "exact")

    if normalized in FONT_SUBSTITUTES:
        family_key, kind = FONT_SUBSTITUTES[normalized]
        note = ""
        if kind == "visual":
            note = (
                f"'{requested}' isn't available; used the visually similar "
                f"{BUNDLED_FONTS[family_key]['label']}."
            )
        return _bundled_resolution(requested, family_key, kind, note=note)

    try:
        res = _google_resolution(requested, normalized)
        if res is not None:
            return res
    except Exception:  # noqa: BLE001
        logger.warning("Google-font resolution failed for %r", requested, exc_info=True)

    fallback = _bundled_resolution(requested, FALLBACK_FONT, "fallback")
    fallback.note = (
        f"Tried to use '{requested}' but it isn't available; used "
        f"{fallback.actual} instead."
    )
    return fallback


def resolve_fonts(styles: dict, org=None) -> dict[str, FontResolution]:
    """Resolve every distinct font named in ``styles`` (deduped by name)."""
    names = {
        (styles.get(key) or "").strip()
        for key in ("body_font", "heading_font", "header_font", "footer_font")
    }
    return {name: resolve_font(name, org=org) for name in names if name}


# --- org font upload --------------------------------------------------------

_EXT_FOR_FMT = {"truetype": "ttf", "opentype": "otf", "woff": "woff", "woff2": "woff2"}


def ingest_uploaded_font(org, *, filename, data, created_by=None, max_bytes=None):
    """Validate and store one uploaded font face for ``org``.

    Returns the created (or existing, by content hash) ``FontAsset``. Raises
    ``ValueError`` with a user-facing message if the file isn't a usable,
    embeddable font. One file = one face; an org uploads Regular/Bold/Italic/
    BoldItalic separately and they group under the same family.
    """
    import hashlib

    from django.core.files.base import ContentFile

    from accounts.models import FontAsset

    if not data:
        raise ValueError("The font file is empty.")
    if max_bytes and len(data) > max_bytes:
        raise ValueError(f"Font is too large (max {max_bytes // 1_000_000} MB).")

    meta = inspect_font(data)  # raises ValueError if not a font
    if not meta["embeddable"]:
        raise ValueError(
            "This font's license doesn't permit embedding, so it can't be used in PDFs."
        )
    family = meta["family"] or Path(filename).stem
    normalized = normalize_font_name(family)
    if not normalized:
        raise ValueError("Could not read the font's family name.")

    sha = hashlib.sha256(data).hexdigest()
    existing = FontAsset.objects.filter(organization=org, sha256=sha).first()
    if existing is not None:
        return existing

    asset = FontAsset(
        organization=org,
        family=family,
        family_norm=normalized,
        source=FontAsset.SOURCE_UPLOAD,
        content_type=f"font/{meta['fmt']}",
        font_format=meta["fmt"],
        weight=meta["weight"],
        style=meta["style"],
        size_bytes=len(data),
        sha256=sha,
        embeddable=True,
        created_by=created_by,
    )
    ext = _EXT_FOR_FMT.get(meta["fmt"], "ttf")
    asset.blob.save(f"{asset.id}.{ext}", ContentFile(data), save=True)
    return asset


def org_font_families(org) -> list[dict]:
    """Uploaded fonts for ``org``, grouped by family (for the settings UI)."""
    from accounts.models import FontAsset

    rows = FontAsset.objects.filter(
        organization=org, source=FontAsset.SOURCE_UPLOAD
    ).order_by("family", "weight", "style")
    families: dict[str, dict] = {}
    for row in rows:
        fam = families.setdefault(
            row.family_norm, {"family": row.family, "family_norm": row.family_norm, "faces": []}
        )
        fam["faces"].append({"weight": row.weight, "style": row.style})
    return list(families.values())
