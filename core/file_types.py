"""Single source of truth for upload file-type capabilities.

Pure-data module (no Django imports) so it can be imported from
``config/settings.py`` at settings-load time as well as from views and
services.  Every upload surface derives its allow-list by filtering this
table on ``kind``:

* data rooms (``documents``)        -> all kinds
* chat attachments (``chat``)       -> all kinds **except** audio/email
* meeting attachments (``meetings``)-> the chat-compatible kinds

Per file type we track two MIME sets:

* ``canonical_mimes`` — the "official" MIME type(s), unioned into the global
  ``DOCUMENT_ALLOWED_MIME_TYPES`` allow-list.
* ``accepted_mimes`` — the broader set a browser may actually report for the
  extension (always a superset of ``canonical_mimes``). Used for the
  per-extension cross-check (``DOCUMENT_EXTENSION_MIME_MAP``) and for chat
  content-type matching.

Size caps are intentionally *not* modelled here — they remain surface-specific
(data-room document vs. audio cap, chat 10 MB vs. PDF 30 MB, meeting cap).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- File-kind constants -------------------------------------------------
KIND_IMAGE = "image"
KIND_PDF = "pdf"
KIND_DOCX = "docx"
KIND_TEXT = "text"
KIND_EMAIL = "email"
KIND_AUDIO = "audio"


@dataclass(frozen=True)
class FileType:
    ext: str  # canonical extension, lowercase, no leading dot
    kind: str
    canonical_mimes: frozenset  # official MIME type(s) -> global allow-list
    accepted_mimes: frozenset  # browser-reported superset -> per-ext cross-check


def _ft(ext, kind, canonical, accepted=None):
    canonical = frozenset(canonical)
    accepted = frozenset(accepted) | canonical if accepted else canonical
    return FileType(ext=ext, kind=kind, canonical_mimes=canonical, accepted_mimes=accepted)


# Ordered table. Existing data-room entries reproduce the previous
# DOCUMENT_* settings exactly; image rows are the new addition.
FILE_TYPES: tuple[FileType, ...] = (
    # --- Images (NEW) ---
    _ft("png", KIND_IMAGE, {"image/png"}),
    _ft("jpg", KIND_IMAGE, {"image/jpeg"}),
    _ft("jpeg", KIND_IMAGE, {"image/jpeg"}),
    _ft("gif", KIND_IMAGE, {"image/gif"}),
    _ft("webp", KIND_IMAGE, {"image/webp"}),
    # --- PDF ---
    _ft("pdf", KIND_PDF, {"application/pdf"}),
    # --- DOCX ---
    _ft(
        "docx",
        KIND_DOCX,
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    ),
    # --- Plain-text-decodable ---
    _ft("txt", KIND_TEXT, {"text/plain"}),
    _ft("md", KIND_TEXT, {"text/markdown"}, {"text/markdown", "text/plain"}),
    _ft("html", KIND_TEXT, {"text/html"}),
    _ft(
        "csv",
        KIND_TEXT,
        {"text/csv"},
        {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
    ),
    _ft("json", KIND_TEXT, {"application/json"}, {"application/json", "text/plain"}),
    _ft("xml", KIND_TEXT, {"application/xml", "text/xml"}, {"application/xml", "text/xml", "text/plain"}),
    _ft("rst", KIND_TEXT, {"text/x-rst"}, {"text/x-rst", "text/plain"}),
    _ft("tex", KIND_TEXT, {"application/x-tex"}, {"application/x-tex", "text/x-tex", "text/plain"}),
    _ft("yaml", KIND_TEXT, {"application/x-yaml", "text/yaml"}, {"application/x-yaml", "text/yaml", "text/plain"}),
    _ft("yml", KIND_TEXT, {"application/x-yaml", "text/yaml"}, {"application/x-yaml", "text/yaml", "text/plain"}),
    _ft("log", KIND_TEXT, {"text/plain"}),
    # --- Email (data-room only) ---
    _ft("msg", KIND_EMAIL, {"application/vnd.ms-outlook"}),
    _ft("eml", KIND_EMAIL, {"message/rfc822"}),
    # --- Audio / transcription (data-room only) ---
    _ft("mp3", KIND_AUDIO, {"audio/mpeg"}, {"audio/mpeg", "audio/mp3"}),
    _ft("mpeg", KIND_AUDIO, {"audio/mpeg"}, {"audio/mpeg", "video/mpeg"}),
    _ft("mpga", KIND_AUDIO, {"audio/mpeg"}),
    _ft("mp4", KIND_AUDIO, {"audio/mp4"}, {"audio/mp4", "video/mp4"}),
    _ft("m4a", KIND_AUDIO, {"audio/x-m4a"}, {"audio/x-m4a", "audio/mp4", "audio/m4a"}),
    _ft("wav", KIND_AUDIO, {"audio/wav", "audio/x-wav"}, {"audio/wav", "audio/x-wav", "audio/wave"}),
    _ft("webm", KIND_AUDIO, {"audio/webm"}, {"audio/webm", "video/webm"}),
    _ft("flac", KIND_AUDIO, {"audio/flac"}, {"audio/flac", "audio/x-flac"}),
    _ft("ogg", KIND_AUDIO, {"audio/ogg"}, {"audio/ogg", "application/ogg"}),
)

# --- Per-surface kind selections ----------------------------------------
DATA_ROOM_KINDS = frozenset({KIND_IMAGE, KIND_PDF, KIND_DOCX, KIND_TEXT, KIND_EMAIL, KIND_AUDIO})
# Chat can only render image/pdf/docx natively or decode text; it has no path
# for email (.msg/.eml) or audio, so those kinds are excluded.
CHAT_KINDS = frozenset({KIND_IMAGE, KIND_PDF, KIND_DOCX, KIND_TEXT})
# Meeting attachments are copied into the "minutes with Wilfred" chat thread,
# so they accept exactly what chat can consume.
MEETING_ATTACHMENT_KINDS = CHAT_KINDS

# Generic MIME tokens browsers send for unfamiliar types — always pass the
# per-extension cross-check (mirrors documents.views._GENERIC_MIME_TYPES).
GENERIC_MIMES = frozenset({"", "application/octet-stream"})


def _types_for_kinds(kinds):
    kinds = frozenset(kinds)
    return [ft for ft in FILE_TYPES if ft.kind in kinds]


def allowed_extensions(kinds) -> set[str]:
    """All extensions for the given kinds."""
    return {ft.ext for ft in _types_for_kinds(kinds)}


def canonical_mimes_for_kinds(kinds) -> set[str]:
    """Union of canonical MIME types for the given kinds (global allow-list)."""
    out: set[str] = set()
    for ft in _types_for_kinds(kinds):
        out |= ft.canonical_mimes
    return out


def accepted_mimes_for_kinds(kinds) -> frozenset:
    """Union of accepted (browser-reported) MIME types for the given kinds.

    This is the set used for content-type membership checks (e.g. chat
    attachment routing).
    """
    out: set[str] = set()
    for ft in _types_for_kinds(kinds):
        out |= ft.accepted_mimes
    return frozenset(out)


def extension_mime_map(kinds) -> dict[str, set[str]]:
    """Per-extension accepted-MIME map for the given kinds."""
    return {ft.ext: set(ft.accepted_mimes) for ft in _types_for_kinds(kinds)}


def global_allowed_mimes(kinds) -> frozenset:
    """Canonical MIME allow-list for the given kinds, plus generic tokens."""
    return frozenset(canonical_mimes_for_kinds(kinds) | {"application/octet-stream"})


# Reverse lookups -------------------------------------------------------
_BY_EXT = {ft.ext: ft for ft in FILE_TYPES}


def kind_for_extension(ext: str) -> str | None:
    """Return the kind for a (dotless, lowercase) extension, or None."""
    ft = _BY_EXT.get((ext or "").lower().lstrip("."))
    return ft.kind if ft else None


def kind_for_mime(mime: str) -> str | None:
    """Return the kind for a MIME type (first match in table), or None."""
    mime = (mime or "").lower()
    for ft in FILE_TYPES:
        if mime in ft.accepted_mimes:
            return ft.kind
    return None


def canonical_mime_for_extension(ext: str) -> str | None:
    """Return the canonical MIME type for an extension (e.g. "png" -> "image/png").

    Deterministic (first canonical MIME, sorted). Useful when a stored/browser
    content-type is missing or generic and a clean MIME is needed downstream.
    """
    ft = _BY_EXT.get((ext or "").lower().lstrip("."))
    if not ft or not ft.canonical_mimes:
        return None
    return sorted(ft.canonical_mimes)[0]


def is_image_extension(ext: str) -> bool:
    return kind_for_extension(ext) == KIND_IMAGE
