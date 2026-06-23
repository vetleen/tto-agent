"""Persist chat-embedded images as Assets (canvas docx import, and
docx/pdf chat attachments).

Mirrors documents.services.image_assets.image_asset_sink, but scopes assets to a
ChatCanvas or ChatMessage and uses the user-scoped image describer (these are
interactive, user-initiated actions).
"""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

# Collapses runs of whitespace (incl. newlines) in a token label.
_TOKEN_WS_RE = re.compile(r"\s+")


def _ext_for(content_type: str) -> str:
    return (content_type.split("/")[-1] if content_type else "bin").lower().lstrip("x-") or "bin"


def _sanitize_for_token(text: str) -> str:
    """Strip token-breaking chars and collapse whitespace — the [[image:uuid|desc]]
    token is single-line; an embedded newline (multi-paragraph description) breaks
    the Markdown renderer that turns it into an <img>."""
    text = text.replace("[", "(").replace("]", ")").replace("|", "/")
    return _TOKEN_WS_RE.sub(" ", text).strip()


def image_token(asset_id, label: str) -> str:
    """Build a canvas image reference token ``[[image:<uuid>|<label>]]``."""
    return f"[[image:{asset_id}|{_sanitize_for_token(label)}]]"


# Shown (in preview, chat, and .docx export) when a token can't be resolved —
# the source was deleted or the viewer lost access. Threads/canvases are
# fleeting, so a neutral note is friendlier than a broken-image icon.
IMAGE_UNAVAILABLE_TEXT = "[An image was here, but it can no longer be accessed]"


def get_or_create_version_image_token(*, version_id, mime="", description="") -> str:
    """Return a stable ``[[image:uuid|]]`` token for a data-room image.

    Lazily creates a single *reference* Asset (no blob) scoped to the
    version — the bytes stay on the document's native file; the asset is just the
    addressable token target. Idempotent: the same version always yields the same
    asset (and token), so the model can reuse the token across the thread, and
    existing images work with no backfill.

    The caption is left **empty**: it's not a stored alt-text. The model may add
    its own caption between the ``|`` and ``]]`` when it embeds the token.
    """
    from chat.models import Asset

    asset = Asset.objects.filter(
        version_id=version_id, blob="", kind=Asset.KIND_IMAGE
    ).first()
    if asset is None:
        asset = Asset.objects.create(
            version_id=version_id, kind=Asset.KIND_IMAGE,
            content_type=mime or "", description=description or "",
        )
    return image_token(asset.id, "")


def image_asset_source(asset):
    """Resolve where an Asset's bytes live → ``(filefield_or_None, content_type)``.

    A normal asset owns its ``blob``. A *reference* asset (empty blob,
    version-owned) falls back to the data-room version's native image
    (``native_blob`` else the document's ``original_file``).
    """
    if asset.blob:
        return asset.blob, (asset.content_type or "application/octet-stream")
    if asset.version_id:
        version = asset.version
        document = version.document
        source = version.native_blob if version.native_blob else document.original_file
        ct = asset.content_type or document.mime_type or "image/png"
        if source:
            return source, ct
    return None, (asset.content_type or "application/octet-stream")


# Shown (in chat and .docx export) when a file token can't be resolved — the
# source was deleted or the viewer lost access. Mirrors IMAGE_UNAVAILABLE_TEXT.
FILE_UNAVAILABLE_TEXT = "[A file was here, but it can no longer be accessed]"


def file_token(asset_id, label: str) -> str:
    """Build a data-room file-download reference token ``[[file:<uuid>|<label>]]``."""
    return f"[[file:{asset_id}|{_sanitize_for_token(label)}]]"


def get_or_create_version_file_token(*, version_id, mime="", filename="") -> str:
    """Return a stable ``[[file:uuid|]]`` token for downloading a data-room document.

    Mirrors get_or_create_version_image_token but mints a *file* reference asset
    (``kind=file``), kept distinct from any image reference for the same version so
    the two never share a uuid. The download resolves the document's LATEST native
    file at serve time (see latest_native_file), so an old token tracks the newest
    upload — like a symlink, not a snapshot.

    The label is left **empty**; the model may add its own between ``|`` and ``]]``.
    """
    from chat.models import Asset

    asset = Asset.objects.filter(
        version_id=version_id, blob="", kind=Asset.KIND_FILE
    ).first()
    if asset is None:
        asset = Asset.objects.create(
            version_id=version_id, kind=Asset.KIND_FILE,
            content_type=mime or "", description=(filename or "")[:1024],
        )
    return file_token(asset.id, "")


def latest_native_file(document):
    """``(filefield_or_None, filename, content_type)`` for *document*'s newest
    downloadable native file.

    Walks versions newest-first for one carrying native bytes, else falls back to
    the document's ``original_file``. Markdown/canvas-only edit versions (empty
    ``native_blob``) are skipped — they have no downloadable original form — so the
    result is the latest *uploaded* file, which is what a ``[[file:]]`` link offers.
    """
    version = (
        document.versions.exclude(native_blob="")
        .order_by("-version_index")
        .first()
    )
    if version and version.native_blob:
        return (
            version.native_blob,
            version.native_filename or document.original_filename or "",
            version.mime_type or document.mime_type or "application/octet-stream",
        )
    if document.original_file:
        return (
            document.original_file,
            document.original_filename or "",
            document.mime_type or "application/octet-stream",
        )
    return None, "", "application/octet-stream"


def file_asset_source(asset):
    """Resolve a file reference asset to its document's latest native file →
    ``(filefield_or_None, filename, content_type)``. Anchored on a version for the
    one-owner constraint + ACL walk, but the *version is ignored* for resolution so
    a stale token always serves the newest upload."""
    if asset.version_id:
        return latest_native_file(asset.version.document)
    return None, "", "application/octet-stream"


def file_token_for_document(document) -> str | None:
    """Return a ``[[file:uuid|]]`` download handle for *document*, or ``None`` if it
    has no downloadable native file (e.g. a from-scratch canvas doc).

    Surfaced by the document tools so the model can offer a download. Anchors the
    reference on the document's current version (any version works — serving always
    resolves the document's latest native file)."""
    source, filename, ct = latest_native_file(document)
    if source is None:
        return None
    version_id = document.current_version_id or document.active_searchable_version_id
    if not version_id:
        return None
    return get_or_create_version_file_token(version_id=version_id, mime=ct, filename=filename)


def store_canvas_image(
    canvas, *, img_bytes, content_type, description="", alt_text="", created_by=None, dedupe=True
):
    """Persist *img_bytes* as an Asset scoped to *canvas*; return the asset.

    When *dedupe* is set, an existing canvas asset with the same bytes (sha256)
    is reused rather than storing a second copy — so re-inserting the same image
    doesn't bloat storage.
    """
    from django.core.files.base import ContentFile

    from chat.models import Asset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = Asset.objects.filter(canvas=canvas, sha256=sha).first()
        if existing is not None:
            return existing

    asset = Asset(
        canvas=canvas,
        content_type=ct,
        size_bytes=len(img_bytes),
        sha256=sha,
        description=description or "",
        alt_text=(alt_text or "")[:1024],
        created_by=created_by,
    )
    asset.blob.save(f"{asset.id}.{_ext_for(ct)}", ContentFile(img_bytes), save=True)
    return asset


def store_thread_image(
    thread, *, img_bytes, content_type, description="", alt_text="", created_by=None, dedupe=True
):
    """Persist *img_bytes* as an Asset scoped to *thread*; return the asset.

    Used for tool-generated images (e.g. ``chat_generate_image``): the assistant
    message doesn't exist yet when the tool runs, but the thread does, so it owns
    the asset. The returned asset's id is the stable token target the model
    embeds in its reply.

    When *dedupe* is set, an existing thread asset with the same bytes (sha256)
    is reused rather than storing a second copy.
    """
    from django.core.files.base import ContentFile

    from chat.models import Asset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = Asset.objects.filter(thread=thread, sha256=sha).first()
        if existing is not None:
            return existing

    asset = Asset(
        thread=thread,
        content_type=ct,
        size_bytes=len(img_bytes),
        sha256=sha,
        description=description or "",
        alt_text=(alt_text or "")[:1024],
        created_by=created_by,
    )
    asset.blob.save(f"{asset.id}.{_ext_for(ct)}", ContentFile(img_bytes), save=True)
    return asset


def store_message_image(
    message, *, img_bytes, content_type, description="", alt_text="", created_by=None, dedupe=True
):
    """Persist *img_bytes* as an Asset scoped to *message*; return the asset.

    Used for images extracted from a user's docx/pdf attachment: the user
    message exists by the time extraction runs, so it owns the asset and the
    token renders inline in the conversation (see serve_image_asset / the
    frontend token pipeline).

    When *dedupe* is set, an existing message asset with the same bytes (sha256)
    is reused rather than storing a second copy.
    """
    from django.core.files.base import ContentFile

    from chat.models import Asset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = Asset.objects.filter(message=message, sha256=sha).first()
        if existing is not None:
            return existing

    asset = Asset(
        message=message,
        content_type=ct,
        size_bytes=len(img_bytes),
        sha256=sha,
        description=description or "",
        alt_text=(alt_text or "")[:1024],
        created_by=created_by,
    )
    asset.blob.save(f"{asset.id}.{_ext_for(ct)}", ContentFile(img_bytes), save=True)
    return asset


def _describe_and_store_sink(store, user, *, max_described, model=None):
    """Shared body for the canvas/message asset sinks: describe (capped) then
    persist via *store*, a ``(img_bytes, content_type, description, alt_text)``
    -> asset callable. Emits a ``[[image:uuid|Image N: desc]]`` token."""
    from chat.services import describe_image

    def sink(image, idx: int) -> str:
        ct = image.content_type or "application/octet-stream"
        with image.open() as f:
            img_bytes = f.read()

        description = ""
        if idx <= max_described:
            try:
                description = describe_image(img_bytes, ct, user, alt_text=image.alt_text, model=model) or ""
            except Exception:
                logger.exception("Failed to describe embedded image")
        if not description:
            fmt = ct.split("/")[-1].upper().lstrip("X-")
            description = f"{fmt} image" if fmt else "image"

        asset = store(img_bytes, ct, description, image.alt_text or "")
        return image_token(asset.id, f"Image {idx}: {description}")

    return sink


def canvas_asset_sink(canvas, user, *, max_described: int = 25):
    """Return a docx/pdf image_sink that stores each embedded image as an
    Asset scoped to *canvas* and emits a ``[[image:uuid|Image N: desc]]``
    token.

    Descriptions (capped at *max_described*) use the user's vision describer;
    beyond that, images are still stored with a format-only label.
    """

    def store(img_bytes, ct, description, alt_text):
        return store_canvas_image(
            canvas,
            img_bytes=img_bytes,
            content_type=ct,
            description=description,
            alt_text=alt_text,
            created_by=user,
            # docx import emits each embedded image once; no cross-image dedupe.
            dedupe=False,
        )

    return _describe_and_store_sink(store, user, max_described=max_described)


def message_image_asset_sink(message, user, *, max_described: int = 10, model=None):
    """Return a docx/pdf image_sink that stores each embedded image as an
    Asset scoped to *message* and emits a ``[[image:uuid|Image N: desc]]``
    token — the chat/meeting attachment counterpart of the data-room
    image_asset_sink. Descriptions are capped at *max_described*.
    """

    def store(img_bytes, ct, description, alt_text):
        return store_message_image(
            message,
            img_bytes=img_bytes,
            content_type=ct,
            description=description,
            alt_text=alt_text,
            created_by=user,
            # core.pdf/core.docx already dedupe within a document; keep this on so
            # a re-enriched attachment can't double-store.
            dedupe=True,
        )

    return _describe_and_store_sink(store, user, max_described=max_described, model=model)
