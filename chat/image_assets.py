"""Persist chat-embedded images as ImageAssets (canvas docx import, and
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

    Lazily creates a single *reference* ImageAsset (no blob) scoped to the
    version — the bytes stay on the document's native file; the asset is just the
    addressable token target. Idempotent: the same version always yields the same
    asset (and token), so the model can reuse the token across the thread, and
    existing images work with no backfill.

    The caption is left **empty**: it's not a stored alt-text. The model may add
    its own caption between the ``|`` and ``]]`` when it embeds the token.
    """
    from chat.models import ImageAsset

    asset = ImageAsset.objects.filter(version_id=version_id, blob="").first()
    if asset is None:
        asset = ImageAsset.objects.create(
            version_id=version_id, content_type=mime or "", description=description or "",
        )
    return image_token(asset.id, "")


def image_asset_source(asset):
    """Resolve where an ImageAsset's bytes live → ``(filefield_or_None, content_type)``.

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


def store_canvas_image(
    canvas, *, img_bytes, content_type, description="", alt_text="", created_by=None, dedupe=True
):
    """Persist *img_bytes* as an ImageAsset scoped to *canvas*; return the asset.

    When *dedupe* is set, an existing canvas asset with the same bytes (sha256)
    is reused rather than storing a second copy — so re-inserting the same image
    doesn't bloat storage.
    """
    from django.core.files.base import ContentFile

    from chat.models import ImageAsset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = ImageAsset.objects.filter(canvas=canvas, sha256=sha).first()
        if existing is not None:
            return existing

    asset = ImageAsset(
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
    """Persist *img_bytes* as an ImageAsset scoped to *thread*; return the asset.

    Used for tool-generated images (e.g. ``chat_generate_image``): the assistant
    message doesn't exist yet when the tool runs, but the thread does, so it owns
    the asset. The returned asset's id is the stable token target the model
    embeds in its reply.

    When *dedupe* is set, an existing thread asset with the same bytes (sha256)
    is reused rather than storing a second copy.
    """
    from django.core.files.base import ContentFile

    from chat.models import ImageAsset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = ImageAsset.objects.filter(thread=thread, sha256=sha).first()
        if existing is not None:
            return existing

    asset = ImageAsset(
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
    """Persist *img_bytes* as an ImageAsset scoped to *message*; return the asset.

    Used for images extracted from a user's docx/pdf attachment: the user
    message exists by the time extraction runs, so it owns the asset and the
    token renders inline in the conversation (see serve_image_asset / the
    frontend token pipeline).

    When *dedupe* is set, an existing message asset with the same bytes (sha256)
    is reused rather than storing a second copy.
    """
    from django.core.files.base import ContentFile

    from chat.models import ImageAsset

    ct = content_type or "application/octet-stream"
    sha = hashlib.sha256(img_bytes).hexdigest()
    if dedupe:
        existing = ImageAsset.objects.filter(message=message, sha256=sha).first()
        if existing is not None:
            return existing

    asset = ImageAsset(
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
    ImageAsset scoped to *canvas* and emits a ``[[image:uuid|Image N: desc]]``
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
    ImageAsset scoped to *message* and emits a ``[[image:uuid|Image N: desc]]``
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
