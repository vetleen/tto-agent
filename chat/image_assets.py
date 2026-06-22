"""Persist canvas-embedded images as ImageAssets (canvas docx import).

Mirrors documents.services.image_assets.docx_asset_sink, but scopes assets to a
ChatCanvas and uses the user-scoped image describer (canvas import is an
interactive, user-initiated action).
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)


def _ext_for(content_type: str) -> str:
    return (content_type.split("/")[-1] if content_type else "bin").lower().lstrip("x-") or "bin"


def _sanitize_for_token(text: str) -> str:
    return text.replace("[", "(").replace("]", ")").replace("|", "/").strip()


def image_token(asset_id, label: str) -> str:
    """Build a canvas image reference token ``[[image:<uuid>|<label>]]``."""
    return f"[[image:{asset_id}|{_sanitize_for_token(label)}]]"


# Shown (in preview, chat, and .docx export) when a token can't be resolved —
# the source was deleted or the viewer lost access. Threads/canvases are
# fleeting, so a neutral note is friendlier than a broken-image icon.
IMAGE_UNAVAILABLE_TEXT = "[An image was here, but it can no longer be accessed]"


def get_or_create_version_image_token(*, version_id, mime="", description="", filename="") -> str:
    """Return a stable ``[[image:uuid|label]]`` token for a data-room image.

    Lazily creates a single *reference* ImageAsset (no blob) scoped to the
    version — the bytes stay on the document's native file; the asset is just the
    addressable token target. Idempotent: the same version always yields the same
    asset (and token), so the model can reuse the token across the thread, and
    existing images work with no backfill.
    """
    from chat.models import ImageAsset

    asset = ImageAsset.objects.filter(version_id=version_id, blob="").first()
    if asset is None:
        asset = ImageAsset.objects.create(
            version_id=version_id, content_type=mime or "", description=description or "",
        )
    return image_token(asset.id, (description or filename or "image")[:120])


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


def canvas_asset_sink(canvas, user, *, max_described: int = 25):
    """Return a docx image_sink that stores each embedded image as an ImageAsset
    scoped to *canvas* and emits a ``[[image:uuid|Image N: desc]]`` token.

    Descriptions (capped at *max_described*) use the user's vision describer;
    beyond that, images are still stored with a format-only label.
    """
    from chat.services import describe_image

    def sink(image, idx: int) -> str:
        ct = image.content_type or "application/octet-stream"
        with image.open() as f:
            img_bytes = f.read()

        description = ""
        if idx <= max_described:
            try:
                description = describe_image(img_bytes, ct, user, alt_text=image.alt_text) or ""
            except Exception:
                logger.exception("Failed to describe canvas image")
        if not description:
            fmt = ct.split("/")[-1].upper().lstrip("X-")
            description = f"{fmt} image" if fmt else "image"

        asset = store_canvas_image(
            canvas,
            img_bytes=img_bytes,
            content_type=ct,
            description=description,
            alt_text=image.alt_text or "",
            created_by=user,
            # docx import emits each embedded image once; no cross-image dedupe.
            dedupe=False,
        )
        return image_token(asset.id, f"Image {idx}: {description}")

    return sink
