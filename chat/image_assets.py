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
