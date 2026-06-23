"""Persist embedded document images as ImageAssets during data-room ingestion.

Provides an ``image_sink`` (see core.docx.docx_to_markdown and core.pdf.pdf_to_text)
that stores each embedded image's bytes on an ImageAsset scoped to the document
version and leaves an inline ``[[image:<uuid>|Image N: <description>]]`` token in
the extracted markdown — so the image is never lost and its description stays
searchable. Shared by both the docx and pdf extraction paths. Only the
description (text) is guardrail/PII-scanned; the bytes are not independently
scanned (description-only for v1, same gap as standalone image uploads).
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Cap how many embedded images get a vision description per document; beyond
# this they're still stored (bytes preserved) but labelled by format only, so a
# 200-image deck can't fan out into 200 vision calls.
MAX_DESCRIBED_EMBEDDED_IMAGES = 20


def _ext_for(content_type: str) -> str:
    return (content_type.split("/")[-1] if content_type else "bin").lower().lstrip("x-") or "bin"


def _sanitize_for_token(text: str) -> str:
    """Strip characters that would break the [[image:uuid|desc]] token grammar."""
    return text.replace("[", "(").replace("]", ")").replace("|", "/").strip()


def image_asset_sink(version, doc):
    """Return an image_sink that stores each embedded image as an ImageAsset
    scoped to *version* and emits a ``[[image:uuid|Image N: desc]]`` token.

    Format-neutral: used by both the docx (core.docx.docx_to_markdown) and pdf
    (core.pdf.pdf_to_text) extraction paths. Descriptions (capped at
    MAX_DESCRIBED_EMBEDDED_IMAGES) use the org's vision-capable describer when
    one is configured; otherwise images are still stored with a format-only
    label so nothing is lost.
    """
    from django.core.files.base import ContentFile

    from chat.models import ImageAsset
    from chat.services import describe_image
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import org_id_for_document

    org_id = org_id_for_document(doc)
    # "" when the org has no vision-capable model — assets are still stored.
    model = resolve_org_feature_model(org_id, "document_image_description")

    def sink(image, idx: int) -> str:
        content_type = image.content_type or "application/octet-stream"
        with image.open() as f:
            img_bytes = f.read()

        description = ""
        if model and idx <= MAX_DESCRIBED_EMBEDDED_IMAGES:
            try:
                description = describe_image(
                    img_bytes, content_type, doc.uploaded_by,
                    alt_text=image.alt_text, model=model,
                ) or ""
            except Exception:
                logger.exception("Failed to describe embedded image for version %s", version.id)
        if not description:
            fmt = content_type.split("/")[-1].upper().lstrip("X-")
            description = f"{fmt} image" if fmt else "image"

        asset = ImageAsset(
            version=version,
            content_type=content_type,
            size_bytes=len(img_bytes),
            sha256=hashlib.sha256(img_bytes).hexdigest(),
            description=description,
            alt_text=(image.alt_text or "")[:1024],
            created_by=doc.uploaded_by,
        )
        asset.blob.save(f"{asset.id}.{_ext_for(content_type)}", ContentFile(img_bytes), save=True)
        return f"[[image:{asset.id}|Image {idx}: {_sanitize_for_token(description)}]]"

    return sink
