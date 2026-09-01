"""Persist embedded document images as Assets during data-room ingestion.

Provides an ``image_sink`` (see core.docx.docx_to_markdown and core.pdf.pdf_to_text)
that stores each embedded image's bytes on an Asset scoped to the document
version and leaves an inline ``[[image:<uuid>|Image N: <description>]]`` token in
the extracted markdown — so the image is never lost and its description stays
searchable. Shared by both the docx and pdf extraction paths. Only the
description (text) is guardrail/PII-scanned; the bytes are not independently
scanned (description-only for v1, same gap as standalone image uploads).
"""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

# Collapses runs of whitespace (incl. newlines) in a token label.
_TOKEN_WS_RE = re.compile(r"\s+")

# Cap how many embedded images get a vision description per document (or, for an
# email attachment tree, across the whole tree — see the ``counter`` arg); beyond
# this they're still stored (bytes preserved) but labelled by format only, so a
# 200-image deck can't fan out into 200 vision calls.
MAX_DESCRIBED_EMBEDDED_IMAGES = 50


def _ext_for(content_type: str) -> str:
    return (content_type.split("/")[-1] if content_type else "bin").lower().lstrip("x-") or "bin"


def _sanitize_for_token(text: str) -> str:
    """Strip characters that would break the [[image:uuid|desc]] token grammar.

    The token is single-line: as well as the [], | delimiters, collapse all
    whitespace (incl. newlines) to single spaces. A multi-paragraph vision
    description would otherwise embed a blank line in the token, which breaks the
    Markdown renderer that turns it into an <img> (the blank line splits the
    placeholder span across paragraphs and it renders as literal text).
    """
    text = text.replace("[", "(").replace("]", ")").replace("|", "/")
    return _TOKEN_WS_RE.sub(" ", text).strip()


def image_asset_sink(version, doc):
    """Return an image_sink that stores each embedded image as an Asset
    scoped to *version* and emits a ``[[image:uuid|Image N: desc]]`` token.

    Format-neutral: used by the docx (core.docx.docx_to_markdown), pdf
    (core.pdf.pdf_to_text), and pptx (documents.services.chunking) extraction
    paths, and threaded once through a whole email attachment tree
    (documents.services.chunking) so a single sink spans every nested file.
    Descriptions (capped at MAX_DESCRIBED_EMBEDDED_IMAGES) use the org's
    vision-capable describer when one is configured; otherwise images are still
    stored with a format-only label so nothing is lost.

    Identical images are de-duplicated by content hash across the whole run: a
    logo repeated on every slide (or shared by two attachments of one email) is
    stored and vision-described exactly once, and every occurrence re-emits the
    same token. This stops a repeated image from burning a vision call apiece
    and from consuming the description budget, which now applies to *distinct*
    images. The ``idx`` the loader passes is ignored for numbering — the sink
    owns a distinct-image counter, so the ``Image N`` sequence has no gaps and
    spans the whole email tree (the pdf loader keeps its own earlier dedup; the
    sink simply never sees a pdf duplicate).
    """
    from django.core.files.base import ContentFile

    from chat.models import Asset
    from chat.services import describe_image
    from core.preferences import resolve_org_feature_model
    from documents.services.pii_scan import org_id_for_document

    org_id = org_id_for_document(doc)
    # "" when the org has no vision-capable model — assets are still stored.
    model = resolve_org_feature_model(org_id, "document_image_description")

    # Per-run state, shared across the whole email tree via this single sink
    # instance: sha256 -> the token already emitted for that image, and a
    # counter of distinct images (drives ``Image N`` and the description cap).
    seen: dict[str, str] = {}
    count = {"n": 0}

    def sink(image, idx: int) -> str:
        content_type = image.content_type or "application/octet-stream"
        with image.open() as f:
            img_bytes = f.read()
        sha = hashlib.sha256(img_bytes).hexdigest()

        # Same bytes seen earlier this run: re-emit the original token (same
        # asset id and label), skipping the vision call and a duplicate row.
        cached = seen.get(sha)
        if cached is not None:
            return cached

        count["n"] += 1
        n = count["n"]

        description = ""
        if model and n <= MAX_DESCRIBED_EMBEDDED_IMAGES:
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

        asset = Asset(
            version=version,
            content_type=content_type,
            size_bytes=len(img_bytes),
            sha256=sha,
            description=description,
            alt_text=(image.alt_text or "")[:1024],
            created_by=doc.uploaded_by,
        )
        asset.blob.save(f"{asset.id}.{_ext_for(content_type)}", ContentFile(img_bytes), save=True)
        token = f"[[image:{asset.id}|Image {n}: {_sanitize_for_token(description)}]]"
        seen[sha] = token
        return token

    return sink
