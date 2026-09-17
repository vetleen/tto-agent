"""Persist embedded document images as Assets during data-room ingestion.

Provides an ``EmbeddedImageDescriber`` whose ``.sink`` (see
core.docx.docx_to_markdown and core.pdf.pdf_to_text) stores each embedded image's
bytes on an Asset scoped to the document version and leaves an inline
``[[image:<uuid>|Image N: <description>]]`` token in the extracted markdown — so
the image is never lost and its description stays searchable. Shared by the docx,
pdf, pptx and email extraction paths.

Description is **two-phase** so the vision calls run concurrently instead of one
blocking round-trip per image inside the extraction walk:

* **Phase 1 (``sink``)** — runs inline during extraction (serial). It optimizes
  the image, dedupes by content hash, checks an org-scoped description cache, and
  stores the Asset immediately with the real UUID and a *fallback* label. Images
  that still need a description are queued in ``pending``.
* **Phase 2 (``run_descriptions``)** — after extraction, describes the queued
  images on a small thread pool (the calls are I/O-bound), updating each Asset's
  ``description`` and reporting progress.
* **Phase 3 (``substitute``)** — swaps each described image's fallback label for
  the real description in the combined text, before it is chunked.

Only the description (text) is guardrail/PII-scanned; the bytes are not
independently scanned (description-only for v1, same gap as standalone image
uploads).
"""

from __future__ import annotations

import hashlib
import logging
import re

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Collapses runs of whitespace (incl. newlines) in a token label.
_TOKEN_WS_RE = re.compile(r"\s+")

# Cap how many embedded images get a vision description per document (or, for an
# email attachment tree, across the whole tree); beyond this they're still stored
# (bytes preserved) but labelled by format only, so a 200-image deck can't fan out
# into 200 vision calls.
MAX_DESCRIBED_EMBEDDED_IMAGES = 50

# Bump this when the vision prompt (chat.services.describe_image) changes, so
# stale-prompt cached descriptions age out instead of being reused.
_CACHE_KEY = "imgdesc:v1:{org_id}:{sha}"


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


def _fallback_label(content_type: str) -> str:
    fmt = (content_type or "").split("/")[-1].upper().lstrip("X-")
    return f"{fmt} image" if fmt else "image"


class EmbeddedImageDescriber:
    """Two-phase describer for embedded document images (see module docstring).

    One instance is threaded through a whole document (or email attachment tree)
    so ``Image N`` numbering, the description budget, and content-hash dedup all
    span the whole thing. Construct it, pass ``.sink`` to
    ``documents.services.chunking.load_documents``, then call
    ``run_descriptions()`` and ``substitute()`` on the returned text.

    Identical images are de-duplicated by content hash across the whole run: a
    logo repeated on every slide (or shared by two attachments of one email) is
    stored and described exactly once, and every occurrence re-emits the same
    token. The org-scoped cache extends that dedup across documents in the org.
    """

    def __init__(self, version, doc):
        from core.preferences import resolve_org_feature_model
        from documents.services.pii_scan import org_id_for_document

        self.version = version
        self.doc = doc
        self.uploaded_by = doc.uploaded_by
        self.org_id = org_id_for_document(doc)
        # "" when the org has no vision-capable model — assets are still stored.
        self.model = resolve_org_feature_model(self.org_id, "document_image_description")

        # sha256 -> the token already emitted for that image.
        self.seen: dict[str, str] = {}
        self.count = 0
        # Distinct images awaiting a vision call: one record per pending image.
        self.pending: list[dict] = []

    @property
    def total(self) -> int:
        """How many images will actually get a vision call (the progress denom)."""
        return len(self.pending)

    # ── Phase 1: inline sink ──────────────────────────────────────────────
    def sink(self, image, idx: int) -> str:
        from django.core.files.base import ContentFile

        from chat.models import Asset

        content_type = image.content_type or "application/octet-stream"
        with image.open() as f:
            raw = f.read()

        # Optimize before hashing/storing/describing: providers downsample past
        # ~1.15 MP anyway, so this only trims wasted bytes (smaller base64 upload,
        # lower tokens) and shrinks the working set held during phase 2. Format is
        # preserved (allow_transcode=False) so the stored content_type stays valid.
        img_bytes = raw
        from core.images import optimize_for_vision

        opt = optimize_for_vision(raw, allow_transcode=False)
        if opt is not None and len(opt[0]) < len(img_bytes):
            img_bytes = opt[0]

        sha = hashlib.sha256(img_bytes).hexdigest()

        # Same bytes seen earlier this run: re-emit the original token (same asset
        # id and label), skipping a duplicate row and a duplicate vision call.
        cached_token = self.seen.get(sha)
        if cached_token is not None:
            return cached_token

        self.count += 1
        n = self.count

        # Org-scoped cache: reuse a real description for this image if the org has
        # one from an earlier document. Never reuse a format-only fallback.
        cached_desc = self._cache_get(sha)
        description = cached_desc or _fallback_label(content_type)

        asset = Asset(
            version=self.version,
            content_type=content_type,
            size_bytes=len(img_bytes),
            sha256=sha,
            description=description,
            alt_text=(image.alt_text or "")[:1024],
            created_by=self.uploaded_by,
        )
        asset.blob.save(f"{asset.id}.{_ext_for(content_type)}", ContentFile(img_bytes), save=True)
        token = f"[[image:{asset.id}|Image {n}: {_sanitize_for_token(description)}]]"
        self.seen[sha] = token

        # Queue a vision call only when there's no cached description, a model is
        # configured, and we're within the per-run description budget. Over-cap /
        # no-model images keep their fallback label.
        if not cached_desc and self.model and n <= MAX_DESCRIBED_EMBEDDED_IMAGES:
            self.pending.append({
                "asset": asset, "sha": sha, "bytes": img_bytes,
                "content_type": content_type, "alt_text": image.alt_text, "n": n,
            })
        return token

    # ── Phase 2: concurrent description ───────────────────────────────────
    def run_descriptions(self, progress_cb=None) -> dict:
        """Describe the queued images concurrently. Returns ``{asset_id: desc}``
        for successes, updates each described Asset's ``description``, and
        populates the org cache. ``progress_cb(current, total)`` is called on the
        main thread as each description completes.
        """
        results: dict = {}
        if not self.pending:
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        from chat.services import describe_image

        total = len(self.pending)
        max_workers = max(1, min(getattr(settings, "DOCUMENT_IMAGE_DESCRIBE_CONCURRENCY", 5), total))
        current = 0
        # Workers do ONLY the pure vision call (network I/O, no DB); all shared
        # state (results, cache, progress) is mutated on this main thread as
        # futures complete, so no locking is needed.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    describe_image, rec["bytes"], rec["content_type"],
                    self.uploaded_by, alt_text=rec["alt_text"], model=self.model,
                ): rec
                for rec in self.pending
            }
            for fut in as_completed(futures):
                rec = futures[fut]
                current += 1
                try:
                    desc = (fut.result() or "").strip()
                except Exception:
                    logger.exception("Failed to describe embedded image for version %s", self.version.id)
                    desc = ""
                if desc:
                    results[rec["asset"].id] = desc
                    rec["asset"].description = desc
                    self._cache_set(rec["sha"], desc)
                if progress_cb is not None:
                    try:
                        progress_cb(current, total)
                    except Exception:  # progress is best-effort; never fail ingest
                        logger.debug("progress_cb failed for version %s", self.version.id, exc_info=True)

        described = [rec["asset"] for rec in self.pending if rec["asset"].id in results]
        if described:
            from chat.models import Asset

            Asset.objects.bulk_update(described, ["description"])
        return results

    # ── Phase 3: label substitution ───────────────────────────────────────
    def substitute(self, text: str, described: dict) -> str:
        """Replace each described image's fallback token label with its real
        description in *text*. Over-cap / failed / cache-hit images are untouched
        (they already carry their final label from phase 1).
        """
        if not described:
            return text
        for rec in self.pending:
            desc = described.get(rec["asset"].id)
            if not desc:
                continue
            final = f"[[image:{rec['asset'].id}|Image {rec['n']}: {_sanitize_for_token(desc)}]]"
            pattern = re.compile(r"\[\[image:" + re.escape(str(rec["asset"].id)) + r"\|[^\]]*\]\]")
            text = pattern.sub(lambda _m, f=final: f, text)
        return text

    # ── Org-scoped description cache ──────────────────────────────────────
    def _cache_get(self, sha: str):
        if self.org_id is None:
            return None
        try:
            return cache.get(_CACHE_KEY.format(org_id=self.org_id, sha=sha)) or None
        except Exception:  # pragma: no cover - cache is already fail-open
            return None

    def _cache_set(self, sha: str, desc: str) -> None:
        if self.org_id is None or not desc:
            return
        try:
            cache.set(
                _CACHE_KEY.format(org_id=self.org_id, sha=sha),
                desc,
                getattr(settings, "IMGDESC_CACHE_TTL", 2_592_000),
            )
        except Exception:  # pragma: no cover - cache is already fail-open
            pass
