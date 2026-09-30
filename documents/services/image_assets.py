"""Persist embedded document images as Assets and describe them concurrently.

Provides the two-phase image describer whose ``.sink`` (see
core.docx.docx_to_markdown, core.pdf.pdf_to_text and
documents.services.chunking.pptx_to_markdown) stores each embedded image's bytes
on an Asset and leaves an inline ``[[image:<uuid>|Image N: <description>]]``
token in the extracted markdown — so the image is never lost and its description
stays searchable. The core is owner-agnostic: ``EmbeddedImageDescriber`` is the
data-room arm (Assets owned by a document version) and
``chat.assets.AttachmentImageDescriber`` the chat/meeting-attachment arm (Assets
owned by a ``ChatAttachment``). Shared by the docx, pdf, pptx and email paths.

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
  the real description in the combined text, before it is chunked / cached.

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

# Default cap on how many embedded images get a vision description per data-room
# document (or, for an email attachment tree, across the whole tree); beyond this
# they're still stored (bytes preserved) but labelled by format only, so a
# 200-image deck can't fan out into 200 vision calls. Overridden by
# settings.DOCUMENT_MAX_DESCRIBED_IMAGES; chat attachments use
# settings.CHAT_ATTACHMENT_MAX_DESCRIBED_IMAGES.
MAX_DESCRIBED_EMBEDDED_IMAGES = 50

# Bump this when the vision prompt (chat.services.IMAGE_DESCRIPTION_PROMPT)
# changes, so stale-prompt cached descriptions age out instead of being reused.
# v2: transcription-first prompt (2026-09-28).
_CACHE_KEY = "imgdesc:v2:{org_id}:{sha}"


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


class EmbeddedImageDescriberBase:
    """Owner-agnostic two-phase describer (see module docstring).

    One instance is threaded through a whole document (or email attachment tree)
    so ``Image N`` numbering, the description budget, and content-hash dedup all
    span the whole thing. Construct it, pass ``.sink`` to the extractor, then call
    ``run_descriptions()`` and ``substitute()`` on the returned text.

    Identical images are de-duplicated by content hash across the whole run: a
    logo repeated on every slide (or shared by two attachments of one email) is
    stored and described exactly once, and every occurrence re-emits the same
    token (the extractor's own ``idx`` is ignored; numbering is per unique image).
    The org-scoped cache extends that dedup across documents and chats in the org.

    Subclasses supply the owner: ``_store_asset`` persists the bytes as an Asset
    row (and may hand back a pre-existing row for the same bytes — its stored
    description is then authoritative and it is not re-described).
    """

    def __init__(
        self, *, user, org_id, model, max_described=MAX_DESCRIBED_EMBEDDED_IMAGES, label="",
        conversation_id=None,
    ):
        self.uploaded_by = user
        # Chat thread the vision calls are billed to (/cost); None for data rooms.
        self.conversation_id = conversation_id
        # None → org cache off.
        self.org_id = org_id
        # "" when there is no vision-capable model — assets are still stored.
        self.model = model or ""
        self.max_described = int(max_described)
        # Log-line identity only ("version_id=12" / "attachment_id=<uuid>").
        self.label = label

        # sha256 -> the token already emitted for that image.
        self.seen: dict[str, str] = {}
        self.count = 0
        # Distinct images awaiting a vision call: one record per pending image.
        self.pending: list[dict] = []

    def _store_asset(self, *, img_bytes: bytes, content_type: str, sha: str, description: str, alt_text: str):
        """Persist *img_bytes* as an Asset owned by this describer's owner and
        return it (blob written). Owner-specific; see the subclasses."""
        raise NotImplementedError

    @property
    def total(self) -> int:
        """How many images will actually get a vision call (the progress denom)."""
        return len(self.pending)

    # ── Phase 1: inline sink ──────────────────────────────────────────────
    def sink(self, image, idx: int) -> str:
        content_type = image.content_type or "application/octet-stream"
        with image.open() as f:
            raw = f.read()

        # Optimize before hashing/storing/describing: providers downsample past
        # ~1.15 MP anyway, so this only trims wasted bytes (smaller base64 upload,
        # lower tokens) and shrinks the working set held during phase 2. Format is
        # preserved (allow_transcode=False) so the stored content_type stays valid.
        img_bytes = raw
        from core.images import optimize_for_vision, to_vision_format

        # JPEG 2000 / TIFF / BMP (common in scanned PDFs) are accepted by no
        # vision provider and shown by few browsers: store and describe a
        # PNG/JPEG conversion instead, or the image is silently never described.
        converted = to_vision_format(raw)
        if converted is not None:
            img_bytes, content_type = converted

        opt = optimize_for_vision(img_bytes, allow_transcode=False)
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
        fallback = _fallback_label(content_type)
        cached_desc = self._cache_get(sha)
        description = cached_desc or fallback

        asset = self._store_asset(
            img_bytes=img_bytes,
            content_type=content_type,
            sha=sha,
            description=description,
            alt_text=(image.alt_text or "")[:1024],
        )
        # A reused pre-existing row (re-processed attachment) may already carry a
        # real description: it wins, and the image is not described again.
        label = (asset.description or "").strip() or description
        token = f"[[image:{asset.id}|Image {n}: {_sanitize_for_token(label)}]]"
        self.seen[sha] = token

        # Queue a vision call only when there's no description yet, a model is
        # configured, and we're within the per-run description budget. Over-cap /
        # no-model images keep their fallback label.
        already_described = bool(cached_desc) or label != fallback
        if not already_described and self.model and n <= self.max_described:
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

        from django.db import close_old_connections

        from chat.services import describe_image

        total = len(self.pending)
        max_workers = max(1, min(getattr(settings, "DOCUMENT_IMAGE_DESCRIBE_CONCURRENCY", 5), total))
        current = 0

        def _job(rec):
            # The vision call logs an LLMCallLog row on this thread; release the
            # thread-local DB connection afterwards (conn_max_age=0) so a pool
            # never leaves idle connections behind.
            try:
                return describe_image(
                    rec["bytes"], rec["content_type"], self.uploaded_by,
                    alt_text=rec["alt_text"], model=self.model,
                    conversation_id=self.conversation_id,
                )
            finally:
                close_old_connections()

        # Workers do ONLY the vision call; all shared state (results, cache,
        # progress) is mutated on this main thread as futures complete, so no
        # locking is needed. An exception on the main thread (e.g. Celery's soft
        # time limit) cancels what hasn't started instead of waiting it out.
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = {pool.submit(_job, rec): rec for rec in self.pending}
            for fut in as_completed(futures):
                rec = futures[fut]
                current += 1
                try:
                    desc = (fut.result() or "").strip()
                except Exception:
                    logger.exception("Failed to describe embedded image for %s", self.label)
                    desc = ""
                rec["bytes"] = None  # working set shrinks as descriptions land
                if desc:
                    results[rec["asset"].id] = desc
                    rec["asset"].description = desc
                    self._cache_set(rec["sha"], desc)
                if progress_cb is not None:
                    try:
                        progress_cb(current, total)
                    except Exception:  # progress is best-effort; never fail ingest
                        logger.debug("progress_cb failed for %s", self.label, exc_info=True)
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

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


class EmbeddedImageDescriber(EmbeddedImageDescriberBase):
    """Data-room arm: Assets owned by a ``DataRoomDocumentVersion``.

    Model = the org's ``document_image_description`` feature model; cache keyed by
    the document's org; cap ``settings.DOCUMENT_MAX_DESCRIBED_IMAGES`` (default
    ``MAX_DESCRIBED_EMBEDDED_IMAGES``).
    """

    def __init__(self, version, doc):
        from core.preferences import resolve_org_feature_model
        from documents.services.pii_scan import org_id_for_document

        org_id = org_id_for_document(doc)
        super().__init__(
            user=doc.uploaded_by,
            org_id=org_id,
            # "" when the org has no vision-capable model — assets are still stored.
            model=resolve_org_feature_model(org_id, "document_image_description"),
            max_described=getattr(settings, "DOCUMENT_MAX_DESCRIBED_IMAGES", MAX_DESCRIBED_EMBEDDED_IMAGES),
            label=f"version_id={version.id}",
        )
        self.version = version
        self.doc = doc

    def _store_asset(self, *, img_bytes, content_type, sha, description, alt_text):
        from django.core.files.base import ContentFile

        from chat.models import Asset

        asset = Asset(
            version=self.version,
            content_type=content_type,
            size_bytes=len(img_bytes),
            sha256=sha,
            description=description,
            alt_text=alt_text,
            created_by=self.uploaded_by,
        )
        asset.blob.save(f"{asset.id}.{_ext_for(content_type)}", ContentFile(img_bytes), save=True)
        return asset
