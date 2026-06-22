"""Document versioning service.

Owns the version lifecycle on top of the two-pointer model:

- ``current_version`` — the working/editing head, advances immediately on save.
- ``active_searchable_version`` — what retrieval reads, advances only when a
  version finishes processing+scan as READY and not quarantined (see
  ``documents.tasks.finalize_document_metadata``).

Callers are responsible for access checks (this layer is access-agnostic, like
``chat.services.save_canvas_to_data_room``).
"""
from __future__ import annotations

import logging

from django.db import transaction
from django.db.models import Max
from django.utils import timezone

logger = logging.getLogger(__name__)


def _next_version_index(document_id: int) -> int:
    """Next version_index for a document. Call inside a locked transaction."""
    from documents.models import DataRoomDocumentVersion

    mx = DataRoomDocumentVersion.objects.filter(document_id=document_id).aggregate(
        Max("version_index")
    )["version_index__max"]
    return 0 if mx is None else mx + 1


def _enqueue_processing(version_id: int) -> None:
    """Start async processing for a version (Celery), with a sync fallback."""
    from documents.models import DataRoomDocumentVersion

    try:
        from documents.tasks import process_document_version_task

        process_document_version_task.delay(version_id)
    except ImportError:
        try:
            from documents.services.process_document import process_document_version

            process_document_version(version_id)
        except Exception as exc:  # pragma: no cover - defensive sync fallback
            logger.exception(
                "versioning: sync processing failed for version_id=%s", version_id
            )
            DataRoomDocumentVersion.objects.filter(pk=version_id).update(
                status="failed", processing_error=str(exc)[:2000]
            )
    except Exception as exc:
        logger.exception(
            "versioning: failed to enqueue processing for version_id=%s", version_id
        )
        DataRoomDocumentVersion.objects.filter(pk=version_id).update(
            status="failed", processing_error=str(exc)[:2000]
        )


def create_version(
    document,
    *,
    content: str,
    origin: str,
    created_by=None,
    native_blob=None,
    native_filename: str = "",
    mime_type: str = "",
    size_bytes: int | None = None,
    enqueue: bool = True,
):
    """Create a new working version from markdown ``content`` and start processing.

    Advances ``current_version`` immediately. The new version is processed
    asynchronously (chunk → embed is_searchable=False → guardrails → PII); the
    ``active_searchable_version`` pointer only advances when finalize releases it
    as READY and not quarantined. Returns the created ``DataRoomDocumentVersion``.
    """
    from documents.models import DataRoomDocument, DataRoomDocumentVersion

    with transaction.atomic():
        # Lock the document row to serialize version_index assignment + pointer moves.
        doc = DataRoomDocument.objects.select_for_update().get(pk=document.pk)
        idx = _next_version_index(doc.pk)
        version = DataRoomDocumentVersion.objects.create(
            document=doc,
            version_index=idx,
            origin=origin,
            content=content or "",
            native_filename=native_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            created_by=created_by,
        )
        if native_blob is not None:
            version.native_blob = native_blob
            version.save(update_fields=["native_blob", "updated_at"])
        # Working head advances immediately; searchable pointer waits for finalize.
        DataRoomDocument.objects.filter(pk=doc.pk).update(current_version=version)

    if enqueue:
        _enqueue_processing(version.id)
    return version


def rename_document(document, name: str):
    """Set the document's mutable display ``name`` (original_filename is preserved)."""
    from documents.views import _safe_original_filename

    document.name = _safe_original_filename(name, max_length=75)
    document.save(update_fields=["name", "updated_at"])
    return document


def restore_version(document, target):
    """Make a prior READY version the live one again — instant pointer flip.

    Flips the pgvector ``is_searchable`` flag (no re-embed) and moves both
    pointers to ``target``. Only READY, non-quarantined versions may be restored.
    Raises ValueError otherwise.
    """
    from documents.models import (
        DataRoomDocument,
        DataRoomDocumentVersion,
    )
    from documents.services.vector_store import set_searchable_for_version

    if target.status != DataRoomDocument.Status.READY or target.is_quarantined:
        raise ValueError(
            f"Version {target.version_index} is not restorable "
            f"(status={target.status}, quarantined={target.is_quarantined})."
        )

    with transaction.atomic():
        doc = DataRoomDocument.objects.select_for_update().get(pk=document.pk)
        old_active_id = doc.active_searchable_version_id
        if old_active_id and old_active_id != target.id:
            DataRoomDocumentVersion.objects.filter(pk=old_active_id).update(is_searchable=False)
        DataRoomDocumentVersion.objects.filter(pk=target.id).update(is_searchable=True)
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            active_searchable_version=target,
            current_version=target,
            status=target.status,
            token_count=target.token_count,
            parser_type=target.parser_type,
            chunking_strategy=target.chunking_strategy,
            embedding_model=target.embedding_model,
            processed_at=target.processed_at,
        )

    # Flip pgvector flags after the app-DB pointer is committed (authoritative).
    if old_active_id and old_active_id != target.id:
        set_searchable_for_version(old_active_id, False)
    set_searchable_for_version(target.id, True)
    recompute_document_sensitivity(document.pk)
    return target


def advance_active_to(document_id: int, version) -> None:
    """Make ``version`` the document's active searchable version.

    Flips the pgvector ``is_searchable`` flag old→False/new→True (no re-embed) and
    moves the pointer. Called by finalize when a version is released READY and clean.
    """
    from documents.models import DataRoomDocument, DataRoomDocumentVersion
    from documents.services.vector_store import set_searchable_for_version

    with transaction.atomic():
        doc = DataRoomDocument.objects.select_for_update().get(pk=document_id)
        old_active_id = doc.active_searchable_version_id
        if old_active_id and old_active_id != version.id:
            DataRoomDocumentVersion.objects.filter(pk=old_active_id).update(is_searchable=False)
        DataRoomDocumentVersion.objects.filter(pk=version.id).update(is_searchable=True)
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            active_searchable_version=version,
            status=DataRoomDocument.Status.READY,
            token_count=version.token_count,
            parser_type=version.parser_type,
            chunking_strategy=version.chunking_strategy,
            embedding_model=version.embedding_model,
            processed_at=version.processed_at,
            updated_at=timezone.now(),
        )
    # pgvector flips after the authoritative app-DB pointer commits.
    if old_active_id and old_active_id != version.id:
        set_searchable_for_version(old_active_id, False)
    set_searchable_for_version(version.id, True)


def recompute_document_sensitivity(document_id: int) -> None:
    """Recompute document-level quarantine rollups as the union over retained versions.

    Editing out (or pruning) a sensitive version only clears the document flag once
    *no* retained version carries it — so a v0 that contained Article-9 data keeps
    the document flagged while v0 is retained.
    """
    from documents.models import DataRoomDocument, DataRoomDocumentVersion

    versions = DataRoomDocumentVersion.objects.filter(document_id=document_id)
    is_q = versions.filter(is_quarantined=True).exists()
    is_pq = is_q or versions.filter(is_partially_quarantined=True).exists()
    reason = ""
    if is_q:
        v = (
            versions.filter(is_quarantined=True)
            .exclude(quarantine_reason="")
            .order_by("version_index")
            .first()
        )
        reason = v.quarantine_reason if v else ""
    DataRoomDocument.objects.filter(pk=document_id).update(
        is_quarantined=is_q,
        is_partially_quarantined=is_pq,
        quarantine_reason=reason,
    )


def _joined_chunk_text(version) -> str:
    parts = list(version.chunks.order_by("chunk_index").values_list("text", flat=True))
    return "\n\n".join(p for p in parts if p)


def is_agent_remediable(version) -> bool:
    """Whether the assistant may open/edit ``version`` while it is quarantined.

    Quarantine controls disclosure of content the system did NOT author. Only content
    the system itself produced — canvas exports and agent edits/rewrites — is remediable
    in loop while quarantined. Uploaded / user-edited content stays locked: a quarantined
    upload is never surfaced to the assistant, even by index. A non-quarantined version
    is always accessible (normal rules apply).
    """
    from documents.models import DataRoomDocumentVersion as V

    if version is None or not version.is_quarantined:
        return True
    return version.origin in (V.Origin.AGENT_CREATED, V.Origin.CANVAS_EXPORT)


def open_working_version(document):
    """Return ``(content, version, warning)`` for editing the working head.

    Reads the version's markdown directly (NOT via gated retrieval), so it works
    even when the working version is quarantined — the originating agent can keep
    remediating. Legacy v0s (empty content, backfilled) fall back to joined chunks.

    Callers that let the agent edit must first gate on :func:`is_agent_remediable`
    (see ``chat.tools._agent_edit_block_reason``) so a quarantined *upload* stays locked.
    """
    version = document.current_version or document.active_searchable_version
    if version is None:
        version = document.versions.order_by("-version_index").first()
    if version is None:
        return "", None, ""

    content = version.content or ""
    if not content.strip():
        content = _joined_chunk_text(version)

    warning = ""
    if version.is_quarantined:
        warning = (
            f"This version is quarantined ({version.quarantine_reason or 'flagged content'}) "
            "and is not searchable. Edit it to remove the flagged content, then save again."
        )
    return content, version, warning


def document_status(document) -> dict:
    """Report processing/searchability state of a document for the agent.

    The async quarantine/processing verdict reaches the agent here, since save
    returns before the pipeline finishes.
    """
    from documents.models import DataRoomDocument

    cur = document.current_version
    act = document.active_searchable_version
    Status = DataRoomDocument.Status

    if cur is None:
        return {"state": "unknown", "reason": "No working version."}

    if cur.is_quarantined:
        state = "quarantined"
        reason = cur.quarantine_reason or "Flagged by content/PII guardrails."
    elif cur.status in (Status.UPLOADED, Status.PROCESSING, Status.SCANNING):
        state = "processing"
        reason = "Chunking, embedding and scanning in progress."
    elif cur.status in (Status.FAILED, Status.SCAN_FAILED):
        state = "failed"
        reason = cur.processing_error or "Processing failed."
    else:  # READY
        state = "ready"
        reason = ""

    note = ""
    if act and cur.id != act.id:
        note = (
            f"The live searchable document is still v{act.version_index}; "
            f"v{cur.version_index} is not retrievable yet."
        )

    return {
        "state": state,
        "current_version": cur.version_index,
        "active_version": act.version_index if act else None,
        "reason": reason,
        "note": note,
    }
