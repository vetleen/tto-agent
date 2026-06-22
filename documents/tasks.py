"""Celery tasks for the documents app."""

from __future__ import annotations

import logging

from celery import shared_task

from documents.services.process_document import process_document, process_document_version

logger = logging.getLogger(__name__)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
    time_limit=600,
    soft_time_limit=540,
)
def process_document_task(document_id: int) -> None:
    # Back-compat entry (upload path + stale sweeper): ensures v0 then processes it.
    process_document(document_id)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
    time_limit=600,
    soft_time_limit=540,
)
def process_document_version_task(version_id: int) -> None:
    process_document_version(version_id)


# Sweeper staleness thresholds (mirror process_document.STALE_PROCESSING_MINUTES
# for UPLOADED/PROCESSING; SCANNING gets a wider one spanning the guardrail scan
# plus finalize).
STALE_UPLOADED_MINUTES = 15
STALE_SCANNING_MINUTES = 60
MAX_REQUEUES = 3


@shared_task(time_limit=60)
def requeue_stale_documents() -> int:
    """Periodic recovery of *versions* stranded by a worker restart.

    With versioning the processing unit is a ``DataRoomDocumentVersion``, so the
    sweeper operates on versions: UPLOADED/PROCESSING past the stale window are
    re-enqueued (capped at MAX_REQUEUES via the version's ``requeue_count``);
    SCANNING past its window fails closed to SCAN_FAILED. Document-level status is
    mirrored for fresh uploads (no active version yet) so the UI isn't stuck.

    Transient DB unavailability is logged at INFO and skipped — the next tick retries.
    """
    from datetime import timedelta

    from django.db.models import F, Q
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from documents.models import DataRoomDocument, DataRoomDocumentVersion
    from documents.services.process_document import STALE_PROCESSING_MINUTES
    from documents.services.pii_scan import SCAN_FAILED_MESSAGE

    Status = DataRoomDocument.Status
    now = timezone.now()
    handled = 0

    try:
        stale_pipeline = Q(
            status=Status.UPLOADED,
            updated_at__lt=now - timedelta(minutes=STALE_UPLOADED_MINUTES),
        ) | Q(
            status=Status.PROCESSING,
            updated_at__lt=now - timedelta(minutes=STALE_PROCESSING_MINUTES),
        )

        # Requeue cap reached → permanent FAILED.
        exhausted_ids = list(
            DataRoomDocumentVersion.objects.filter(
                stale_pipeline, requeue_count__gte=MAX_REQUEUES,
            ).values_list("pk", flat=True)
        )
        if exhausted_ids:
            DataRoomDocumentVersion.objects.filter(pk__in=exhausted_ids).update(
                status=Status.FAILED,
                processing_error="Processing was interrupted repeatedly and has been stopped.",
                updated_at=now,
            )
            _mirror_terminal_doc_status(exhausted_ids, Status.FAILED, now)
            logger.warning(
                "requeue_stale_documents: %s version(s) exceeded %s requeues, marked FAILED",
                len(exhausted_ids), MAX_REQUEUES,
            )
            handled += len(exhausted_ids)

        requeue_ids = list(
            DataRoomDocumentVersion.objects.filter(
                stale_pipeline, requeue_count__lt=MAX_REQUEUES,
            ).values_list("pk", flat=True)
        )
        for version_id in requeue_ids:
            updated = DataRoomDocumentVersion.objects.filter(
                stale_pipeline, pk=version_id, requeue_count__lt=MAX_REQUEUES,
            ).update(requeue_count=F("requeue_count") + 1)
            if updated:
                logger.warning(
                    "requeue_stale_documents: version_id=%s stale, re-enqueueing", version_id,
                )
                process_document_version_task.delay(version_id)
                handled += 1

        # Stuck SCANNING → fail closed.
        scan_cutoff = now - timedelta(minutes=STALE_SCANNING_MINUTES)
        stuck_ids = list(
            DataRoomDocumentVersion.objects.filter(status=Status.SCANNING).filter(
                Q(processed_at__lt=scan_cutoff)
                | Q(processed_at__isnull=True, updated_at__lt=scan_cutoff)
            ).values_list("pk", flat=True)
        )
        if stuck_ids:
            DataRoomDocumentVersion.objects.filter(pk__in=stuck_ids).update(
                status=Status.SCAN_FAILED,
                processing_error=SCAN_FAILED_MESSAGE,
                updated_at=now,
            )
            _mirror_terminal_doc_status(stuck_ids, Status.SCAN_FAILED, now, error=SCAN_FAILED_MESSAGE)
            logger.warning(
                "requeue_stale_documents: %s version(s) stuck in SCANNING marked SCAN_FAILED",
                len(stuck_ids),
            )
            handled += len(stuck_ids)
    except (OperationalError, InterfaceError):
        logger.info(
            "Skipping stale document sweep: database temporarily unavailable; "
            "will retry on next beat tick.",
            exc_info=True,
        )
        return 0

    return handled


def _mirror_terminal_doc_status(version_ids, status, now, error=None):
    """Mirror a terminal version status onto documents that have no live version yet.

    Only fresh uploads (active_searchable_version is None) whose current_version is
    one of these versions are mirrored — an edit that fails leaves the previously
    live version untouched, so the document is not "stuck".
    """
    from documents.models import DataRoomDocument

    fields = {"status": status, "updated_at": now}
    if error is not None:
        fields["processing_error"] = error
    DataRoomDocument.objects.filter(
        active_searchable_version__isnull=True,
        current_version_id__in=version_ids,
    ).update(**fields)


@shared_task(bind=True, max_retries=3, time_limit=600, soft_time_limit=540)
def finalize_document_metadata(self, version_id: int) -> None:
    """Generate description + tags, run the PII scan, and release a held version.

    Dispatched by ``scan_document_version`` once the guardrail chunk scan completes.
    Operates on a single version: tags (description + PII) attach to the version,
    quarantine sets the version's flag, and the document-level sensitivity union is
    recomputed. This task is the SOLE releaser of SCANNING and the only place
    ``active_searchable_version`` advances:

    - On a clean scan the version is released READY and becomes the active searchable
      version (pgvector ``is_searchable`` flipped on — no re-embed).
    - A quarantined version is released READY but NOT made searchable; the previously
      active version stays live, so the bad draft is invisible to every other thread.
    """
    from celery.exceptions import MaxRetriesExceededError
    from django.utils import timezone

    from core.preferences import resolve_org_feature_model
    from documents.models import (
        DataRoomDocument,
        DataRoomDocumentTag,
        DataRoomDocumentVersion,
    )
    from documents.services.chunk_access import build_head_tail_text
    from documents.services.description import generate_description_and_tags_from_text
    from documents.services.pii_scan import (
        SCAN_FAILED_MESSAGE,
        org_id_for_document,
        resolve_pii_gate,
        scan_pii_categories_for_version,
    )
    from documents.services.versioning import advance_active_to, recompute_document_sensitivity
    from llm.service.errors import LLMAuthError, LLMConfigurationError, LLMPolicyDenied

    Status = DataRoomDocument.Status

    try:
        version = DataRoomDocumentVersion.objects.select_related("document").get(pk=version_id)
    except DataRoomDocumentVersion.DoesNotExist:
        logger.info("finalize_document_metadata: version_id=%s not found (deleted before finalize)", version_id)
        return

    doc = version.document
    document_id = doc.id
    held = version.status == Status.SCANNING

    def _finish_release():
        """Release the held version: READY, and advance the active pointer if clean."""
        if not held:
            return
        DataRoomDocumentVersion.objects.filter(pk=version_id, status=Status.SCANNING).update(
            status=Status.READY, updated_at=timezone.now()
        )
        is_q = DataRoomDocumentVersion.objects.filter(pk=version_id).values_list(
            "is_quarantined", flat=True
        ).first()
        if not is_q:
            advance_active_to(document_id, version)
        else:
            # Quarantined: do not advance active. For a fresh upload (no prior active
            # version) mark the document READY so the UI shows processing finished;
            # retrieval still surfaces nothing because active stays None.
            if doc.active_searchable_version_id is None:
                DataRoomDocument.objects.filter(pk=document_id).update(
                    status=Status.READY, updated_at=timezone.now()
                )
            recompute_document_sensitivity(document_id)

    def _mark_scan_failed():
        updated = DataRoomDocumentVersion.objects.filter(
            pk=version_id, status=Status.SCANNING,
        ).update(status=Status.SCAN_FAILED, processing_error=SCAN_FAILED_MESSAGE, updated_at=timezone.now())
        if updated and doc.active_searchable_version_id in (None, version_id):
            DataRoomDocument.objects.filter(pk=document_id).update(
                status=Status.SCAN_FAILED, processing_error=SCAN_FAILED_MESSAGE, updated_at=timezone.now()
            )

    org_id = org_id_for_document(doc)
    desc_model = resolve_org_feature_model(org_id, "document_description")
    pii_model, pii_enabled, pii_quarantine_enabled = resolve_pii_gate(org_id)
    pii_must_gate = bool(pii_model and pii_enabled and pii_quarantine_enabled)

    if not desc_model and not (pii_enabled and pii_model):
        _finish_release()
        return

    # --- Description + tags (only generated once, on the first version) ---
    if desc_model and not doc.description:
        try:
            text = build_head_tail_text(version_id)
            if text.strip():
                result = generate_description_and_tags_from_text(
                    text, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id, org_id=org_id,
                    is_image=(getattr(version, "parser_type", "") == "image"),
                )
                doc.description = result["description"]
                update_fields = ["description", "updated_at"]
                if result.get("document_date"):
                    doc.document_date = result["document_date"]
                    update_fields.append("document_date")
                doc.save(update_fields=update_fields)
                for tag_key, tag_value in result.get("tags", {}).items():
                    DataRoomDocumentTag.objects.update_or_create(
                        version=version, key=tag_key, defaults={"value": tag_value},
                    )
        except DataRoomDocument.NotUpdated:
            logger.info(
                "finalize_document_metadata: version_id=%s document deleted during description generation, skipping",
                version_id,
            )
        except (LLMPolicyDenied, LLMConfigurationError, LLMAuthError):
            logger.exception(
                "finalize_document_metadata: version_id=%s description blocked by LLM config/policy",
                version_id,
            )
        except Exception:
            logger.exception(
                "finalize_document_metadata: version_id=%s description generation failed (non-critical)",
                version_id,
            )

    # --- PII category scan (entire version, windowed) ---
    pii_result: dict[str, bool] = {}
    if pii_enabled and pii_model:
        try:
            pii_result = scan_pii_categories_for_version(
                version_id, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id, org_id=org_id,
            )
            for category in pii_result:
                DataRoomDocumentTag.objects.update_or_create(
                    version=version, key=category, defaults={"value": "true"},
                )
        except (LLMPolicyDenied, LLMConfigurationError, LLMAuthError):
            logger.exception(
                "finalize_document_metadata: version_id=%s PII scan blocked by LLM config/policy", version_id,
            )
            if held and pii_must_gate:
                _mark_scan_failed()
            elif held:
                _finish_release()
            return
        except Exception:
            if not (held and pii_must_gate):
                logger.exception(
                    "finalize_document_metadata: version_id=%s PII scan failed (non-critical)", version_id,
                )
                if held:
                    _finish_release()
                return
            logger.exception(
                "finalize_document_metadata: version_id=%s PII scan failed (attempt %s/%s)",
                version_id, self.request.retries + 1, self.max_retries + 1,
            )
            try:
                raise self.retry(countdown=30 * (2 ** self.request.retries))
            except MaxRetriesExceededError:
                logger.warning(
                    "finalize_document_metadata: version_id=%s scan retries exhausted; marking scan_failed",
                    version_id,
                )
                _mark_scan_failed()
                return

    # --- Quarantine on GDPR Article 9 / 10 detection (version-scoped) ---
    if pii_quarantine_enabled:
        articles = []
        if pii_result.get("pii_special_category"):
            articles.append("Article 9 (special category)")
        if pii_result.get("pii_criminal_offence"):
            articles.append("Article 10 (criminal offence)")
        if articles:
            DataRoomDocumentVersion.objects.filter(pk=version_id).update(
                is_quarantined=True,
                quarantine_reason="Contains GDPR " + " and ".join(articles) + " personal data.",
            )
            recompute_document_sensitivity(document_id)
            logger.warning(
                "finalize_document_metadata: version_id=%s quarantined (%s)",
                version_id, ", ".join(articles),
            )

    _finish_release()


# --- Version pruning -------------------------------------------------------

# Keep at most this many versions per document; beyond it the oldest non-protected
# versions are dropped. Always-protected: v0 (original), current, and active.
MAX_VERSIONS_PER_DOCUMENT = 10


@shared_task(time_limit=300)
def prune_document_versions() -> int:
    """Nightly prune of old document versions.

    Per document, keep: v0 (the original), current_version, active_searchable_version,
    and — among the rest — at most one per calendar day (the newest of that day). If
    more than MAX_VERSIONS_PER_DOCUMENT remain, drop the oldest non-protected until the
    cap is met. Dropped versions delete their chunks/tags (CASCADE), their pgvector
    rows, and their native blob; the document-level sensitivity union is recomputed.
    """
    from django.db.utils import InterfaceError, OperationalError

    from documents.models import DataRoomDocument, DataRoomDocumentVersion
    from documents.services.vector_store import delete_vectors_for_version
    from documents.services.versioning import recompute_document_sensitivity

    pruned = 0
    try:
        docs = list(
            DataRoomDocument.objects.all()
            .only("pk", "current_version_id", "active_searchable_version_id")
            .iterator(chunk_size=200)
        )
    except (OperationalError, InterfaceError):
        logger.info("prune_document_versions: DB unavailable; will retry next tick.", exc_info=True)
        return 0

    # Fetch versions a batch of documents at a time — one grouped query per batch
    # rather than one query per document (the latter was an N+1, WILFRED-64). Each
    # batch's versions are dropped from memory before the next, preserving the
    # streaming intent of the document iterator above.
    BATCH_SIZE = 200
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        versions_by_doc: dict[int, list[dict]] = {}
        for v in (
            DataRoomDocumentVersion.objects.filter(document_id__in=[d.pk for d in batch])
            .order_by("-version_index")
            .values("pk", "document_id", "version_index", "created_at")
        ):
            versions_by_doc.setdefault(v["document_id"], []).append(v)

        for doc in batch:
            versions = versions_by_doc.get(doc.pk, [])
            if len(versions) <= 1:
                continue

            protected: set[int] = set()
            # v0 (lowest version_index)
            v0 = min(versions, key=lambda v: v["version_index"])
            protected.add(v0["pk"])
            if doc.current_version_id:
                protected.add(doc.current_version_id)
            if doc.active_searchable_version_id:
                protected.add(doc.active_searchable_version_id)

            # Among the non-protected, keep the newest one per calendar day.
            kept_days: set = set()
            droppable: list[int] = []
            for v in versions:  # newest first
                if v["pk"] in protected:
                    continue
                day = v["created_at"].date() if v["created_at"] else None
                if day is not None and day not in kept_days:
                    kept_days.add(day)
                    continue  # keep the newest of this day
                droppable.append(v["pk"])

            # Enforce the hard cap: total kept must be <= MAX. Drop more (oldest first)
            # from the day-survivors if needed, never the always-protected set.
            total_kept = len(versions) - len(droppable)
            if total_kept > MAX_VERSIONS_PER_DOCUMENT:
                day_survivors = [
                    v["pk"] for v in sorted(versions, key=lambda v: v["version_index"])
                    if v["pk"] not in protected and v["pk"] not in droppable
                ]
                overflow = total_kept - MAX_VERSIONS_PER_DOCUMENT
                droppable.extend(day_survivors[:overflow])

            for vid in droppable:
                try:
                    delete_vectors_for_version(vid)
                    DataRoomDocumentVersion.objects.filter(pk=vid).delete()
                    pruned += 1
                except Exception:
                    logger.exception("prune_document_versions: failed to drop version_id=%s", vid)

            if droppable:
                recompute_document_sensitivity(doc.pk)

    if pruned:
        logger.info("prune_document_versions: dropped %s version(s)", pruned)
    return pruned
