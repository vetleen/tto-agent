"""Celery tasks for the documents app."""

from __future__ import annotations

import logging

from celery import shared_task

from documents.services.process_document import process_document

logger = logging.getLogger(__name__)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
    time_limit=600,
    soft_time_limit=540,
)
def process_document_task(document_id: int) -> None:
    process_document(document_id)


# Sweeper staleness thresholds. UPLOADED uses the same window as the
# PROCESSING stale guard in documents/services/process_document.py; SCANNING
# gets a wider one because it spans the guardrail chunk scan plus
# finalize_document_metadata, each with a 600s limit and retries with backoff.
STALE_UPLOADED_MINUTES = 15
STALE_SCANNING_MINUTES = 60
MAX_REQUEUES = 3


@shared_task(time_limit=60)
def requeue_stale_documents() -> int:
    """Periodic recovery of documents stranded by a worker restart.

    Celery acks tasks early, so a dyno restart (Heroku cycles dynos daily)
    silently drops any in-flight task: documents stay UPLOADED (enqueue lost),
    PROCESSING (worker died mid-pipeline), or SCANNING (guardrail scan or
    finalize lost) forever.

    - UPLOADED/PROCESSING past the stale window are re-enqueued — safe because
      process_document uses select_for_update(skip_locked=True) plus its own
      stale-PROCESSING guard. ``requeue_count`` caps this at MAX_REQUEUES so a
      poison document (e.g. one that OOMs the worker) becomes FAILED instead
      of crash-looping forever.
    - SCANNING past its window fails closed to SCAN_FAILED (the document never
      reached retrieval), matching guardrails' own failure path; the user
      recovers via the existing rescan button.

    Like chat.tasks.expire_stale_subagent_runs, transient DB unavailability is
    logged at INFO and skipped — the next beat tick retries.
    """
    from datetime import timedelta

    from django.db.models import F, Q
    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from documents.models import DataRoomDocument
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

        # Requeue cap reached → permanent FAILED so the user sees a state
        # they can act on (re-upload) instead of an eternal spinner.
        exhausted = DataRoomDocument.objects.filter(
            stale_pipeline, requeue_count__gte=MAX_REQUEUES,
        ).update(
            status=Status.FAILED,
            processing_error=(
                "Processing was interrupted repeatedly and has been stopped."
            ),
            updated_at=now,
        )
        if exhausted:
            logger.warning(
                "requeue_stale_documents: %s document(s) exceeded %s requeues, marked FAILED",
                exhausted, MAX_REQUEUES,
            )
            handled += exhausted

        requeue_ids = list(
            DataRoomDocument.objects.filter(
                stale_pipeline, requeue_count__lt=MAX_REQUEUES,
            ).values_list("pk", flat=True)
        )
        for doc_id in requeue_ids:
            # Conditional per-row update: a no-op if the document moved on
            # (or was requeued by a concurrent tick) since the list query.
            # Deliberately leaves updated_at stale so process_document's
            # stale-PROCESSING guard force-reprocesses instead of skipping.
            updated = DataRoomDocument.objects.filter(
                stale_pipeline, pk=doc_id, requeue_count__lt=MAX_REQUEUES,
            ).update(requeue_count=F("requeue_count") + 1)
            if updated:
                logger.warning(
                    "requeue_stale_documents: document_id=%s stale, re-enqueueing", doc_id,
                )
                process_document_task.delay(doc_id)
                handled += 1

        # Stuck SCANNING → fail closed (same terminal state guardrails uses
        # when the chunk scan exhausts retries). Staleness is measured from
        # processed_at (when processing handed off to the scan); updated_at
        # is only a fallback — description generation refreshes it mid-scan.
        scan_cutoff = now - timedelta(minutes=STALE_SCANNING_MINUTES)
        stuck_scans = DataRoomDocument.objects.filter(status=Status.SCANNING).filter(
            Q(processed_at__lt=scan_cutoff)
            | Q(processed_at__isnull=True, updated_at__lt=scan_cutoff)
        ).update(
            status=Status.SCAN_FAILED,
            processing_error=SCAN_FAILED_MESSAGE,
            updated_at=now,
        )
        if stuck_scans:
            logger.warning(
                "requeue_stale_documents: %s document(s) stuck in SCANNING marked SCAN_FAILED",
                stuck_scans,
            )
            handled += stuck_scans
    except (OperationalError, InterfaceError):
        logger.info(
            "Skipping stale document sweep: database temporarily unavailable; "
            "will retry on next beat tick.",
            exc_info=True,
        )
        return 0

    return handled


@shared_task(bind=True, max_retries=3, time_limit=600, soft_time_limit=540)
def finalize_document_metadata(self, document_id: int) -> None:
    """Generate the document description + tags and run the full-document PII scan.

    Dispatched (fire-and-forget) by ``scan_document_chunks`` once the guardrail chunk
    scan completes, so the heavy ``process_document`` frame can return and free every
    copy of the document text before any LLM work begins. Reads the text back from the
    persisted chunks — head/tail for the description, the full document in windows for
    PII — so the worker never holds the whole document in memory here.

    Documents arrive here held in SCANNING (the guardrail chunk scan hands off to
    this task after it completes) — held out of retrieval — and this task is the sole
    releaser: READY on a completed scan (quarantine flags applied as needed),
    SCAN_FAILED when the *PII quarantine gate* is active and its scan can't complete
    so the user can retry from the document list. When PII scanning is informational
    only (or absent), a scan failure still releases the document rather than stranding
    it. Transient gated-scan failures retry (max 3 with backoff); config/policy errors
    won't self-heal and fail immediately. Description generation stays best-effort.

    Imports are kept inside the function (like ``guardrails/tasks.py``) so importing
    this module at Celery autodiscover stays cheap.
    """
    from celery.exceptions import MaxRetriesExceededError
    from django.utils import timezone

    from core.preferences import resolve_org_feature_model
    from documents.models import DataRoomDocument, DataRoomDocumentTag
    from documents.services.chunk_access import build_head_tail_text
    from documents.services.description import generate_description_and_tags_from_text
    from documents.services.pii_scan import (
        SCAN_FAILED_MESSAGE,
        org_id_for_document,
        resolve_pii_gate,
        scan_pii_categories_for_document,
    )
    from llm.service.errors import LLMAuthError, LLMConfigurationError, LLMPolicyDenied

    try:
        doc = DataRoomDocument.objects.get(pk=document_id)
    except DataRoomDocument.DoesNotExist:
        # Document was deleted between processing and this task — expected, not an error.
        logger.info("finalize_document_metadata: document_id=%s not found (deleted before finalize)", document_id)
        return

    # Every document now arrives held in SCANNING (the guardrail chunk scan hands off
    # here after it completes) and must leave as READY or SCAN_FAILED — never stuck.
    # ``held`` is the release responsibility; whether a *PII* scan failure must fail
    # the doc closed is a separate question answered by ``pii_must_gate`` below.
    held = doc.status == DataRoomDocument.Status.SCANNING

    def _release_to_ready():
        # Conditional update: a no-op if the document was deleted or changed since.
        DataRoomDocument.objects.filter(
            pk=doc.pk, status=DataRoomDocument.Status.SCANNING,
        ).update(status=DataRoomDocument.Status.READY, updated_at=timezone.now())

    def _mark_scan_failed():
        DataRoomDocument.objects.filter(
            pk=doc.pk, status=DataRoomDocument.Status.SCANNING,
        ).update(
            status=DataRoomDocument.Status.SCAN_FAILED,
            processing_error=SCAN_FAILED_MESSAGE,
            updated_at=timezone.now(),
        )

    org_id = org_id_for_document(doc)
    desc_model = resolve_org_feature_model(org_id, "document_description")
    pii_model, pii_enabled, pii_quarantine_enabled = resolve_pii_gate(org_id)
    # The PII *quarantine* gate (GDPR Art. 9/10): only when this is active must a PII
    # scan failure fail the held document closed. An informational-only scan that
    # fails must still release the document, not strand it in SCANNING.
    pii_must_gate = bool(pii_model and pii_enabled and pii_quarantine_enabled)

    # Nothing to do — skip the chunk reads entirely. A held document is still
    # released (the org may have disabled the scan since processing started).
    if not desc_model and not (pii_enabled and pii_model):
        if held:
            _release_to_ready()
        return

    # --- Description + tags (head/tail text is all the relevance gist needs) ---
    # Skipped when a description already exists so scan retries don't burn LLM calls.
    if desc_model and not doc.description:
        try:
            text = build_head_tail_text(document_id)
            if text.strip():
                result = generate_description_and_tags_from_text(
                    text, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id, org_id=org_id,
                )
                doc.description = result["description"]
                update_fields = ["description", "updated_at"]
                if result.get("document_date"):
                    doc.document_date = result["document_date"]
                    update_fields.append("document_date")
                doc.save(update_fields=update_fields)
                for tag_key, tag_value in result.get("tags", {}).items():
                    DataRoomDocumentTag.objects.update_or_create(
                        document=doc, key=tag_key, defaults={"value": tag_value},
                    )
        except DataRoomDocument.NotUpdated:
            # Document was deleted during description generation — expected, not an error.
            logger.info(
                "finalize_document_metadata: document_id=%s deleted during description generation, skipping",
                document_id,
            )
        except (LLMPolicyDenied, LLMConfigurationError, LLMAuthError):
            # Config/policy errors won't self-heal — surface as a distinct Sentry issue.
            logger.exception(
                "finalize_document_metadata: document_id=%s description generation blocked by LLM config/policy",
                document_id,
            )
        except Exception:
            logger.exception(
                "finalize_document_metadata: document_id=%s description generation failed (non-critical)",
                document_id,
            )

    # --- PII category scan (entire document, windowed) ---
    pii_result: dict[str, bool] = {}
    if pii_enabled and pii_model:
        try:
            pii_result = scan_pii_categories_for_document(
                document_id, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id, org_id=org_id,
            )
            # NOTE: detections are union-only — categories are added here but never
            # cleared, and is_quarantined is never un-set by a later scan. That's the
            # right bias while document content is immutable, but if a rescan-after-edit
            # feature is ever added (e.g. the user edits and resaves a document), it must
            # first clear existing pii_* tags and re-evaluate quarantine from scratch,
            # or stale detections from the old content will stick forever.
            for category in pii_result:
                DataRoomDocumentTag.objects.update_or_create(
                    document=doc, key=category, defaults={"value": "true"},
                )
        except (LLMPolicyDenied, LLMConfigurationError, LLMAuthError):
            # Config/policy errors won't self-heal — retrying won't help.
            logger.exception(
                "finalize_document_metadata: document_id=%s PII scan blocked by LLM config/policy",
                document_id,
            )
            if held and pii_must_gate:
                _mark_scan_failed()
            elif held:
                # Informational scan — don't strand the held document.
                _release_to_ready()
            return
        except Exception:
            if not (held and pii_must_gate):
                # Best-effort: either we don't own this doc's release (not held) or
                # the PII scan is informational. Log, and release if we're holding it.
                logger.exception(
                    "finalize_document_metadata: document_id=%s PII scan failed (non-critical)",
                    document_id,
                )
                if held:
                    _release_to_ready()
                return
            logger.exception(
                "finalize_document_metadata: document_id=%s PII scan failed (attempt %s/%s)",
                document_id, self.request.retries + 1, self.max_retries + 1,
            )
            try:
                # Raises Retry to hand the task back to the worker; raises
                # MaxRetriesExceededError once attempts are exhausted.
                raise self.retry(countdown=30 * (2 ** self.request.retries))
            except MaxRetriesExceededError:
                logger.warning(
                    "finalize_document_metadata: document_id=%s scan retries exhausted; marking scan_failed",
                    document_id,
                )
                _mark_scan_failed()
                return

    # --- Quarantine on GDPR Article 9 / 10 detection ---
    # Special category (Art. 9) and criminal offence (Art. 10) data must never reach
    # the LLM. Flag the document; read-time filters in documents/services/retrieval.py
    # keep its chunks out of every retrieval path.
    if pii_quarantine_enabled:
        articles = []
        if pii_result.get("pii_special_category"):
            articles.append("Article 9 (special category)")
        if pii_result.get("pii_criminal_offence"):
            articles.append("Article 10 (criminal offence)")
        if articles:
            try:
                doc.is_quarantined = True
                doc.quarantine_reason = "Contains GDPR " + " and ".join(articles) + " personal data."
                doc.save(update_fields=["is_quarantined", "quarantine_reason", "updated_at"])
                logger.warning(
                    "finalize_document_metadata: document_id=%s quarantined (%s)",
                    document_id,
                    ", ".join(articles),
                )
            except DataRoomDocument.NotUpdated:
                # Document was deleted before quarantine could be applied — expected, not an error.
                logger.info(
                    "finalize_document_metadata: document_id=%s deleted before quarantine, skipping",
                    document_id,
                )
                return

    # Scan complete (quarantine flags applied where needed) — release the
    # held document to retrieval.
    if held:
        _release_to_ready()
