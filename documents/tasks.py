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


@shared_task(bind=True, max_retries=3, time_limit=600, soft_time_limit=540)
def finalize_document_metadata(self, document_id: int) -> None:
    """Generate the document description + tags and run the full-document PII scan.

    Dispatched (fire-and-forget) by ``process_document`` after processing, so the
    heavy ``process_document`` frame can return and free every copy of the document
    text before any LLM work begins. Reads the text back from the persisted chunks —
    head/tail for the description, the full document in windows for PII — so the
    worker never holds the whole document in memory here.

    When the org's PII quarantine gate is active the document arrives here in
    SCANNING — held out of retrieval — and this task is what releases it: READY on
    a completed scan (quarantine flags applied as needed), SCAN_FAILED when the scan
    can't complete so the user can retry from the document list. Transient scan
    failures retry (max 3 with backoff); config/policy errors won't self-heal and
    fail immediately. Description generation stays best-effort.

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

    # Gated documents are held in SCANNING by process_document (or document_rescan)
    # and must leave this task as READY or SCAN_FAILED — never stay stuck.
    gated = doc.status == DataRoomDocument.Status.SCANNING

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

    # Nothing to do — skip the chunk reads entirely. A gated document is still
    # released (the org may have disabled the scan since processing started).
    if not desc_model and not (pii_enabled and pii_model):
        if gated:
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
            if gated:
                _mark_scan_failed()
            return
        except Exception:
            if not gated:
                # Informational scan only — log and move on, as before.
                logger.exception(
                    "finalize_document_metadata: document_id=%s PII scan failed (non-critical)",
                    document_id,
                )
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
    # document to retrieval.
    if gated:
        _release_to_ready()
