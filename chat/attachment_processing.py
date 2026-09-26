"""Worker-side processing of chat attachments.

Every pdf/docx/pptx attachment is processed once, right after upload (chat ``+``
menu) or copy (meeting → "minutes with Wilfred" thread), by the Celery task
``chat.tasks.process_chat_attachment``:

* text extraction with embedded pictures persisted as **attachment-owned** Assets
  (``get_or_extract_attachment_text``), cached on ``extracted_content``;
* the PDF page count / pptx slide count on ``page_count``;
* for pptx, per-slide JPEG renders through the Gotenberg render service
  (``chat.attachment_render.render_attachment_pages``), stored as attachment-owned
  ``role=page_render`` Assets.

``ChatAttachment.processing_state`` is the gate the chat consumer waits on before
starting a turn (``_wait_for_attachments``): PENDING while the task runs, READY
when the model can use the file (renders done, skipped or failed), FAILED when
extraction itself failed or the task could not be queued. Images and text files
need nothing and are READY from the start. The turn-time path keeps today's lazy
extraction as a fallback, so a lost task never blocks a conversation.

Rows are only ever touched with ``.update()`` / ``update_fields`` here: the
consumer links the attachment to its message concurrently (``_link_attachments``
does ``.update(message=…)``) and a full ``save()`` would race it.
"""
from __future__ import annotations

import logging

from django.db import transaction

logger = logging.getLogger(__name__)


def needs_processing(content_type: str) -> bool:
    """Whether uploads of this type go through the worker task (pdf/docx/pptx)."""
    from chat.services import SUPPORTED_DOCX_TYPES, SUPPORTED_PDF_TYPES, SUPPORTED_PPTX_TYPES

    ct = content_type or ""
    return ct in SUPPORTED_PDF_TYPES or ct in SUPPORTED_DOCX_TYPES or ct in SUPPORTED_PPTX_TYPES


def initial_processing_state(content_type: str) -> str:
    """The ``processing_state`` a freshly created attachment row should carry."""
    from chat.models import ChatAttachment

    State = ChatAttachment.ProcessingState
    return State.PENDING if needs_processing(content_type) else State.READY


def dispatch_processing(attachment_id) -> bool:
    """Queue the processing task; on a publish failure mark the row FAILED so the
    consumer never waits for a task that will not come."""
    from chat.models import ChatAttachment
    from chat.tasks import process_chat_attachment

    try:
        process_chat_attachment.delay(str(attachment_id))
    except Exception:  # noqa: BLE001 — broker blip; the turn falls back to lazy extraction
        logger.warning(
            "attachment_processing: dispatch failed for attachment_id=%s", attachment_id, exc_info=True,
        )
        ChatAttachment.objects.filter(
            pk=attachment_id, processing_state=ChatAttachment.ProcessingState.PENDING,
        ).update(
            processing_state=ChatAttachment.ProcessingState.FAILED,
            processing_error="Could not queue processing.",
        )
        return False
    return True


def dispatch_after_commit(attachment) -> None:
    """Dispatch processing for a PENDING row once the surrounding transaction commits
    (or immediately when there is none)."""
    from chat.models import ChatAttachment

    if attachment.processing_state != ChatAttachment.ProcessingState.PENDING:
        return
    transaction.on_commit(lambda aid=attachment.id: dispatch_processing(aid))


def mark_for_reprocessing(attachment) -> None:
    """Flip a copied pptx attachment (reattach / thread branch) back to PENDING and
    queue processing: the extracted text was copied, the slide renders were not."""
    from chat.models import ChatAttachment

    ChatAttachment.objects.filter(pk=attachment.pk).update(
        processing_state=ChatAttachment.ProcessingState.PENDING,
        page_render_state=ChatAttachment.PageRenderState.NONE,
        processing_error="",
    )
    attachment.processing_state = ChatAttachment.ProcessingState.PENDING
    dispatch_after_commit(attachment)


def _fail(attachment_id, error: str) -> None:
    from chat.models import ChatAttachment

    ChatAttachment.objects.filter(pk=attachment_id).update(
        processing_state=ChatAttachment.ProcessingState.FAILED,
        processing_error=(error or "Processing failed.")[:2000],
    )


def process_attachment(attachment_id: str, *, first_delivery: bool = True) -> str:
    """Process one attachment; returns ``"ready"``, ``"failed"`` or ``"missing"``.

    Raises ``RenderBusy`` / ``RenderUnavailable`` from the pptx render step so the
    Celery task retries with backoff — the row stays PENDING and the retry resumes
    at the first missing slide (extraction is already cached by then).
    """
    from chat.models import ChatAttachment
    from chat.services import SUPPORTED_PDF_TYPES, SUPPORTED_PPTX_TYPES, get_or_extract_attachment_text

    State = ChatAttachment.ProcessingState
    Render = ChatAttachment.PageRenderState
    try:
        att = ChatAttachment.objects.select_related("uploaded_by").get(pk=attachment_id)
    except ChatAttachment.DoesNotExist:
        logger.info("attachment_processing: attachment_id=%s not found (deleted before processing)", attachment_id)
        return "missing"
    if att.processing_state == State.READY and att.page_render_state != Render.PENDING:
        return "ready"

    ct = att.content_type or ""
    if not needs_processing(ct):
        ChatAttachment.objects.filter(pk=att.pk).update(processing_state=State.READY, processing_error="")
        return "ready"
    is_pdf = ct in SUPPORTED_PDF_TYPES
    is_pptx = ct in SUPPORTED_PPTX_TYPES

    try:
        with att.file.open("rb") as fh:
            data = fh.read()
    except Exception:  # noqa: BLE001
        logger.warning("attachment_processing: attachment_id=%s could not read the file", att.id, exc_info=True)
        _fail(att.pk, "The uploaded file could not be read.")
        return "failed"

    try:
        get_or_extract_attachment_text(att, data, user=att.uploaded_by)
        if is_pdf:
            from chat.pdf_attach import pdf_page_count

            n_pages = pdf_page_count(data)
            if n_pages and att.page_count != n_pages:
                ChatAttachment.objects.filter(pk=att.pk).update(page_count=n_pages)
                att.page_count = n_pages
        elif is_pptx:
            from documents.services.page_render import count_slides

            n_slides = count_slides(data)
            if att.page_count != n_slides:
                ChatAttachment.objects.filter(pk=att.pk).update(page_count=n_slides)
                att.page_count = n_slides
    except Exception as exc:  # noqa: BLE001 — a bad file is not retryable
        logger.warning("attachment_processing: attachment_id=%s extraction failed", att.id, exc_info=True)
        _fail(att.pk, f"The file could not be read: {exc.__class__.__name__}")
        return "failed"
    del data

    if is_pptx:
        from chat.attachment_render import render_attachment_pages

        # Sets page_render_state itself (ready / partial / skipped / failed);
        # RenderBusy / RenderUnavailable propagate for the task's retry.
        render_state = render_attachment_pages(att, count_attempt=first_delivery)
        logger.info("attachment_processing: attachment_id=%s slide renders %s", att.id, render_state)

    # Only the state flips here: a render outcome may have left an explanatory
    # processing_error (skipped / failed) that the view tool surfaces.
    ChatAttachment.objects.filter(pk=att.pk).update(processing_state=State.READY)
    return "ready"


__all__ = [
    "dispatch_after_commit",
    "dispatch_processing",
    "initial_processing_state",
    "mark_for_reprocessing",
    "needs_processing",
    "process_attachment",
]
