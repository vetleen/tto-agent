"""Canvas file import, run on the worker.

``chat.views.canvas_import`` validates the upload, picks or creates the target
canvas, stores the file on a :class:`chat.models.CanvasImportJob` and queues
``chat.tasks.process_canvas_import``. Extraction (PDF parse, docx conversion,
one vision call per embedded picture) and the thread-title LLM call used to run
inside the request and hit Heroku's 30 s router timeout on big files while
blocking the web dyno's shared sync thread (staging, 2026-10-15).

The browser polls ``canvas_import_status`` (authoritative) and also receives a
best-effort ``canvas.updated`` push over the thread group (channel-layer pubsub
is at-most-once, so the push is only an accelerator / multi-tab update).
"""

from __future__ import annotations

import logging

from django.utils import timezone

logger = logging.getLogger(__name__)

EMPTY_IMPORT_ERROR = "That file has no extractable text (it may be a scanned PDF or an empty document)."
STALE_AFTER_MINUTES = 15


def dispatch_import(job_id) -> bool:
    """Queue the worker task; on a broker failure mark the job FAILED (and drop
    the canvas it created). Returns whether the task was queued."""
    from chat.tasks import process_canvas_import

    try:
        process_canvas_import.delay(str(job_id))
        return True
    except Exception:
        logger.exception("Could not queue canvas import %s", job_id)
        fail_job(job_id, "Could not queue the import. Try again.")
        return False


def fail_job(job_id, error: str) -> None:
    """Settle a job as FAILED: delete the canvas it created, drop the stored
    file, clear progress. Safe to call more than once."""
    from chat.models import CanvasImportJob
    from documents.services.progress import canvas_imports as progress

    job = CanvasImportJob.objects.filter(pk=job_id).select_related("canvas").first()
    if job is None:
        return
    if job.state in (CanvasImportJob.State.READY,):
        return
    if job.created_canvas and job.canvas_id:
        canvas = job.canvas
        # Only drop it if nobody filled it in the meantime.
        if canvas is not None and not (canvas.content or "").strip():
            canvas.delete()
    CanvasImportJob.objects.filter(pk=job.pk).update(
        state=CanvasImportJob.State.FAILED, error=error[:2000], finished_at=timezone.now(),
    )
    _delete_file(job)
    progress.clear(str(job.pk))


def _delete_file(job) -> None:
    if not job.file:
        return
    try:
        job.file.delete(save=False)
    except Exception:
        logger.warning("Could not delete stored canvas import file for job %s", job.pk, exc_info=True)
    from chat.models import CanvasImportJob

    CanvasImportJob.objects.filter(pk=job.pk).update(file="")


def _uploaded_from_job(job):
    """Rebuild an UploadedFile-like for the extractors from the stored file:
    they detect the kind from ``content_type`` / ``name``, so use the original
    filename, not the storage path."""
    from django.core.files.uploadedfile import SimpleUploadedFile

    with job.file.open("rb") as f:
        data = f.read()
    return SimpleUploadedFile(job.original_filename, data, content_type=job.content_type or "")


def canvas_updated_payload(canvas, *, job=None) -> dict:
    """The browser's ``canvas.updated`` event for *canvas* (same shape the
    consumer sends after an AI write), plus the import's truncation report."""
    from chat.services import canvas_diff_baseline

    accepted_content, pending_ai_review = canvas_diff_baseline(canvas)
    payload = {
        "event_type": "canvas.updated",
        "canvas_id": str(canvas.pk),
        "title": canvas.title,
        "content": canvas.content,
        "accepted_content": accepted_content,
        "pending_ai_review": pending_ai_review,
    }
    if job is not None:
        payload["import_job_id"] = str(job.pk)
        if job.original_chars:
            payload["truncated"] = True
            payload["original_chars"] = job.original_chars
    return payload


def _push(thread_id, event: dict) -> None:
    """Best-effort: forward *event* to every open page on the thread via the
    consumer's ``loop.event`` passthrough (``ChatConsumer.loop_event``)."""
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer

    from core.redis_errors import log_broadcast_failure

    try:
        async_to_sync(get_channel_layer().group_send)(
            f"thread_{thread_id}", {"type": "loop.event", "event": event},
        )
    except Exception as exc:
        log_broadcast_failure(logger, exc, "Could not push canvas import result for thread %s", thread_id)


def run_canvas_import(job_id) -> str:
    """Extract the stored file into the job's canvas. Returns the final state."""
    from chat.models import CanvasImportJob, ChatThread
    from chat.services import (
        create_canvas_checkpoint,
        generate_canvas_title,
        import_file_to_canvas,
        snapshot_user_edits,
    )
    from documents.services.progress import canvas_imports as progress

    claimed = CanvasImportJob.objects.filter(
        pk=job_id, state=CanvasImportJob.State.PENDING,
    ).update(state=CanvasImportJob.State.RUNNING)
    job = CanvasImportJob.objects.filter(pk=job_id).select_related("canvas", "thread", "created_by").first()
    if job is None:
        return "missing"
    if not claimed and job.state != CanvasImportJob.State.RUNNING:
        return job.state  # already settled (a duplicate delivery)
    canvas = job.canvas
    if canvas is None or canvas.deleted_at is not None:
        fail_job(job.pk, "The canvas was deleted before the import finished.")
        return CanvasImportJob.State.FAILED

    key = str(job.pk)
    progress.set_stage(key, "extracting")

    def _progress(stage, current=0, total=0):
        progress.set_stage(key, stage, current=current, total=total)

    uploaded = _uploaded_from_job(job)
    _title, content, original_chars = import_file_to_canvas(
        uploaded, job.created_by, canvas=canvas, progress=_progress,
    )
    if not (content or "").strip():
        fail_job(job.pk, EMPTY_IMPORT_ERROR)
        return CanvasImportJob.State.FAILED

    # The user may have typed into this canvas while the import ran: keep their
    # text in the version history before the import replaces it.
    canvas.refresh_from_db()
    snapshot_user_edits(canvas)
    title = (job.original_filename.rsplit(".", 1)[0] or "Untitled document")[:255]
    canvas.title = title
    canvas.content = content
    canvas.truncated_from_chars = original_chars or None
    canvas.source_document = None
    canvas.save(update_fields=["title", "content", "truncated_from_chars", "source_document", "updated_at"])
    cp = create_canvas_checkpoint(canvas, source="import", description="Imported from file")
    canvas.accepted_checkpoint = cp
    canvas.save(update_fields=["accepted_checkpoint"])

    thread_title = ""
    thread = ChatThread.objects.filter(pk=job.thread_id).only("title").first()
    if thread is not None and not thread.title:
        thread_title = generate_canvas_title(
            title, content, job.created_by, conversation_id=str(job.thread_id),
        ) or ""
        if thread_title:
            ChatThread.objects.filter(pk=job.thread_id, title="").update(title=thread_title)

    CanvasImportJob.objects.filter(pk=job.pk).update(
        state=CanvasImportJob.State.READY,
        original_chars=original_chars or None,
        thread_title=thread_title[:255],
        finished_at=timezone.now(),
    )
    job.original_chars = original_chars or None
    _delete_file(job)
    progress.clear(key)
    logger.info(
        "canvas import ready: job=%s canvas=%s chars=%d truncated_from=%s",
        job.pk, canvas.pk, len(content), original_chars,
    )

    _push(job.thread_id, canvas_updated_payload(canvas, job=job))
    if thread_title:
        _push(job.thread_id, {
            "event_type": "thread.title_updated", "thread_id": str(job.thread_id), "title": thread_title,
        })
    return CanvasImportJob.State.READY


def expire_stale_imports() -> int:
    """Fail jobs stuck PENDING/RUNNING for more than STALE_AFTER_MINUTES (a worker
    restart mid-import, a lost task)."""
    from datetime import timedelta

    from chat.models import CanvasImportJob

    cutoff = timezone.now() - timedelta(minutes=STALE_AFTER_MINUTES)
    stale = list(
        CanvasImportJob.objects.filter(
            state__in=[CanvasImportJob.State.PENDING, CanvasImportJob.State.RUNNING],
            created_at__lt=cutoff,
        ).values_list("pk", flat=True)[:200]
    )
    for pk in stale:
        fail_job(pk, "The import timed out. Try again.")
    return len(stale)
