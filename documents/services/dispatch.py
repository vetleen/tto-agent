"""Execution queue for async document processing (the document dispatch gate).

Mirrors the sub-agent slot pattern (``chat.subagent_limits``): the
``DataRoomDocumentVersion`` table is the queue. ``queued_at`` marks queue
membership — only the gated async paths (upload, meeting export, the sweeper's
orphan recovery) set it, so versions the web dyno processes synchronously
(canvas/agent saves, ``enqueue=False``) are invisible to the dispatcher and can
never be double-processed. ``dispatched_at`` is the claim/slot marker: NULL
while a version waits in line (in no Celery queue, occupying no worker thread),
set when the dispatcher hands it to Celery, from which point it holds one of
``DOCUMENT_WORKER_SLOTS`` until the pipeline ends (READY / FAILED / terminal
scan_failed — release is implicit: terminal statuses drop out of the count).

Dispatch is level-triggered from every edge where a slot may free or a version
may join the line: the upload view, ``create_version(enqueue=True)``, every
pipeline task's ``after_return`` (via :class:`DocumentPipelineTask`), and the
stale-document sweeper as a backstop.
"""

from __future__ import annotations

import logging
import os

from celery import Task
from django.db import connection, transaction
from django.utils import timezone

logger = logging.getLogger(__name__)

# How many documents may occupy the worker at once, system-wide. This is the
# document-processing thread/memory lever; a change takes effect on the next
# dispatch, no restart. Distinct from DOCUMENT_EXTRACT_CONCURRENCY (the
# in-process memory backstop on the parse stage) and from
# DOCUMENT_MAX_IN_FLIGHT_PER_USER (the per-user admission quota).
DOCUMENT_WORKER_SLOTS = int(os.environ.get("DOCUMENT_WORKER_SLOTS", "4"))

# Transaction-scoped Postgres advisory lock serializing concurrent dispatchers.
# Unique among this app's advisory locks (sub-agents use 7211040001).
_DISPATCH_LOCK_KEY = 7211040002


def _slot_statuses():
    from documents.models import DataRoomDocument

    S = DataRoomDocument.Status
    return (S.UPLOADED, S.PROCESSING, S.SCANNING)


def _waiting_statuses():
    from documents.models import DataRoomDocument

    S = DataRoomDocument.Status
    # PROCESSING appears here only via a sweeper stale-reset (the row was
    # dispatched, went silent, and had its claim cleared to rejoin the line).
    return (S.UPLOADED, S.PROCESSING)


def _slots_in_use() -> int:
    """Versions holding an execution slot: dispatched and not yet terminal."""
    from documents.models import DataRoomDocumentVersion

    return DataRoomDocumentVersion.objects.filter(
        dispatched_at__isnull=False, status__in=_slot_statuses(),
    ).count()


def _waiting_versions():
    """Queued versions not yet handed to Celery, oldest first."""
    from documents.models import DataRoomDocumentVersion

    return DataRoomDocumentVersion.objects.filter(
        queued_at__isnull=False,
        dispatched_at__isnull=True,
        status__in=_waiting_statuses(),
    ).order_by("queued_at")


def get_queue_depth() -> dict:
    """Current queue state: slots in use, versions waiting, and the slot count."""
    return {
        "running": _slots_in_use(),
        "waiting": _waiting_versions().count(),
        "worker_slots": DOCUMENT_WORKER_SLOTS,
    }


def mark_version_queued(version_id: int) -> bool:
    """Put a version in the dispatch queue. Idempotent; returns whether it joined now."""
    from documents.models import DataRoomDocumentVersion

    return bool(
        DataRoomDocumentVersion.objects.filter(
            pk=version_id, queued_at__isnull=True,
        ).update(queued_at=timezone.now())
    )


def dispatch_pending_document_versions() -> list[int]:
    """Hand waiting versions to Celery while execution slots are free.

    Returns the version ids dispatched, oldest first. See the module docstring
    for the queue semantics; the shape (advisory lock → free count → CAS claim
    → publish after COMMIT → guarded revert) mirrors
    ``chat.subagent_limits.dispatch_pending_subagents`` — see the comments
    there for the locking rationale.
    """
    from documents.models import DataRoomDocumentVersion

    now = timezone.now()
    with transaction.atomic():
        if connection.vendor == "postgresql":
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", [_DISPATCH_LOCK_KEY])

        free = DOCUMENT_WORKER_SLOTS - _slots_in_use()
        if free <= 0:
            return []

        claimed: list[int] = []
        for version_id in _waiting_versions().values_list("id", flat=True)[:free]:
            # Conditional claim: loses the row to a concurrent terminal
            # transition. Also the only guard on SQLite (no advisory lock).
            if DataRoomDocumentVersion.objects.filter(
                pk=version_id,
                queued_at__isnull=False,
                dispatched_at__isnull=True,
                status__in=_waiting_statuses(),
            ).update(dispatched_at=now):
                claimed.append(version_id)

    if not claimed:
        return []

    # Publish after COMMIT: a broker publish can spend seconds in retries and
    # must not hold the advisory lock (or the PgBouncer server connection).
    from documents.tasks import process_document_version_task

    dispatched: list[int] = []
    for version_id in claimed:
        try:
            process_document_version_task.delay(version_id)
        except Exception as exc:
            # Back to waiting; the next trigger retries. Guarded on the claim
            # timestamp so an intervening transition isn't resurrected.
            DataRoomDocumentVersion.objects.filter(
                pk=version_id, dispatched_at=now, status__in=_waiting_statuses(),
            ).update(dispatched_at=None)
            logger.warning(
                "Could not enqueue document version %s (%s); left waiting in the queue",
                version_id, type(exc).__name__,
            )
            continue
        dispatched.append(version_id)
    return dispatched


def safe_dispatch(source: str) -> None:
    """Run the dispatcher, swallowing every error — trigger sites must not fail."""
    try:
        dispatched = dispatch_pending_document_versions()
        if dispatched:
            logger.info(
                "document dispatch (%s): version_ids=%s", source, dispatched,
            )
    except Exception:
        logger.warning("document dispatch (%s) failed", source, exc_info=True)


class DocumentPipelineTask(Task):
    """Base for pipeline tasks: every task end may free a slot, so re-dispatch.

    Celery skips ``after_return`` on RETRY, so a retrying task keeps its slot.
    Attached to the process, scan, and finalize tasks — a dispatch attempt while
    the version merely advanced a stage (still holding its slot) is a cheap
    no-op, and the one after the terminal stage is what drains the queue.
    """

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        safe_dispatch(f"after_return:{self.name}")
