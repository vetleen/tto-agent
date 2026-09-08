"""Concurrency limits and the execution queue for sub-agent runs.

Admission (``create_subagent_run_if_allowed``) is a hard door: a spawn is
refused when the user or the system already has too many active runs
(waiting + running). Execution is gated separately by
``dispatch_pending_subagents``: admitted runs wait in arrival order as PENDING
rows and are handed to Celery only while fewer than ``SUBAGENT_WORKER_SLOTS``
runs hold an execution slot.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.db import connection, transaction
from django.db.models import Q
from django.utils import timezone

logger = logging.getLogger(__name__)


# Per-user cap on active runs (waiting + running). Hard deny at admission.
SUBAGENT_MAX_PER_USER = int(os.environ.get("SUBAGENT_MAX_PER_USER", "4"))
# System-wide cap on active runs (waiting + running), i.e. the queue depth.
# Hard deny at admission ("system busy").
SUBAGENT_MAX_SYSTEM = int(os.environ.get("SUBAGENT_MAX_SYSTEM", "8"))
# How many runs may execute at once, system-wide. This is the sub-agent memory
# lever on the worker; a change takes effect on the next dispatch, no restart.
SUBAGENT_WORKER_SLOTS = int(os.environ.get("SUBAGENT_WORKER_SLOTS", "4"))

# A dispatched run no worker has started within this window is lost (worker
# restart, dropped message) or the worker is starved of threads; expire it.
# Clocked from dispatched_at — time spent waiting in line doesn't count.
STALE_PENDING_MINUTES = 7
# Safety valve for a runaway queue: a run still waiting for a slot after this
# long is given up on. Bounds the wait; it is not a throughput knob.
STALE_WAITING_MINUTES = 45
# Must comfortably exceed the Celery run_subagent_task hard time_limit (600s = 10
# min) so the sweeper never expires a run the worker is still legitimately
# executing (which would discard its completed result).
STALE_RUNNING_MINUTES = 15

# Transaction-scoped Postgres advisory lock that serializes dispatchers.
# Arbitrary value; only has to be unique among the advisory locks this app takes.
_DISPATCH_LOCK_KEY = 7211040001


def _expire_stale_runs() -> int:
    """Mark sub-agent runs stuck in PENDING/RUNNING as FAILED.

    Returns the number of expired runs.
    """
    from chat.models import SubAgentRun

    now = timezone.now()

    # Dispatched but never picked up by a worker: the message was lost or the
    # worker is starved. Clocked from dispatched_at so a run that legitimately
    # waited in line isn't killed the moment it is handed to Celery.
    stale_pending = list(SubAgentRun.objects.filter(
        status=SubAgentRun.Status.PENDING,
        dispatched_at__lt=now - timedelta(minutes=STALE_PENDING_MINUTES),
    ).values_list("id", "thread_id"))

    # Still waiting for an execution slot after the safety-valve window.
    stale_waiting = list(SubAgentRun.objects.filter(
        status=SubAgentRun.Status.PENDING,
        dispatched_at__isnull=True,
        created_at__lt=now - timedelta(minutes=STALE_WAITING_MINUTES),
    ).values_list("id", "thread_id"))

    # Measure RUNNING staleness from started_at, not created_at — a run that
    # waited in the queue before starting must get its full execution window
    # (created_at fallback covers legacy rows that predate started_at).
    running_cutoff = now - timedelta(minutes=STALE_RUNNING_MINUTES)
    stale_running = list(SubAgentRun.objects.filter(
        status=SubAgentRun.Status.RUNNING,
    ).filter(
        Q(started_at__lt=running_cutoff)
        | Q(started_at__isnull=True, created_at__lt=running_cutoff)
    ).values_list("id", "thread_id"))

    expired = 0
    # Re-assert the status in the UPDATE filter: a run that transitioned since the
    # SELECT above (PENDING→RUNNING, or RUNNING→COMPLETED) must not be clobbered to
    # FAILED — that would kill a legitimately-started run or discard a just-finished
    # result.
    if stale_pending:
        expired += SubAgentRun.objects.filter(
            pk__in=[r[0] for r in stale_pending],
            status=SubAgentRun.Status.PENDING,
        ).update(
            status=SubAgentRun.Status.FAILED,
            error="Expired: stuck in pending too long.",
            completed_at=now,
        )
    if stale_waiting:
        expired += SubAgentRun.objects.filter(
            pk__in=[r[0] for r in stale_waiting],
            status=SubAgentRun.Status.PENDING,
            dispatched_at__isnull=True,
        ).update(
            status=SubAgentRun.Status.FAILED,
            error="Expired: waited in the queue too long.",
            completed_at=now,
        )
    if stale_running:
        expired += SubAgentRun.objects.filter(
            pk__in=[r[0] for r in stale_running],
            status=SubAgentRun.Status.RUNNING,
        ).update(
            status=SubAgentRun.Status.FAILED,
            error="Expired: stuck in running too long.",
            completed_at=now,
        )

    if expired:
        all_stale = stale_pending + stale_waiting + stale_running
        _persist_expiry_failure_messages([r[0] for r in all_stale])
        _notify_expired_threads(all_stale)

    return expired


def _persist_expiry_failure_messages(run_ids: list) -> None:
    """Write a hidden failure message for each run the sweeper just expired.

    Makes the expiry visible to the orchestrator (via the unreported-claim
    logic) instead of only decrementing the status bar. Best-effort; scoped to
    rows the guarded updates actually flipped (status=FAILED with the expiry
    error), so a run that legitimately transitioned mid-sweep is untouched.
    """
    try:
        from chat.models import SubAgentRun
        from chat.subagent_service import _create_subagent_failure_message

        expired_runs = SubAgentRun.objects.filter(
            pk__in=run_ids,
            status=SubAgentRun.Status.FAILED,
            error__startswith="Expired:",
        )
        for run in expired_runs:
            _create_subagent_failure_message(run)
    except Exception:
        logger.exception(
            "Failed to persist failure messages for expired sub-agent runs"
        )


def _notify_expired_threads(expired_runs: list[tuple]) -> None:
    """Best-effort notify consumers of expired stale runs."""
    try:
        from chat.tasks import _notify_consumer

        notified = set()
        for run_id, thread_id in expired_runs:
            tid = str(thread_id)
            if tid not in notified:
                notified.add(tid)
                _notify_consumer(str(run_id), tid)
    except Exception:
        pass


def check_subagent_limits(user) -> tuple[bool, str]:
    """Check whether the user can start a new sub-agent.

    Returns (allowed, error_message). If allowed is True, error_message is empty.
    """
    from chat.models import SubAgentRun

    _expire_stale_runs()

    active_statuses = [SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING]

    user_count = SubAgentRun.objects.filter(
        user=user, status__in=active_statuses,
    ).count()
    if user_count >= SUBAGENT_MAX_PER_USER:
        return (False, "You have too many sub-agents running. Please wait for some to finish.")

    system_count = SubAgentRun.objects.filter(
        status__in=active_statuses,
    ).count()
    if system_count >= SUBAGENT_MAX_SYSTEM:
        return (False, "The system is busy. Please try again shortly.")

    return (True, "")


def _slots_in_use() -> int:
    """Runs holding an execution slot: RUNNING, plus PENDING runs already handed to Celery."""
    from chat.models import SubAgentRun

    return SubAgentRun.objects.filter(
        Q(status=SubAgentRun.Status.RUNNING)
        | Q(status=SubAgentRun.Status.PENDING, dispatched_at__isnull=False)
    ).count()


def _waiting_runs():
    """PENDING runs not yet handed to Celery, oldest first."""
    from chat.models import SubAgentRun

    return SubAgentRun.objects.filter(
        status=SubAgentRun.Status.PENDING, dispatched_at__isnull=True,
    ).order_by("created_at")


def get_queue_depth() -> dict:
    """Current queue state: slots in use, runs waiting for a slot, and the slot count."""
    return {
        "running": _slots_in_use(),
        "waiting": _waiting_runs().count(),
        "worker_slots": SUBAGENT_WORKER_SLOTS,
    }


def create_subagent_run_if_allowed(user, **run_kwargs):
    """Atomically check limits and create a SubAgentRun.

    Uses a transaction with select_for_update to prevent race conditions
    where two concurrent requests both pass the limit check.

    Returns (run, "") on success or (None, error_message) on denial.
    """
    from chat.models import SubAgentRun

    _expire_stale_runs()

    active_statuses = [SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING]

    with transaction.atomic():
        # Lock the user row to serialize concurrent creates for the SAME user —
        # the realistic race is one turn's parallel chat_subagent_create calls
        # (simple_chat runs tool calls in a ThreadPoolExecutor). select_for_update
        # on the COUNT query itself takes no lock (Django strips FOR UPDATE from
        # aggregates), so the per-user cap needs a real row lock to hold. The
        # system-wide cap stays best-effort.
        get_user_model().objects.select_for_update().get(pk=user.pk)

        active = SubAgentRun.objects.filter(status__in=active_statuses)
        user_count = active.filter(user=user).count()
        if user_count >= SUBAGENT_MAX_PER_USER:
            return (None, "You have too many sub-agents running. Please wait for some to finish.")

        system_count = active.count()
        if system_count >= SUBAGENT_MAX_SYSTEM:
            return (None, "The system is busy. Please try again shortly.")

        run = SubAgentRun.objects.create(user=user, **run_kwargs)
        return (run, "")


def dispatch_pending_subagents() -> list[uuid.UUID]:
    """Hand waiting runs to Celery while execution slots are free.

    The SubAgentRun table is the queue: a PENDING row with ``dispatched_at``
    NULL waits in line and holds no slot; claiming it (setting
    ``dispatched_at``) makes it hold one until the run finishes. Called wherever
    a slot may have opened or a run may have joined the line — the create tool,
    the task's ``after_return``, the user-cancel path, and the stale-run
    sweeper as a backstop. Returns the ids dispatched, oldest first.
    """
    from chat.models import SubAgentRun

    now = timezone.now()
    with transaction.atomic():
        if connection.vendor == "postgresql":
            # Serialize concurrent dispatchers (web-dyno tool call, worker
            # after_return, beat sweeper): two of them reading the same free
            # count would each claim *different* rows and overshoot the slot
            # cap — the conditional claim below only prevents claiming the same
            # row twice. Transaction-scoped, so it is released at COMMIT and is
            # safe behind PgBouncer in transaction mode (a session-level
            # pg_advisory_lock would not be: the server connection can change
            # between transactions).
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", [_DISPATCH_LOCK_KEY])

        free = SUBAGENT_WORKER_SLOTS - _slots_in_use()
        if free <= 0:
            return []

        claimed: list[uuid.UUID] = []
        for run_id in _waiting_runs().values_list("id", flat=True)[:free]:
            # Conditional claim: a cancel that flipped the row to FAILED in the
            # meantime loses it. Also the only guard on SQLite (no advisory lock).
            if SubAgentRun.objects.filter(
                pk=run_id,
                status=SubAgentRun.Status.PENDING,
                dispatched_at__isnull=True,
            ).update(dispatched_at=now):
                claimed.append(run_id)

    if not claimed:
        return []

    # Publish after COMMIT: a broker publish can spend seconds in retries and
    # must not hold the advisory lock (or the PgBouncer server connection).
    from chat.tasks import run_subagent_task

    dispatched: list[uuid.UUID] = []
    for run_id in claimed:
        try:
            task = run_subagent_task.delay(str(run_id))
        except Exception as exc:
            # Back to waiting; the sweeper retries on its next tick. Guarded on
            # the claim timestamp so an intervening cancel isn't resurrected.
            SubAgentRun.objects.filter(
                pk=run_id, status=SubAgentRun.Status.PENDING, dispatched_at=now,
            ).update(dispatched_at=None, celery_task_id="")
            logger.warning(
                "Could not enqueue sub-agent run %s (%s); left waiting in the queue",
                run_id, type(exc).__name__,
            )
            continue
        SubAgentRun.objects.filter(pk=run_id).update(celery_task_id=task.id)
        dispatched.append(run_id)
    return dispatched
