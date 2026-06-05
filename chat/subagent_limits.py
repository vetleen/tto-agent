"""Concurrency limits for sub-agent runs."""

from __future__ import annotations

import os
from datetime import timedelta

from django.db import transaction
from django.utils import timezone


SUBAGENT_MAX_PER_USER = int(os.environ.get("SUBAGENT_MAX_PER_USER", "4"))
SUBAGENT_MAX_SYSTEM = int(os.environ.get("SUBAGENT_MAX_SYSTEM", "8"))
SUBAGENT_WORKER_SLOTS = int(os.environ.get("SUBAGENT_WORKER_SLOTS", "4"))

STALE_PENDING_MINUTES = 7
STALE_RUNNING_MINUTES = 10


def _expire_stale_runs() -> int:
    """Mark sub-agent runs stuck in PENDING/RUNNING as FAILED.

    Returns the number of expired runs.
    """
    from chat.models import SubAgentRun

    now = timezone.now()

    stale_pending = list(SubAgentRun.objects.filter(
        status=SubAgentRun.Status.PENDING,
        created_at__lt=now - timedelta(minutes=STALE_PENDING_MINUTES),
    ).values_list("id", "thread_id"))

    stale_running = list(SubAgentRun.objects.filter(
        status=SubAgentRun.Status.RUNNING,
        created_at__lt=now - timedelta(minutes=STALE_RUNNING_MINUTES),
    ).values_list("id", "thread_id"))

    expired = 0
    if stale_pending:
        expired += SubAgentRun.objects.filter(
            pk__in=[r[0] for r in stale_pending],
        ).update(
            status=SubAgentRun.Status.FAILED,
            error="Expired: stuck in pending too long.",
            completed_at=now,
        )
    if stale_running:
        expired += SubAgentRun.objects.filter(
            pk__in=[r[0] for r in stale_running],
        ).update(
            status=SubAgentRun.Status.FAILED,
            error="Expired: stuck in running too long.",
            completed_at=now,
        )

    if expired:
        _notify_expired_threads(stale_pending + stale_running)

    return expired


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


def get_queue_depth() -> dict:
    """Return current queue state: how many running and how many pending."""
    from chat.models import SubAgentRun

    running = SubAgentRun.objects.filter(status=SubAgentRun.Status.RUNNING).count()
    pending = SubAgentRun.objects.filter(status=SubAgentRun.Status.PENDING).count()
    return {
        "running": running,
        "pending": pending,
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
        locked_qs = SubAgentRun.objects.select_for_update().filter(
            status__in=active_statuses,
        )
        user_count = locked_qs.filter(user=user).count()
        if user_count >= SUBAGENT_MAX_PER_USER:
            return (None, "You have too many sub-agents running. Please wait for some to finish.")

        system_count = locked_qs.count()
        if system_count >= SUBAGENT_MAX_SYSTEM:
            return (None, "The system is busy. Please try again shortly.")

        run = SubAgentRun.objects.create(user=user, **run_kwargs)
        return (run, "")
