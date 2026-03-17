"""Concurrency limits for sub-agent runs."""

from __future__ import annotations

import os
from datetime import timedelta

from django.db import transaction
from django.utils import timezone


SUBAGENT_MAX_PER_USER = int(os.environ.get("SUBAGENT_MAX_PER_USER", "4"))
SUBAGENT_MAX_SYSTEM = int(os.environ.get("SUBAGENT_MAX_SYSTEM", "20"))

STALE_PENDING_MINUTES = 10
STALE_RUNNING_MINUTES = 15


def _expire_stale_runs() -> int:
    """Mark sub-agent runs stuck in PENDING/RUNNING as FAILED.

    Returns the number of expired runs.
    """
    from chat.models import SubAgentRun

    now = timezone.now()
    expired = 0
    expired += SubAgentRun.objects.filter(
        status=SubAgentRun.Status.PENDING,
        created_at__lt=now - timedelta(minutes=STALE_PENDING_MINUTES),
    ).update(
        status=SubAgentRun.Status.FAILED,
        error="Expired: stuck in pending too long.",
        completed_at=now,
    )
    expired += SubAgentRun.objects.filter(
        status=SubAgentRun.Status.RUNNING,
        created_at__lt=now - timedelta(minutes=STALE_RUNNING_MINUTES),
    ).update(
        status=SubAgentRun.Status.FAILED,
        error="Expired: stuck in running too long.",
        completed_at=now,
    )
    return expired


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
        # Lock the user's active runs to prevent concurrent creates
        # from both passing the check.  select_for_update is a no-op
        # on SQLite (used in tests) but provides real locking on Postgres.
        locked_qs = SubAgentRun.objects.select_for_update().filter(
            user=user, status__in=active_statuses,
        )
        user_count = locked_qs.count()
        if user_count >= SUBAGENT_MAX_PER_USER:
            return (None, "You have too many sub-agents running. Please wait for some to finish.")

        system_count = SubAgentRun.objects.filter(
            status__in=active_statuses,
        ).count()
        if system_count >= SUBAGENT_MAX_SYSTEM:
            return (None, "The system is busy. Please try again shortly.")

        run = SubAgentRun.objects.create(user=user, **run_kwargs)
        return (run, "")
