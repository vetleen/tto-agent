"""Concurrency limits for sub-agent runs."""

from __future__ import annotations

import os


SUBAGENT_MAX_PER_USER = int(os.environ.get("SUBAGENT_MAX_PER_USER", "5"))
SUBAGENT_MAX_SYSTEM = int(os.environ.get("SUBAGENT_MAX_SYSTEM", "20"))


def check_subagent_limits(user) -> tuple[bool, str]:
    """Check whether the user can start a new sub-agent.

    Returns (allowed, error_message). If allowed is True, error_message is empty.
    """
    from chat.models import SubAgentRun

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
