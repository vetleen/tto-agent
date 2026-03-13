"""Celery tasks for the chat app."""

from __future__ import annotations

import logging
import uuid

from celery import Task, shared_task

logger = logging.getLogger(__name__)


class _SubagentTask(Task):
    """Custom task class that marks runs as permanently FAILED after all retries."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        from django.utils import timezone

        from chat.models import SubAgentRun

        run_id_str = args[0] if args else kwargs.get("run_id")
        if not run_id_str:
            return
        try:
            SubAgentRun.objects.filter(pk=run_id_str).update(
                status=SubAgentRun.Status.FAILED,
                error=str(exc),
                completed_at=timezone.now(),
            )
        except Exception:
            logger.exception("Failed to mark sub-agent run %s as FAILED", run_id_str)


@shared_task(
    base=_SubagentTask,
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
    time_limit=300,
    soft_time_limit=270,
)
def run_subagent_task(self, run_id: str) -> None:
    """Execute a sub-agent run asynchronously via Celery."""
    from chat.subagent_service import run_subagent

    run_subagent(uuid.UUID(run_id))
