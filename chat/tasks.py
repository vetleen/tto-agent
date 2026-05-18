"""Celery tasks for the chat app."""

from __future__ import annotations

import logging
import uuid

from celery import Task, shared_task

logger = logging.getLogger(__name__)


def _notify_consumer(run_id: str, thread_id: str) -> None:
    """Best-effort channel-layer notification that a subagent run finished."""
    try:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"thread_{thread_id}",
            {
                "type": "subagent.completed",
                "run_id": run_id,
                "thread_id": thread_id,
            },
        )
    except Exception:
        logger.debug("Could not notify consumer of sub-agent %s completion", run_id)


class _SubagentTask(Task):
    """Custom task class that marks runs as permanently FAILED after all retries."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        from django.utils import timezone

        from chat.models import SubAgentRun

        run_id_str = args[0] if args else kwargs.get("run_id")
        if not run_id_str:
            return
        try:
            updated = SubAgentRun.objects.filter(pk=run_id_str).update(
                status=SubAgentRun.Status.FAILED,
                error=str(exc),
                completed_at=timezone.now(),
            )
            if updated:
                run = SubAgentRun.objects.filter(pk=run_id_str).values("thread_id").first()
                if run:
                    _notify_consumer(run_id_str, str(run["thread_id"]))
        except Exception:
            logger.exception("Failed to mark sub-agent run %s as FAILED", run_id_str)


@shared_task(
    base=_SubagentTask,
    bind=True,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 2},
    time_limit=600,
    soft_time_limit=540,
)
def run_subagent_task(self, run_id: str) -> None:
    """Execute a sub-agent run asynchronously via Celery."""
    from django.db.utils import OperationalError
    from chat.subagent_service import run_subagent

    try:
        run_subagent(uuid.UUID(run_id), deadline_seconds=540)
    except OperationalError as exc:
        if "too many connections" in str(exc).lower():
            logger.error("Connection limit hit for sub-agent %s, failing permanently", run_id)
            raise
        raise self.retry(exc=exc)
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise self.retry(exc=exc)


@shared_task(time_limit=30)
def expire_stale_subagent_runs() -> int:
    """Periodic cleanup of stuck subagent runs."""
    from chat.subagent_limits import _expire_stale_runs

    return _expire_stale_runs()
