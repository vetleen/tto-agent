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
            # Guarded: don't clobber an earlier failure reason (e.g.
            # "Cancelled by user.") with the task exception message.
            updated = SubAgentRun.objects.filter(pk=run_id_str).exclude(
                status=SubAgentRun.Status.FAILED,
            ).update(
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
    from chat.subagent_service import is_retryable_subagent_error, run_subagent

    try:
        run_subagent(uuid.UUID(run_id), deadline_seconds=540)
    except Exception as exc:
        if is_retryable_subagent_error(exc):
            # Transient (LLM rate-limit/overload/timeout, network blip, DB
            # hiccup). run_subagent left the row RUNNING so this retry re-enters
            # it; on_failure records FAILED once retries are exhausted.
            raise self.retry(exc=exc)
        # Terminal: run_subagent already recorded FAILED.
        if isinstance(exc, OperationalError) and "too many connections" in str(exc).lower():
            logger.error("Connection limit hit for sub-agent %s, failing permanently", run_id)
        raise


@shared_task(time_limit=30)
def expire_stale_subagent_runs() -> int:
    """Periodic cleanup of stuck subagent runs.

    Tolerates transient database unavailability (e.g. a Postgres restart or
    maintenance window). The cleanup is best-effort and idempotent, so when the
    DB is briefly unreachable we log and skip this tick rather than raising an
    unhandled error; the next beat tick retries once the DB is back. Logged at
    INFO so routine maintenance blips don't surface as Sentry errors (WILFRED-5K).
    """
    from django.db.utils import InterfaceError, OperationalError
    from chat.subagent_limits import _expire_stale_runs

    try:
        return _expire_stale_runs()
    except (OperationalError, InterfaceError):
        logger.info(
            "Skipping stale sub-agent cleanup: database temporarily unavailable; "
            "will retry on next beat tick.",
            exc_info=True,
        )
        return 0


@shared_task(time_limit=600, soft_time_limit=540)
def run_loop(loop_id: str) -> None:
    """Execute one scheduled Loop turn headlessly.

    Runs a full agent turn with no connected browser (see
    ``chat.loop_service.execute_loop_run``). On the default queue for now so it
    is always consumed; a dedicated lower-priority ``loops`` queue can be added
    later once every worker is confirmed to consume it.
    """
    from chat.loop_service import execute_loop_run

    execute_loop_run(uuid.UUID(loop_id))


@shared_task(time_limit=30)
def tick_and_scan_loops() -> int:
    """Periodic: enqueue every Loop that is due to fire.

    Tolerates transient database unavailability like the other sweepers — logs
    and skips this tick rather than raising an unhandled error.
    """
    from django.db.utils import InterfaceError, OperationalError

    from chat.loop_service import enqueue_due_loops

    try:
        return enqueue_due_loops()
    except (OperationalError, InterfaceError):
        logger.info(
            "Skipping loop scan: database temporarily unavailable; "
            "will retry on next beat tick.",
            exc_info=True,
        )
        return 0
