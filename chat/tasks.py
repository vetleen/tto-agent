"""Celery tasks for the chat app."""

from __future__ import annotations

import logging
import uuid

from celery import Task, shared_task
from django.db.utils import OperationalError

from core.redis_errors import log_broadcast_failure

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
    except Exception as exc:
        # Fires once per sub-agent completion, so an unthrottled WARNING is safe.
        # A Redis blip here is worth an alert: the run finished but the browser
        # never hears about it, and the user just sees the sub-agent hang.
        log_broadcast_failure(
            logger,
            exc,
            "Could not notify consumer of sub-agent %s completion",
            run_id,
        )


def _capture_subagent_failure(exc: BaseException, run_id_str: str) -> None:
    """Guarantee a terminally-failed sub-agent run reaches Sentry.

    The ``LoggingIntegration`` that normally turns the upstream
    ``logger.error(exc_info=True)`` into an event has been observed to silently
    drop long-running provider failures — e.g. prod run b268f675, a ~562s
    Anthropic read-timeout that overran Celery's ``soft_time_limit`` (540s), so
    none of its error logs produced a Sentry event despite being written.

    ``on_failure`` fires exactly once per run, only after retries are exhausted,
    so we capture + flush *synchronously* here: the task is already finished, so
    the brief block is harmless, and ``flush`` blocks until the event is
    delivered (defeating any transport/teardown race). The full ``__cause__``
    chain (APITimeoutError → httpx/httpcore ReadTimeout) rides along with the
    captured exception.
    """
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_tag("subagent_run_id", str(run_id_str))
            error_code = getattr(exc, "error_code", None)
            if error_code:
                scope.set_tag("llm_error_code", error_code)
            sentry_sdk.capture_exception(exc)
        sentry_sdk.flush(timeout=5)
    except Exception:
        # Never let observability break the failure path.
        logger.warning(
            "Failed to capture sub-agent run %s failure to Sentry", run_id_str,
            exc_info=True,
        )


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
                run = SubAgentRun.objects.filter(pk=run_id_str).first()
                if run:
                    # Surface the failure to the orchestrator so it reacts
                    # (retry / do the work itself / tell the user) instead of
                    # the thread going silent. Deduplicated internally — the
                    # terminal-failure canvas path may already have written a
                    # result message for this run.
                    from chat.subagent_service import _create_subagent_failure_message

                    try:
                        _create_subagent_failure_message(run)
                    except Exception:
                        logger.exception(
                            "Failed to persist failure message for sub-agent run %s",
                            run_id_str,
                        )
                    _notify_consumer(run_id_str, str(run.thread_id))
        except Exception:
            logger.exception("Failed to mark sub-agent run %s as FAILED", run_id_str)

        _capture_subagent_failure(exc, run_id_str)

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        # This run released its execution slot (SUCCESS, or FAILURE after
        # on_failure ran): hand the slot to the next run waiting in line. Celery
        # skips after_return on RETRY, so a retrying run keeps its slot. Must
        # never raise — an exception escaping here makes the tracer report a
        # finished task as an internal failure.
        try:
            from chat.subagent_limits import dispatch_pending_subagents

            dispatch_pending_subagents()
        except Exception:
            logger.exception(
                "Sub-agent dispatch after run %s failed",
                args[0] if args else kwargs.get("run_id"),
            )


@shared_task(
    base=_SubagentTask,
    bind=True,
    # Read by the manual self.retry() below (retry_backoff / retry_kwargs only
    # apply with autoretry_for). Short delay: a retrying run holds an execution
    # slot while it waits.
    max_retries=2,
    default_retry_delay=30,
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
    from chat.subagent_limits import _expire_stale_runs, dispatch_pending_subagents

    try:
        expired = _expire_stale_runs()
        # Backstop dispatcher: covers a missed after_return (a worker restart
        # left RUNNING ghosts that just expired above, freeing slots), a
        # publish that failed and was reverted, and a cancelled run that Celery
        # discarded at delivery without running the task.
        dispatch_pending_subagents()
        return expired
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


class _SlideRenderTask(Task):
    """Marks a SlideRenderRun FAILED after retries are exhausted (mirrors _SubagentTask)."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        from django.utils import timezone

        from chat.models import SlideRenderRun

        run_id = args[0] if args else kwargs.get("run_id")
        if not run_id:
            return
        try:
            updated = (
                SlideRenderRun.objects.filter(pk=run_id)
                .exclude(status=SlideRenderRun.Status.COMPLETED)
                .update(
                    status=SlideRenderRun.Status.FAILED,
                    error=str(exc)[:2000],
                    finished_at=timezone.now(),
                )
            )
            if updated:
                run = (
                    SlideRenderRun.objects.filter(pk=run_id).select_related("slide_set").first()
                )
                if run and run.purpose in (
                    SlideRenderRun.Purpose.USER_PREVIEW,
                    SlideRenderRun.Purpose.PDF_EXPORT,
                ):
                    from chat.slides.render_service import notify_render_event

                    notify_render_event(run.slide_set, run, "slidedeck.render_failed")
        except Exception:
            logger.exception("Failed to mark slide render run %s FAILED", run_id)


@shared_task(
    base=_SlideRenderTask,
    bind=True,
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    time_limit=600,
    soft_time_limit=540,
)
def render_deck_task(self, run_id: str) -> None:
    """Render a slide deck (preview PNGs or export PDF) on the worker."""
    from chat.slides.render_service import execute_render_run

    execute_render_run(str(run_id))


@shared_task(time_limit=60)
def expire_stale_slide_renders() -> int:
    """Fail runs stuck RUNNING>15min / PENDING>30min; prune old runs (>7d).

    Tolerates transient DB unavailability like the other sweepers.
    """
    from datetime import timedelta

    from django.db.utils import InterfaceError, OperationalError
    from django.utils import timezone

    from chat.models import Asset, SlideRender, SlideRenderRun

    try:
        now = timezone.now()
        stale_running = SlideRenderRun.objects.filter(
            status=SlideRenderRun.Status.RUNNING, started_at__lt=now - timedelta(minutes=15)
        ).update(status=SlideRenderRun.Status.FAILED, error="Render timed out (stale).", finished_at=now)
        stale_pending = SlideRenderRun.objects.filter(
            status=SlideRenderRun.Status.PENDING, created_at__lt=now - timedelta(minutes=30)
        ).update(status=SlideRenderRun.Status.FAILED, error="Render never started (stale).", finished_at=now)
        SlideRenderRun.objects.filter(created_at__lt=now - timedelta(days=7)).delete()

        # Prune orphaned deck-owned assets (blobs cleaned by the Asset post_delete
        # signal). Render images (slide_set + KIND_IMAGE) are only ever referenced
        # by SlideRender; the 1h age guard avoids racing an in-flight render.
        # Export PDFs (KIND_FILE) are ephemeral downloads — drop after 7 days.
        Asset.objects.filter(
            slide_set__isnull=False, kind=Asset.KIND_IMAGE,
            created_at__lt=now - timedelta(hours=1),
        ).exclude(
            pk__in=SlideRender.objects.filter(asset__isnull=False).values("asset_id")
        ).delete()
        Asset.objects.filter(
            slide_set__isnull=False, kind=Asset.KIND_FILE,
            created_at__lt=now - timedelta(days=7),
        ).delete()
        return stale_running + stale_pending
    except (OperationalError, InterfaceError):
        logger.info(
            "Skipping stale slide-render cleanup: database temporarily unavailable.",
            exc_info=True,
        )
        return 0
