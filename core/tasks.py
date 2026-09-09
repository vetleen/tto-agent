"""Core Celery tasks (diagnostics)."""

from __future__ import annotations

import logging
import os

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, time_limit=30)
def restart_worker_nightly(self) -> str:
    """Warm-shutdown this worker so Heroku boots a fresh process.

    The allocator never returns the high-water mark to the OS, so the worker
    carries yesterday's memory peak (~85% of quota observed) until something
    restarts it. Heroku already cycles dynos every ~24h but at an arbitrary
    clock time; exiting here instead pins that daily restart to the quiet
    early morning — Heroku restarts a dyno whose process exits. Warm shutdown
    lets tasks that are already executing finish first; anything acked-early
    and dropped regardless is recovered by the requeue-stale periodic tasks.

    No-ops off Heroku (no ``DYNO`` env var): local workers share a broker
    with development runs and must never shut themselves down.
    """
    if not os.environ.get("DYNO"):
        logger.info("Nightly worker restart: skipped (not on a Heroku dyno).")
        return "skipped"
    hostname = self.request.hostname
    logger.info("Nightly worker restart: warm shutdown of %s.", hostname)
    # Target only the node that consumed this task; a bare broadcast would
    # also stop any other worker attached to the same broker.
    self.app.control.shutdown(destination=[hostname] if hostname else None)
    return "shutdown"


@shared_task(time_limit=60)
def memory_report_task(tag: str = "manual") -> dict:
    """Log a full in-process memory report from inside the worker.

    Runs regardless of ``MEM_DEBUG_WORKER`` — it is only ever enqueued on
    purpose (``manage.py memreport``), so a live worker can be inspected
    without a config change or restart. Returns the headline numbers.
    """
    from core.memreport import memory_report

    result = memory_report(f"ondemand tag={tag}", full=True, force=True, collect=True)
    return {
        "rss_kb": result.get("rss_kb"),
        "swap_kb": result.get("swap_kb"),
        "threads": result.get("threads"),
        "busy_threads": len(result.get("busy_threads") or []),
    }
