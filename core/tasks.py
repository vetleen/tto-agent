"""Core Celery tasks (diagnostics)."""

from __future__ import annotations

import logging

from celery import shared_task

logger = logging.getLogger(__name__)


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
