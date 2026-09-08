import logging
import os
import sys
from celery import Celery
from celery.signals import (
    setup_logging,
    task_failure,
    task_postrun,
    task_prerun,
    worker_init,
    worker_ready,
)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
# Opt in explicitly to retrying the broker connection during worker startup.
# This is the Celery 6 default; setting it silences the CPendingDeprecationWarning
# and locks current behaviour in ahead of that upgrade.
app.conf.broker_connection_retry_on_startup = True
# Prefork pool causes PermissionError on Windows (billiard semaphores). Use solo.
if sys.platform == "win32":
    app.conf.worker_pool = "solo"
app.autodiscover_tasks()

logger = logging.getLogger(__name__)


@setup_logging.connect
def configure_celery_logging(**kwargs):
    """Use Django's LOGGING config instead of Celery's default."""
    pass


from django.db import close_old_connections


@task_prerun.connect
def set_sentry_celery_tags(task_id, task, **kwargs):
    """Clean stale connections from prior tasks, then tag for Sentry."""
    close_old_connections()
    try:
        import sentry_sdk as _sentry_sdk
        _sentry_sdk.set_tag("celery_task_id", task_id)
        _sentry_sdk.set_tag("celery_task_name", task.name)
    except ImportError:
        pass


@task_postrun.connect
def close_db_connections_after_task(**kwargs):
    close_old_connections()


# --- Opt-in worker memory attribution (MEM_DEBUG_WORKER=1) -------------------
# core.memtrace / core.malloc_trim deliberately refuse to run under Celery, so
# the worker had no way to say WHERE its RSS lives. These hooks are no-ops
# unless MEM_DEBUG_WORKER is truthy; see core.memreport for what they log.
# Connected before the malloc_trim hook below so a task_end report shows the
# pre-trim state (the trim then logs its own before->after line).

@worker_init.connect
def start_worker_memory_tracing(**kwargs):
    """Start tracemalloc early (before the pool) when MEM_DEBUG_TRACEMALLOC asks."""
    from core.memreport import ensure_tracemalloc, worker_debug_enabled
    if worker_debug_enabled():
        ensure_tracemalloc()


@worker_ready.connect
def start_worker_memory_sampler(**kwargs):
    from core.memreport import maybe_start_worker_sampler
    maybe_start_worker_sampler()


@task_prerun.connect
def memory_report_before_task(task_id, task, **kwargs):
    from core.memreport import task_prerun_report
    task_prerun_report(task_id, task.name)


@task_postrun.connect
def memory_report_after_task(task_id, task, **kwargs):
    from core.memreport import task_postrun_report
    task_postrun_report(task_id, task.name)


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to *default* on missing/garbage."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@task_postrun.connect
def trim_worker_memory_after_task(**kwargs):
    """Return freed glibc arenas to the OS after each task so the worker's RSS
    recovers after memory-heavy tasks (sub-agent web research builds several-MB
    bs4/lxml trees and streams thousands of LLM events; freed at task end but
    otherwise pinned in glibc's arenas -> sustained R14 even when idle).

    The web dyno has a periodic malloc_trim daemon; the Celery worker has none,
    so this is its per-task equivalent. Threshold- and rate-limited inside
    ``core.malloc_trim.maybe_trim`` (a no-op off glibc / when RSS is lean)."""
    from core.malloc_trim import maybe_trim
    maybe_trim(
        _env_float("WORKER_MALLOC_TRIM_THRESHOLD_MB", 700.0),
        _env_float("WORKER_MALLOC_TRIM_MIN_INTERVAL_S", 15.0),
    )


@task_failure.connect
def close_db_connections_on_failure(**kwargs):
    close_old_connections()


@app.task(bind=True)
def debug_task(self):
    logger.debug("Debug task request: %r", self.request)
