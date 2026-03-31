import logging
import os
import sys
from celery import Celery
from celery.signals import setup_logging, task_postrun, task_prerun

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
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
    """Tag Sentry events with Celery task metadata for correlation."""
    try:
        import sentry_sdk as _sentry_sdk
        _sentry_sdk.set_tag("celery_task_id", task_id)
        _sentry_sdk.set_tag("celery_task_name", task.name)
    except ImportError:
        pass


@task_postrun.connect
def close_db_connections_after_task(**kwargs):
    close_old_connections()


@app.task(bind=True)
def debug_task(self):
    logger.debug("Debug task request: %r", self.request)
