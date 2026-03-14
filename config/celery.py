import logging
import os
import sys
from celery import Celery
from celery.signals import setup_logging, task_postrun

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


@task_postrun.connect
def close_db_connections_after_task(**kwargs):
    close_old_connections()


@app.task(bind=True)
def debug_task(self):
    logger.debug("Debug task request: %r", self.request)
