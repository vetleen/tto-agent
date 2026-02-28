import os
import sys
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
# Prefork pool causes PermissionError on Windows (billiard semaphores). Use solo.
if sys.platform == "win32":
    app.conf.worker_pool = "solo"
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
