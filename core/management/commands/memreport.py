"""Enqueue (or run inline) an in-process memory report.

    manage.py memreport --tag burst1        # enqueue for the Celery worker
    manage.py memreport --local             # report this process instead

The worker variant logs ``MEMREPORT ondemand ...`` lines from inside the worker
process — read them with ``heroku logs --dyno worker | grep MEMREPORT``.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Log an in-process memory report (worker via Celery, or this process)."

    def add_arguments(self, parser):
        parser.add_argument("--tag", default="manual", help="Label carried in the log lines.")
        parser.add_argument(
            "--local", action="store_true",
            help="Report the current process instead of enqueuing for the worker.",
        )

    def handle(self, *args, **options):
        tag = options["tag"]
        if options["local"]:
            from core.memreport import memory_report

            result = memory_report(f"ondemand tag={tag}", full=True, force=True, collect=True)
            rss = result.get("rss_kb")
            self.stdout.write(f"reported rss={rss / 1024:.0f}MB" if rss else "reported")
            return

        from core.tasks import memory_report_task

        async_result = memory_report_task.delay(tag)
        self.stdout.write(f"enqueued memory_report_task id={async_result.id} tag={tag}")
