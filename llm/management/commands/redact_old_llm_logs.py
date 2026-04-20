"""Redact prompt/raw_output/tools on LLMCallLog rows older than the retention
window. Row count, model, tokens, cost, and timings are preserved for
cost/usage analytics.

Designed to run daily via Heroku Scheduler; idempotent — already-redacted rows
are skipped.

Usage:
    python manage.py redact_old_llm_logs
    python manage.py redact_old_llm_logs --days 30
    python manage.py redact_old_llm_logs --batch-size 500 --dry-run
"""
from __future__ import annotations

import time
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from llm.models import LLMCallLog

REDACTED_PROMPT = {"redacted": True}


class Command(BaseCommand):
    help = (
        "Redact prompt, raw_output, and tools on LLMCallLog rows older than "
        "--days (default 90). Safe to run repeatedly."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--days",
            type=int,
            default=90,
            help="Redact rows whose created_at is older than N days (default: 90).",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Rows per UPDATE batch (default: 1000).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Count and report matching rows without modifying them.",
        )

    def handle(self, *args, **options):
        days = options["days"]
        batch_size = options["batch_size"]
        dry_run = options["dry_run"]

        cutoff = timezone.now() - timedelta(days=days)
        base_qs = LLMCallLog.objects.filter(created_at__lt=cutoff).exclude(
            prompt=REDACTED_PROMPT
        )

        matching = base_qs.count()
        if matching == 0:
            self.stdout.write(self.style.SUCCESS("No rows to redact."))
            return

        self.stdout.write(
            f"{matching} row(s) older than {days} day(s) (cutoff {cutoff.isoformat()}) "
            f"awaiting redaction."
        )

        if dry_run:
            self.stdout.write(self.style.WARNING("--dry-run: no rows modified."))
            return

        total_updated = 0
        start = time.monotonic()
        while True:
            # Re-query each loop so we never hold a big ID list in memory.
            ids = list(
                LLMCallLog.objects.filter(created_at__lt=cutoff)
                .exclude(prompt=REDACTED_PROMPT)
                .values_list("pk", flat=True)[:batch_size]
            )
            if not ids:
                break
            updated = LLMCallLog.objects.filter(pk__in=ids).update(
                prompt=REDACTED_PROMPT,
                raw_output="",
                tools=None,
            )
            total_updated += updated
            self.stdout.write(f"  redacted batch of {updated} (total {total_updated})")
            if len(ids) < batch_size:
                break

        elapsed = time.monotonic() - start
        self.stdout.write(
            self.style.SUCCESS(
                f"Redacted {total_updated} row(s) in {elapsed:.2f}s."
            )
        )
