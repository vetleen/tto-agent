"""Enforce data-retention policies: delete expired records and redact old LLM logs.

Designed to run daily via Heroku Scheduler; idempotent.
Rows with retain_until IS NULL are never deleted (safety net).

Usage:
    python manage.py enforce_retention
    python manage.py enforce_retention --dry-run
    python manage.py enforce_retention --target ChatThread --batch-size 500
"""
from __future__ import annotations

import time
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

REDACTED_PROMPT = {"redacted": True}

ALL_TARGETS = [
    "ChatThread",
    "DataRoom",
    "Meeting",
    "GuardrailEvent",
    "Feedback",
    "EmailVerificationToken",
    "LLMCallLog",
]


def _build_targets(now):
    from accounts.models import EmailVerificationToken
    from chat.models import ChatThread
    from documents.models import DataRoom
    from feedback.models import Feedback
    from guardrails.models import GuardrailEvent
    from llm.models import LLMCallLog
    from meetings.models import Meeting

    retain_filter = dict(retain_until__isnull=False, retain_until__lt=now)
    return [
        ("ChatThread", "delete", ChatThread.objects.filter(**retain_filter)),
        ("DataRoom", "delete", DataRoom.objects.filter(**retain_filter)),
        ("Meeting", "delete", Meeting.objects.filter(**retain_filter)),
        ("GuardrailEvent", "delete", GuardrailEvent.objects.filter(**retain_filter)),
        ("Feedback", "delete", Feedback.objects.filter(**retain_filter)),
        (
            "EmailVerificationToken",
            "delete",
            EmailVerificationToken.objects.filter(
                created_at__lt=now - timedelta(days=1),
            ),
        ),
        (
            "LLMCallLog",
            "redact",
            LLMCallLog.objects.filter(
                created_at__lt=now - timedelta(days=90),
            ).exclude(prompt=REDACTED_PROMPT),
        ),
    ]


class Command(BaseCommand):
    help = "Enforce data-retention policies: delete expired records and redact old LLM logs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Rows per batch (default: 100).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Count and report matching rows without modifying them.",
        )
        parser.add_argument(
            "--target",
            type=str,
            default="",
            choices=[""] + ALL_TARGETS,
            help="Process only this target.",
        )

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        dry_run = options["dry_run"]
        target_filter = options["target"]

        now = timezone.now()
        start = time.monotonic()
        grand_total = 0

        for label, action, base_qs in _build_targets(now):
            if target_filter and label != target_filter:
                continue

            count = base_qs.count()
            if count == 0:
                self.stdout.write(f"  {label}: 0 rows to {action}.")
                continue

            self.stdout.write(f"  {label}: {count} row(s) to {action}.")
            if dry_run:
                continue

            if action == "delete":
                grand_total += self._batch_delete(base_qs, batch_size)
            else:
                grand_total += self._batch_redact(base_qs, batch_size)

        elapsed = time.monotonic() - start
        if dry_run:
            self.stdout.write(self.style.WARNING("--dry-run: no rows modified."))
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Retention enforcement complete: {grand_total} row(s) "
                    f"processed in {elapsed:.2f}s."
                )
            )

    def _batch_delete(self, base_qs, batch_size):
        total = 0
        model_cls = base_qs.model
        while True:
            ids = list(base_qs.values_list("pk", flat=True)[:batch_size])
            if not ids:
                break
            deleted_count, _ = model_cls.objects.filter(pk__in=ids).delete()
            total += deleted_count
            self.stdout.write(
                f"    deleted batch ({deleted_count} rows incl. cascades, "
                f"total {total})"
            )
            if len(ids) < batch_size:
                break
        return total

    def _batch_redact(self, base_qs, batch_size):
        total = 0
        model_cls = base_qs.model
        while True:
            ids = list(base_qs.values_list("pk", flat=True)[:batch_size])
            if not ids:
                break
            updated = model_cls.objects.filter(pk__in=ids).update(
                prompt=REDACTED_PROMPT,
                raw_output="",
                tools=None,
            )
            total += updated
            self.stdout.write(f"    redacted batch of {updated} (total {total})")
            if len(ids) < batch_size:
                break
        return total
