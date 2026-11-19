"""Enforce data-retention policies: delete expired records and redact old LLM logs.

Designed to run daily via Heroku Scheduler; idempotent.
Rows with retain_until IS NULL are never deleted (safety net).

LLMCallLog rows are never deleted (llm/retention.py): target ``LLMCallLog``
redacts their content after LLM_LOG_CONTENT_RETENTION_DAYS, target
``LLMCallLogIdentity`` removes the user and the run/trace/conversation/document
references after LLM_LOG_IDENTITY_RETENTION_DAYS (organization kept on org_id).

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

from llm.retention import anonymize_llm_call_logs, redact_llm_call_logs
from llm.retention import REDACTED_PROMPT  # noqa: F401 — re-exported for callers

ALL_TARGETS = [
    "ChatThread",
    "DataRoom",
    "Meeting",
    "GuardrailEvent",
    "PIIReviewEvent",
    "Feedback",
    "EmailVerificationToken",
    "LLMCallLog",
    "LLMCallLogIdentity",
]


def _build_targets(now):
    from accounts.models import EmailVerificationToken
    from chat.models import ChatThread
    from documents.models import DataRoom, PIIReviewEvent
    from feedback.models import Feedback
    from guardrails.models import GuardrailEvent
    from llm import retention as llm_retention
    from llm.models import LLMCallLog
    from meetings.models import Meeting

    retain_filter = dict(retain_until__isnull=False, retain_until__lt=now)
    return [
        ("ChatThread", "delete", ChatThread.objects.filter(**retain_filter)),
        ("DataRoom", "delete", DataRoom.objects.filter(**retain_filter)),
        ("Meeting", "delete", Meeting.objects.filter(**retain_filter)),
        ("GuardrailEvent", "delete", GuardrailEvent.objects.filter(**retain_filter)),
        ("PIIReviewEvent", "delete", PIIReviewEvent.objects.filter(**retain_filter)),
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
            llm_retention.unredacted(
                LLMCallLog.objects.filter(created_at__lt=llm_retention.content_cutoff(now))
            ),
        ),
        (
            "LLMCallLogIdentity",
            "anonymize",
            llm_retention.identified(
                LLMCallLog.objects.filter(created_at__lt=llm_retention.identity_cutoff(now))
            ),
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
            elif action == "anonymize":
                grand_total += self._batch_update(
                    base_qs, batch_size, anonymize_llm_call_logs, "anonymized"
                )
            else:
                grand_total += self._batch_update(
                    base_qs, batch_size, redact_llm_call_logs, "redacted"
                )

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

    def _batch_update(self, base_qs, batch_size, apply, verb):
        """Run ``apply(queryset) -> rows`` over ``base_qs`` in pk batches. Each
        batch drops out of ``base_qs`` once applied, so the loop terminates and a
        rerun finds nothing."""
        total = 0
        model_cls = base_qs.model
        while True:
            ids = list(base_qs.values_list("pk", flat=True)[:batch_size])
            if not ids:
                break
            updated = apply(model_cls.objects.filter(pk__in=ids))
            total += updated
            self.stdout.write(f"    {verb} batch of {updated} (total {total})")
            if len(ids) < batch_size:
                break
        return total
