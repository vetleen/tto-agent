"""Tests for the redact_old_llm_logs management command."""

from datetime import timedelta
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from llm.models import LLMCallLog

User = get_user_model()


def _make_log(created_at, prompt=None, raw_output="body", user=None, tools=None):
    log = LLMCallLog.objects.create(
        user=user,
        model="gpt-5-mini",
        prompt=prompt if prompt is not None else [{"role": "user", "content": "hi"}],
        raw_output=raw_output,
        tools=tools if tools is not None else [{"name": "tool_a"}],
    )
    # Bypass auto_now_add to backdate created_at for the tests.
    LLMCallLog.objects.filter(pk=log.pk).update(created_at=created_at)
    log.refresh_from_db()
    return log


class RedactOldLLMLogsCommandTests(TestCase):
    def test_only_old_rows_redacted(self):
        now = timezone.now()
        old = _make_log(created_at=now - timedelta(days=91))
        recent = _make_log(created_at=now - timedelta(days=1))

        call_command("redact_old_llm_logs", stdout=StringIO())

        old.refresh_from_db()
        recent.refresh_from_db()
        self.assertEqual(old.prompt, {"redacted": True})
        self.assertEqual(old.raw_output, "")
        self.assertIsNone(old.tools)
        self.assertEqual(recent.prompt, [{"role": "user", "content": "hi"}])
        self.assertEqual(recent.raw_output, "body")

    def test_custom_days_argument(self):
        now = timezone.now()
        mid = _make_log(created_at=now - timedelta(days=45))

        call_command("redact_old_llm_logs", "--days", "30", stdout=StringIO())

        mid.refresh_from_db()
        self.assertEqual(mid.prompt, {"redacted": True})

    def test_dry_run_does_not_modify(self):
        now = timezone.now()
        old = _make_log(created_at=now - timedelta(days=91))

        out = StringIO()
        call_command("redact_old_llm_logs", "--dry-run", stdout=out)

        old.refresh_from_db()
        self.assertEqual(old.prompt, [{"role": "user", "content": "hi"}])
        self.assertEqual(old.raw_output, "body")
        self.assertIn("dry-run", out.getvalue())

    def test_idempotent(self):
        now = timezone.now()
        _make_log(created_at=now - timedelta(days=91))

        call_command("redact_old_llm_logs", stdout=StringIO())
        out = StringIO()
        call_command("redact_old_llm_logs", stdout=out)

        self.assertIn("No rows to redact", out.getvalue())

    def test_empty_table_prints_nothing_to_do(self):
        out = StringIO()
        call_command("redact_old_llm_logs", stdout=out)
        self.assertIn("No rows to redact", out.getvalue())

    def test_batch_size_processes_all_rows(self):
        now = timezone.now()
        for _ in range(5):
            _make_log(created_at=now - timedelta(days=91))

        call_command(
            "redact_old_llm_logs", "--batch-size", "2", stdout=StringIO()
        )

        qs = LLMCallLog.objects.all()
        for log in qs:
            self.assertEqual(log.prompt, {"redacted": True})
            self.assertEqual(log.raw_output, "")
