"""Tests for the LLMCallLog model."""

from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm.models import LLMCallLog

User = get_user_model()


class LLMCallLogModelTests(TestCase):
    """Basic CRUD and field behaviour for LLMCallLog."""

    def test_create_minimal_success_entry(self):
        log = LLMCallLog.objects.create(
            model="gpt-4o-mini",
            prompt=[{"role": "user", "content": "Hi"}],
            raw_output="Hello!",
            status=LLMCallLog.Status.SUCCESS,
        )
        log.refresh_from_db()
        self.assertEqual(log.model, "gpt-4o-mini")
        self.assertEqual(log.status, "success")
        self.assertFalse(log.is_stream)
        self.assertIsNotNone(log.id)
        self.assertIsNotNone(log.created_at)

    def test_create_full_entry_with_usage(self):
        user = User.objects.create_user(email="test@example.com", password="pass")
        log = LLMCallLog.objects.create(
            user=user,
            run_id="abc-123",
            model="claude-sonnet-4-20250514",
            is_stream=False,
            prompt=[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}],
            raw_output="Hello there!",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=Decimal("0.00012345"),
            duration_ms=200,
            status=LLMCallLog.Status.SUCCESS,
        )
        log.refresh_from_db()
        self.assertEqual(log.user, user)
        self.assertEqual(log.run_id, "abc-123")
        self.assertEqual(log.input_tokens, 10)
        self.assertEqual(log.output_tokens, 5)
        self.assertEqual(log.total_tokens, 15)
        self.assertEqual(log.cost_usd, Decimal("0.00012345"))
        self.assertEqual(log.duration_ms, 200)

    def test_create_error_entry(self):
        log = LLMCallLog.objects.create(
            model="gpt-4o-mini",
            prompt=[{"role": "user", "content": "Hi"}],
            raw_output="",
            status=LLMCallLog.Status.ERROR,
            error_type="TimeoutError",
            error_message="Request timed out after 30s",
            duration_ms=30000,
        )
        log.refresh_from_db()
        self.assertEqual(log.status, "error")
        self.assertEqual(log.error_type, "TimeoutError")
        self.assertEqual(log.error_message, "Request timed out after 30s")

    def test_str_representation(self):
        log = LLMCallLog.objects.create(
            model="gpt-4o-mini",
            prompt=[],
            raw_output="",
        )
        s = str(log)
        self.assertIn("gpt-4o-mini", s)
        self.assertIn("success", s)

    def test_ordering_is_newest_first(self):
        log1 = LLMCallLog.objects.create(model="a", prompt=[], raw_output="")
        log2 = LLMCallLog.objects.create(model="b", prompt=[], raw_output="")
        logs = list(LLMCallLog.objects.all())
        self.assertEqual(logs[0].pk, log2.pk)
        self.assertEqual(logs[1].pk, log1.pk)

    def test_user_set_null_on_delete(self):
        user = User.objects.create_user(email="del@example.com", password="pass")
        log = LLMCallLog.objects.create(
            user=user,
            model="gpt-4o-mini",
            prompt=[],
            raw_output="",
        )
        user.delete()
        log.refresh_from_db()
        self.assertIsNone(log.user)

    def test_status_choices(self):
        self.assertEqual(LLMCallLog.Status.SUCCESS, "success")
        self.assertEqual(LLMCallLog.Status.ERROR, "error")
