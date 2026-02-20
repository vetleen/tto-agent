"""Tests for LLMCallLog model."""
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase

from llm_service.models import LLMCallLog

User = get_user_model()


class LLMCallLogModelTestCase(TestCase):
    """Test LLMCallLog creation and fields."""

    def test_create_minimal_log(self):
        log = LLMCallLog.objects.create(
            model="openai/gpt-4o",
            is_stream=False,
        )
        self.assertIsNotNone(log.id)
        self.assertEqual(log.model, "openai/gpt-4o")
        self.assertFalse(log.is_stream)
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)
        self.assertEqual(log.input_tokens, 0)
        self.assertEqual(log.output_tokens, 0)
        self.assertIsNone(log.cost_usd)
        self.assertIsNone(log.user_id)

    def test_create_full_log(self):
        user = User.objects.create_user(email="u@t.com", password="pw")
        log = LLMCallLog.objects.create(
            model="openai/gpt-4o",
            is_stream=True,
            user=user,
            metadata={"feature": "chat", "request_id": "req-1"},
            request_id="req-1",
            duration_ms=1500,
            prompt_preview="Hello",
            response_preview="Hi there",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=Decimal("0.001"),
            cost_source="litellm",
            status=LLMCallLog.Status.SUCCESS,
        )
        self.assertEqual(log.user, user)
        self.assertEqual(log.metadata["feature"], "chat")
        self.assertEqual(log.cost_usd, Decimal("0.001"))
        self.assertEqual(log.duration_ms, 1500)

    def test_status_choices(self):
        log = LLMCallLog.objects.create(model="m", status=LLMCallLog.Status.ERROR, error_message="fail")
        self.assertEqual(log.status, LLMCallLog.Status.ERROR)
        self.assertEqual(log.error_message, "fail")

    def test_logging_failed_status(self):
        log = LLMCallLog.objects.create(
            model="openai/gpt-4o",
            is_stream=False,
            status=LLMCallLog.Status.LOGGING_FAILED,
            error_message="log write failed",
        )
        self.assertEqual(log.status, LLMCallLog.Status.LOGGING_FAILED)
