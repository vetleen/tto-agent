"""Tests for chat views."""

import json
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from chat.models import ChatThread
from llm.models import LLMCallLog

User = get_user_model()


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929", "openai/gpt-5-mini"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
)
class ChatHomeModelChoicesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def test_context_includes_model_choices_json(self):
        response = self.client.get(reverse("chat_home"))
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_choices_json", response.context)
        choices = json.loads(response.context["model_choices_json"])
        self.assertIsInstance(choices, list)
        self.assertTrue(len(choices) > 0)
        # Each choice has the required keys
        for c in choices:
            self.assertIn("id", c)
            self.assertIn("display_name", c)
            self.assertIn("supports_thinking", c)

    def test_context_includes_default_model(self):
        response = self.client.get(reverse("chat_home"))
        self.assertIn("default_model", response.context)
        self.assertTrue(len(response.context["default_model"]) > 0)

    def test_context_includes_default_model_display(self):
        response = self.client.get(reverse("chat_home"))
        self.assertIn("default_model_display", response.context)
        self.assertTrue(len(response.context["default_model_display"]) > 0)

    def test_model_selector_rendered_in_html(self):
        response = self.client.get(reverse("chat_home"))
        self.assertContains(response, 'id="model-selector-btn"')
        self.assertContains(response, 'id="model-selector-dropdown"')
        self.assertContains(response, 'name="thinking-level"')


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"],
    LLM_DEFAULT_MODEL="anthropic/claude-sonnet-4-5-20250929",
)
class ThreadCostTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="cost@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)

    def test_no_thread_returns_zero_cost(self):
        response = self.client.get(reverse("chat_home"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["thread_cost_usd"], 0.0)

    def test_thread_with_call_logs_returns_correct_cost(self):
        thread = ChatThread.objects.create(created_by=self.user)
        LLMCallLog.objects.create(
            model="test-model",
            prompt=[{"role": "user", "content": "Hi"}],
            raw_output="Hello!",
            status=LLMCallLog.Status.SUCCESS,
            conversation_id=str(thread.id),
            cost_usd=Decimal("0.00123456"),
        )
        LLMCallLog.objects.create(
            model="test-model",
            prompt=[{"role": "user", "content": "Bye"}],
            raw_output="Goodbye!",
            status=LLMCallLog.Status.SUCCESS,
            conversation_id=str(thread.id),
            cost_usd=Decimal("0.00200000"),
        )
        response = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.context["thread_cost_usd"], 0.00323456, places=6)

    def test_thread_with_no_logs_returns_zero_cost(self):
        thread = ChatThread.objects.create(created_by=self.user)
        response = self.client.get(reverse("chat_home") + f"?thread={thread.id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["thread_cost_usd"], 0.0)
