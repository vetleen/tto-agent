"""Tests for the Usage billing page."""
from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from llm.models import LLMCallLog

User = get_user_model()


def _create_log(user, model="gpt-4o", cost="0.0050", input_tokens=100, output_tokens=50, created_at=None):
    log = LLMCallLog.objects.create(
        user=user,
        model=model,
        cost_usd=Decimal(cost),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt=[{"role": "user", "content": "test"}],
    )
    if created_at:
        # Override auto_now_add
        LLMCallLog.objects.filter(pk=log.pk).update(created_at=created_at)
    return log


@override_settings(ALLOWED_HOSTS=["testserver"])
class UsagePageTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="usage@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:usage")

    def test_requires_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_rejects_post(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 405)

    def test_renders_usage_page(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Usage")
        self.assertContains(response, "Total cost")

    def test_no_data_message(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertContains(response, "No usage data")

    def test_total_cost(self):
        self.client.login(email=self.user.email, password=self.password)
        _create_log(self.user, cost="0.0050", input_tokens=100, output_tokens=50)
        _create_log(self.user, cost="0.0030", input_tokens=200, output_tokens=80)
        response = self.client.get(self.url)
        self.assertContains(response, "$0.0080")
        self.assertContains(response, "2")  # call count

    def test_model_breakdown(self):
        self.client.login(email=self.user.email, password=self.password)
        _create_log(self.user, model="gpt-4o", cost="0.0050")
        _create_log(self.user, model="claude-3-sonnet", cost="0.0030")
        response = self.client.get(self.url)
        self.assertContains(response, "gpt-4o")
        self.assertContains(response, "claude-3-sonnet")

    def test_custom_date_range(self):
        self.client.login(email=self.user.email, password=self.password)
        old_date = timezone.now() - timedelta(days=60)
        _create_log(self.user, model="old-model", cost="0.0100", created_at=old_date)
        _create_log(self.user, model="new-model", cost="0.0020")

        # Current month should only show the recent entry
        response = self.client.get(self.url)
        self.assertContains(response, "new-model")
        self.assertNotContains(response, "old-model")

        # Custom range covering 90 days should show both
        start = (timezone.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end = timezone.now().strftime("%Y-%m-%d")
        response = self.client.get(self.url, {"start": start, "end": end})
        self.assertContains(response, "old-model")
        self.assertContains(response, "new-model")

    def test_other_users_excluded(self):
        self.client.login(email=self.user.email, password=self.password)
        other = User.objects.create_user(email="other@example.com", password="pass123")
        _create_log(other, model="other-model", cost="1.0000")
        _create_log(self.user, model="my-model", cost="0.0010")
        response = self.client.get(self.url)
        self.assertNotContains(response, "other-model")
        self.assertContains(response, "my-model")

    def test_invalid_dates_fallback(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url, {"start": "not-a-date", "end": "also-bad"})
        self.assertEqual(response.status_code, 200)
        # Should fall back to current month display
        self.assertContains(response, "Total cost")

    def test_month_navigation(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        # Should have a prev month link
        self.assertContains(response, "&larr;")
