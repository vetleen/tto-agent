"""Tests for the Organization Usage page."""
from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from accounts.models import Membership, Organization
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
        LLMCallLog.objects.filter(pk=log.pk).update(created_at=created_at)
    return log


def _make_user(email, password="test-pass-123"):
    u = User.objects.create_user(email=email, password=password)
    u.email_verified = True
    u.save(update_fields=["email_verified"])
    return u


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgUsagePageTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        self.admin = _make_user("admin@example.com")
        self.member = _make_user("member@example.com")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_usage")

    def test_requires_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_non_admin_gets_403(self):
        self.client.login(email=self.member.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 403)

    def test_user_without_membership_gets_403(self):
        outsider = _make_user("outsider@example.com")
        self.client.login(email=outsider.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 403)

    def test_rejects_post(self):
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.post(self.url)
        self.assertEqual(response.status_code, 405)

    def test_renders_for_admin_no_data(self):
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Organization usage")
        self.assertContains(response, "Test Org")
        self.assertContains(response, "$0.00")
        self.assertContains(response, "No usage data for this period.")

    def test_totals_include_all_org_members(self):
        _create_log(self.admin, cost="0.10")
        _create_log(self.member, cost="0.05")
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "$0.15")

    def test_per_user_breakdown_shows_both_members(self):
        _create_log(self.admin, cost="0.10")
        _create_log(self.member, cost="0.05")
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.get(self.url)
        self.assertContains(response, self.admin.email)
        self.assertContains(response, self.member.email)

    def test_excludes_non_member_calls(self):
        outsider = _make_user("outsider@example.com")
        _create_log(outsider, cost="9.99")
        _create_log(self.admin, cost="0.01")
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.get(self.url)
        self.assertContains(response, "$0.01")
        self.assertNotContains(response, "$9.99")
        self.assertNotContains(response, outsider.email)

    def test_date_range_filtering(self):
        now = timezone.now()
        last_month = now - timedelta(days=40)
        _create_log(self.admin, cost="0.50", created_at=last_month)
        _create_log(self.admin, cost="0.10")
        self.client.login(email=self.admin.email, password=self.password)
        # Default view: current month only
        response = self.client.get(self.url)
        self.assertContains(response, "$0.10")
        self.assertNotContains(response, "$0.50")

    def test_custom_date_range(self):
        now = timezone.now()
        last_month = now - timedelta(days=40)
        _create_log(self.admin, cost="0.50", created_at=last_month)
        _create_log(self.admin, cost="0.10")
        start = (now - timedelta(days=60)).date().isoformat()
        end = (now + timedelta(days=1)).date().isoformat()
        self.client.login(email=self.admin.email, password=self.password)
        response = self.client.get(self.url, {"start": start, "end": end})
        self.assertContains(response, "$0.60")
