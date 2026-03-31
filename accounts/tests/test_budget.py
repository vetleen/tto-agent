"""Tests for org budget settings views and context processor."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgBudgetUpdateViewTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="admin@test.com", password=self.password
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])

        self.member_user = User.objects.create_user(
            email="member@test.com", password=self.password
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])

        self.org = Organization.objects.create(name="Test Org", slug="test-org")
        Membership.objects.create(
            user=self.admin_user, org=self.org, role=Membership.Role.ADMIN
        )
        Membership.objects.create(
            user=self.member_user, org=self.org, role=Membership.Role.MEMBER
        )
        self.url = reverse("accounts:org_budget_update")

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": 50}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)

    def test_requires_admin(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": 50}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_saves_user_budget(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": 50}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["monthly_budget_per_user"], 50.0)

    def test_saves_org_budget(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_org": 200}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["monthly_budget_org"], 200.0)

    def test_saves_both_budgets(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": 25, "monthly_budget_org": 100}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["monthly_budget_per_user"], 25.0)
        self.assertEqual(self.org.preferences["monthly_budget_org"], 100.0)

    def test_rejects_negative(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": -10}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_rejects_invalid_value(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": "not-a-number"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_zero_is_valid(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"monthly_budget_per_user": 0}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["monthly_budget_per_user"], 0.0)


@override_settings(ALLOWED_HOSTS=["testserver"])
class BudgetContextProcessorTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="ctx@test.com", password=self.password
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Ctx Org", slug="ctx-org")
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.MEMBER
        )

    def test_no_budget_gives_none(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(reverse("accounts:settings"))
        self.assertIsNone(response.context.get("budget_status"))

    def test_budget_set_gives_status(self):
        self.org.preferences = {"monthly_budget_per_user": 50}
        self.org.save()
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(reverse("accounts:settings"))
        status = response.context.get("budget_status")
        self.assertIsNotNone(status)
        self.assertFalse(status["exceeded"])
        self.assertEqual(status["user_budget"], 50)
