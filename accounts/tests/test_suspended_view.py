"""Tests for the suspended page view."""
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from accounts.models import Membership, Organization

User = get_user_model()


class SuspendedViewTests(TestCase):
    def setUp(self) -> None:
        self.url = reverse("accounts:suspended")

    def test_renders_org_name(self) -> None:
        org = Organization.objects.create(name="Acme", slug="acme")
        user = User.objects.create_user(email="s@example.com", password="pass")
        Membership.objects.create(user=user, org=org, is_suspended=True)
        self.client.force_login(user)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
        self.assertContains(resp, "Acme", status_code=403)
        self.assertContains(resp, "has been suspended", status_code=403)

    def test_no_org_fallback_text(self) -> None:
        user = User.objects.create_user(email="none@example.com", password="pass")
        self.client.force_login(user)
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 403)
        self.assertContains(resp, "Please contact your administrator", status_code=403)

    def test_requires_login(self) -> None:
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/accounts/logged-out/", resp.headers["Location"])
