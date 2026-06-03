"""Tests for SuspensionMiddleware app-wide suspension enforcement."""
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from accounts.models import Membership, Organization

User = get_user_model()


class SuspensionMiddlewareTests(TestCase):
    def setUp(self) -> None:
        self.org = Organization.objects.create(name="Acme", slug="acme")
        self.suspended_url = reverse("accounts:suspended")

    def _make_user(self, email, *, suspended=False, is_staff=False, membership=True):
        user = User.objects.create_user(email=email, password="pass", is_staff=is_staff)
        if membership:
            Membership.objects.create(user=user, org=self.org, is_suspended=suspended)
        return user

    def test_suspended_user_redirected_from_app(self) -> None:
        user = self._make_user("s@example.com", suspended=True)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], self.suspended_url)

    def test_suspended_user_redirected_from_settings(self) -> None:
        # Settings lives under /accounts/ but must still be gated.
        user = self._make_user("s2@example.com", suspended=True)
        self.client.force_login(user)
        resp = self.client.get(reverse("accounts:settings"))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], self.suspended_url)

    def test_active_user_not_redirected(self) -> None:
        user = self._make_user("a@example.com", suspended=False)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(resp.status_code, 302)

    def test_suspended_user_can_reach_suspended_page(self) -> None:
        user = self._make_user("s3@example.com", suspended=True)
        self.client.force_login(user)
        resp = self.client.get(self.suspended_url)
        self.assertEqual(resp.status_code, 403)

    def test_suspended_user_can_logout(self) -> None:
        user = self._make_user("s4@example.com", suspended=True)
        self.client.force_login(user)
        resp = self.client.post(reverse("accounts:logout"))
        # Not redirected to the suspended page (logout is exempt).
        self.assertNotEqual(resp.headers.get("Location"), self.suspended_url)

    def test_staff_user_not_gated(self) -> None:
        user = self._make_user("staff@example.com", suspended=True, is_staff=True)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(resp.status_code, 302)

    def test_user_without_membership_not_gated(self) -> None:
        user = self._make_user("none@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(
            resp.headers.get("Location") if resp.status_code == 302 else None,
            self.suspended_url,
        )

    def test_anonymous_redirected_to_login_not_suspended(self) -> None:
        resp = self.client.get("/chat/")
        self.assertEqual(resp.status_code, 302)
        self.assertNotEqual(resp.headers["Location"], self.suspended_url)
        self.assertIn("/accounts/login/", resp.headers["Location"])
