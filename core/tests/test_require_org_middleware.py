"""Tests for RequireOrgMiddleware no-org gating."""
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization

User = get_user_model()


@override_settings(REQUIRE_ORG_MEMBERSHIP=True)
class RequireOrgMiddlewareTests(TestCase):
    """The gate is disabled by default under the test runner; opt back in here."""

    def setUp(self) -> None:
        self.org = Organization.objects.create(name="Acme", slug="acme")
        self.no_org_url = reverse("accounts:no_org")

    def _make_user(self, email, *, is_staff=False, membership=True):
        user = User.objects.create_user(email=email, password="pass", is_staff=is_staff)
        if membership:
            Membership.objects.create(user=user, org=self.org)
        return user

    def test_user_without_org_redirected_from_app(self) -> None:
        user = self._make_user("noorg@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], self.no_org_url)

    def test_user_without_org_redirected_from_settings(self) -> None:
        # Settings lives under /accounts/ but must still be gated.
        user = self._make_user("noorg2@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.get(reverse("accounts:settings"))
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.headers["Location"], self.no_org_url)

    def test_user_with_org_not_gated(self) -> None:
        user = self._make_user("hasorg@example.com", membership=True)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(
            resp.headers.get("Location") if resp.status_code == 302 else None,
            self.no_org_url,
        )

    def test_no_org_user_can_reach_no_org_page(self) -> None:
        user = self._make_user("noorg3@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.get(self.no_org_url)
        self.assertEqual(resp.status_code, 403)

    def test_no_org_user_can_logout(self) -> None:
        user = self._make_user("noorg4@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.post(reverse("accounts:logout"))
        self.assertNotEqual(resp.headers.get("Location"), self.no_org_url)

    def test_staff_user_not_gated(self) -> None:
        user = self._make_user("staff@example.com", membership=False, is_staff=True)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(
            resp.headers.get("Location") if resp.status_code == 302 else None,
            self.no_org_url,
        )

    def test_anonymous_redirected_to_login_not_no_org(self) -> None:
        # Anonymous users skip the gate (handled by the normal auth/login flow).
        resp = self.client.get("/chat/")
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("/accounts/no-org/", resp.headers["Location"])

    @override_settings(REQUIRE_ORG_MEMBERSHIP=False)
    def test_gate_disabled_lets_org_less_user_through(self) -> None:
        user = self._make_user("noorg5@example.com", membership=False)
        self.client.force_login(user)
        resp = self.client.get("/chat/")
        self.assertNotEqual(
            resp.headers.get("Location") if resp.status_code == 302 else None,
            self.no_org_url,
        )
