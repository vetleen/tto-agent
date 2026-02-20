"""Tests for Organization, Scope, and Membership models."""
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Membership, Organization, Scope

User = get_user_model()


class OrganizationModelTests(TestCase):
    def test_create_organization(self) -> None:
        org = Organization.objects.create(name="Acme Corp", slug="acme-corp")
        self.assertEqual(org.name, "Acme Corp")
        self.assertEqual(org.slug, "acme-corp")
        self.assertIsNotNone(org.pk)

    def test_organization_str(self) -> None:
        org = Organization.objects.create(name="Acme Corp", slug="acme-corp")
        self.assertEqual(str(org), "Acme Corp")

    def test_organization_slug_unique(self) -> None:
        Organization.objects.create(name="Acme", slug="acme")
        with self.assertRaises(IntegrityError):
            Organization.objects.create(name="Acme Other", slug="acme")

    def test_organization_ordering(self) -> None:
        Organization.objects.create(name="Zebra", slug="zebra")
        Organization.objects.create(name="Alpha", slug="alpha")
        names = [o.name for o in Organization.objects.all()]
        self.assertEqual(names, ["Alpha", "Zebra"])


class ScopeModelTests(TestCase):
    def test_create_scope(self) -> None:
        scope = Scope.objects.create(code="billing", name="Billing")
        self.assertEqual(scope.code, "billing")
        self.assertEqual(scope.name, "Billing")
        self.assertIsNotNone(scope.pk)

    def test_scope_str_uses_name(self) -> None:
        scope = Scope.objects.create(code="billing", name="Billing")
        self.assertEqual(str(scope), "Billing")

    def test_scope_str_fallback_to_code(self) -> None:
        scope = Scope.objects.create(code="billing", name="")
        self.assertEqual(str(scope), "billing")

    def test_scope_code_unique(self) -> None:
        Scope.objects.create(code="billing", name="Billing")
        with self.assertRaises(IntegrityError):
            Scope.objects.create(code="billing", name="Other")

    def test_scope_ordering(self) -> None:
        Scope.objects.create(code="zebra", name="Z")
        Scope.objects.create(code="alpha", name="A")
        codes = [s.code for s in Scope.objects.all()]
        self.assertEqual(codes, ["alpha", "zebra"])


class MembershipModelTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(email="user@example.com", password="pass")
        self.org = Organization.objects.create(name="Acme", slug="acme")

    def test_create_membership_default_role(self) -> None:
        m = Membership.objects.create(user=self.user, org=self.org)
        self.assertEqual(m.role, Membership.Role.MEMBER)
        self.assertIsNotNone(m.pk)

    def test_create_membership_with_role(self) -> None:
        m = Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN
        )
        self.assertEqual(m.role, Membership.Role.ADMIN)

    def test_membership_str(self) -> None:
        m = Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.VIEWER
        )
        self.assertIn("user@example.com", str(m))
        self.assertIn("Acme", str(m))
        self.assertIn("viewer", str(m))

    def test_unique_user_org(self) -> None:
        Membership.objects.create(user=self.user, org=self.org)
        with self.assertRaises(IntegrityError):
            Membership.objects.create(user=self.user, org=self.org)

    def test_user_organization_memberships_reverse(self) -> None:
        m = Membership.objects.create(user=self.user, org=self.org)
        self.assertEqual(list(self.user.organization_memberships.all()), [m])

    def test_org_memberships_reverse(self) -> None:
        m = Membership.objects.create(user=self.user, org=self.org)
        self.assertEqual(list(self.org.memberships.all()), [m])

    def test_membership_with_scopes(self) -> None:
        billing = Scope.objects.create(code="billing", name="Billing")
        settings = Scope.objects.create(code="settings", name="Settings")
        m = Membership.objects.create(user=self.user, org=self.org)
        m.scopes.add(billing, settings)
        self.assertEqual(set(m.scopes.values_list("code", flat=True)), {"billing", "settings"})

    def test_membership_scopes_optional(self) -> None:
        m = Membership.objects.create(user=self.user, org=self.org)
        self.assertEqual(m.scopes.count(), 0)

    def test_role_choices(self) -> None:
        self.assertEqual(Membership.Role.ADMIN, "admin")
        self.assertEqual(Membership.Role.MEMBER, "member")
        self.assertEqual(Membership.Role.VIEWER, "viewer")
