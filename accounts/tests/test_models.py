"""Tests for Organization, Scope, and Membership models."""
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
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

    def test_organization_description_defaults_blank(self) -> None:
        org = Organization.objects.create(name="Acme", slug="acme2")
        self.assertEqual(org.description, "")

    def test_organization_description_saved(self) -> None:
        org = Organization.objects.create(name="Acme", slug="acme3", description="A biotech TTO.")
        org.refresh_from_db()
        self.assertEqual(org.description, "A biotech TTO.")


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

    def test_unique_membership_per_user(self) -> None:
        Membership.objects.create(user=self.user, org=self.org)
        other = Organization.objects.create(name="Other", slug="other")
        with self.assertRaises(IntegrityError):
            Membership.objects.create(user=self.user, org=other)

    def test_validate_unique_rejects_second_org(self) -> None:
        Membership.objects.create(user=self.user, org=self.org)
        other = Organization.objects.create(name="Other", slug="other")
        with self.assertRaises(ValidationError) as ctx:
            Membership(user=self.user, org=other).full_clean()
        self.assertIn("user", ctx.exception.message_dict)

    def test_get_user_org(self) -> None:
        from accounts.models import get_user_org, invalidate_membership_cache

        self.assertIsNone(get_user_org(self.user))
        Membership.objects.create(user=self.user, org=self.org)
        # get_user_org memoizes the membership on the user instance; after a
        # same-instance membership change the cache must be invalidated.
        invalidate_membership_cache(self.user)
        self.assertEqual(get_user_org(self.user), self.org)

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


class UserProfileFieldTests(TestCase):
    def test_profile_fields_default_blank(self) -> None:
        user = User.objects.create_user(email="u@example.com", password="pw")
        self.assertEqual(user.first_name, "")
        self.assertEqual(user.last_name, "")
        self.assertEqual(user.title, "")
        self.assertEqual(user.description, "")

    def test_profile_fields_saved(self) -> None:
        user = User.objects.create_user(email="u2@example.com", password="pw")
        user.first_name = "Alice"
        user.last_name = "Smith"
        user.title = "Patent Attorney"
        user.description = "Specializes in biotech IP."
        user.save()
        user.refresh_from_db()
        self.assertEqual(user.first_name, "Alice")
        self.assertEqual(user.last_name, "Smith")
        self.assertEqual(user.title, "Patent Attorney")
        self.assertEqual(user.description, "Specializes in biotech IP.")


class MembershipSuspensionLifecycleTests(TestCase):
    """Membership.save() keeps the suspension bookkeeping consistent for
    every writer (guardrails, admin, shell)."""

    def setUp(self) -> None:
        self.user = User.objects.create_user(email="susp@example.com", password="pass")
        self.org = Organization.objects.create(name="SuspOrg", slug="susporg")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def test_suspending_stamps_timestamp(self) -> None:
        self.membership.is_suspended = True
        self.membership.save()
        self.membership.refresh_from_db()
        self.assertTrue(self.membership.is_suspended)
        self.assertIsNotNone(self.membership.suspended_at)

    def test_explicit_timestamp_is_preserved(self) -> None:
        from django.utils import timezone
        explicit = timezone.now() - timezone.timedelta(days=3)
        self.membership.is_suspended = True
        self.membership.suspended_at = explicit
        self.membership.save()
        self.membership.refresh_from_db()
        self.assertEqual(self.membership.suspended_at, explicit)

    def test_unsuspending_clears_timestamp_and_reason(self) -> None:
        self.membership.suspend("policy violation")
        self.membership.refresh_from_db()
        self.assertIsNotNone(self.membership.suspended_at)
        self.assertEqual(self.membership.suspended_reason, "policy violation")

        self.membership.unsuspend()
        self.membership.refresh_from_db()
        self.assertFalse(self.membership.is_suspended)
        self.assertIsNone(self.membership.suspended_at)
        self.assertEqual(self.membership.suspended_reason, "")

    def test_save_with_update_fields_persists_bookkeeping(self) -> None:
        # update_fields without the lifecycle fields still persists the stamp.
        self.membership.is_suspended = True
        self.membership.save(update_fields=["is_suspended"])
        self.membership.refresh_from_db()
        self.assertIsNotNone(self.membership.suspended_at)

    def test_suspend_truncates_reason(self) -> None:
        self.membership.suspend("x" * 5000)
        self.membership.refresh_from_db()
        self.assertEqual(len(self.membership.suspended_reason), 2000)


class MembershipMemoizationTests(TestCase):
    """get_membership memoizes on the user instance (request-scoped in HTTP;
    consumers invalidate explicitly)."""

    def setUp(self) -> None:
        self.user = User.objects.create_user(email="memo@example.com", password="pass")
        self.org = Organization.objects.create(name="MemoOrg", slug="memoorg")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def test_second_call_hits_cache(self) -> None:
        from accounts.models import get_membership

        with self.assertNumQueries(1):
            first = get_membership(self.user)
            second = get_membership(self.user)
        self.assertEqual(first.pk, self.membership.pk)
        self.assertIs(first, second)

    def test_none_result_is_cached(self) -> None:
        from accounts.models import get_membership

        loner = User.objects.create_user(email="loner@example.com", password="pass")
        with self.assertNumQueries(1):
            self.assertIsNone(get_membership(loner))
            self.assertIsNone(get_membership(loner))

    def test_invalidate_forces_requery(self) -> None:
        from accounts.models import get_membership, invalidate_membership_cache

        get_membership(self.user)
        self.membership.role = Membership.Role.ADMIN
        self.membership.save(update_fields=["role"])

        invalidate_membership_cache(self.user)
        self.assertEqual(get_membership(self.user).role, Membership.Role.ADMIN)

    def test_anonymous_returns_none(self) -> None:
        from django.contrib.auth.models import AnonymousUser

        from accounts.models import get_membership

        self.assertIsNone(get_membership(AnonymousUser()))
        self.assertIsNone(get_membership(None))
