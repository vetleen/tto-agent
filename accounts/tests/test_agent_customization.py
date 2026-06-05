"""Tests for accounts.agent_customization — SOUL cascade and effective-value resolution."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.agent_customization import (
    DEFAULT_ORG_DESCRIPTION,
    DEFAULT_SOUL,
    DEFAULT_USER_NAME,
    DEFAULT_USER_TITLE,
    org_allows_user_soul,
    resolve_agent_customization,
    resolve_soul,
)
from accounts.models import Membership, Organization

User = get_user_model()


class ResolveSoulTests(TestCase):
    """The pure cascade function: personal (if allowed) -> org -> system default."""

    def test_user_soul_wins_when_allowed(self):
        self.assertEqual(
            resolve_soul("personal", "org", allow_user_soul=True), "personal"
        )

    def test_org_soul_when_user_blank(self):
        self.assertEqual(
            resolve_soul("   ", "org", allow_user_soul=True), "org"
        )

    def test_system_default_when_both_blank(self):
        self.assertEqual(
            resolve_soul("", "", allow_user_soul=True), DEFAULT_SOUL
        )

    def test_user_soul_ignored_when_not_allowed(self):
        self.assertEqual(
            resolve_soul("personal", "org", allow_user_soul=False), "org"
        )

    def test_falls_to_system_when_not_allowed_and_no_org(self):
        self.assertEqual(
            resolve_soul("personal", "", allow_user_soul=False), DEFAULT_SOUL
        )


class OrgAllowsUserSoulTests(TestCase):
    def test_none_org_defaults_permissive(self):
        self.assertTrue(org_allows_user_soul(None))

    def test_org_without_pref_defaults_permissive(self):
        org = Organization.objects.create(name="Acme", slug="acme")
        self.assertTrue(org_allows_user_soul(org))

    def test_org_pref_false(self):
        org = Organization.objects.create(
            name="Acme", slug="acme", preferences={"allow_user_soul": False}
        )
        self.assertFalse(org_allows_user_soul(org))

    def test_org_pref_true(self):
        org = Organization.objects.create(
            name="Acme", slug="acme", preferences={"allow_user_soul": True}
        )
        self.assertTrue(org_allows_user_soul(org))


class ResolveAgentCustomizationTests(TestCase):
    def _user(self, **kwargs):
        defaults = {"email": "u@example.com", "password": "x"}
        u = User.objects.create_user(email=defaults["email"], password=defaults["password"])
        for k, v in kwargs.items():
            setattr(u, k, v)
        if kwargs:
            u.save()
        return u

    # -- SOUL cascade through the model layer --

    def test_personal_soul_applied_when_org_allows(self):
        user = self._user(soul="I am terse.")
        org = Organization.objects.create(name="Org", slug="org", soul="Org voice.")
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.soul, "I am terse.")
        self.assertTrue(cust.is_user_soul_customized)

    def test_org_soul_applied_when_user_blank(self):
        user = self._user()
        org = Organization.objects.create(name="Org", slug="org", soul="Org voice.")
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.soul, "Org voice.")
        self.assertFalse(cust.is_user_soul_customized)

    def test_system_default_soul_when_all_blank(self):
        user = self._user()
        org = Organization.objects.create(name="Org", slug="org")
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.soul, DEFAULT_SOUL)

    def test_personal_soul_ignored_when_org_disallows(self):
        user = self._user(soul="I am terse.")
        org = Organization.objects.create(
            name="Org", slug="org", soul="Org voice.",
            preferences={"allow_user_soul": False},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.soul, "Org voice.")
        self.assertFalse(cust.is_user_soul_customized)
        self.assertFalse(cust.allow_user_soul)

    def test_org_soul_effective_falls_back_to_default(self):
        """org_soul (the admin editor baseline) is the system default when org has none."""
        user = self._user()
        org = Organization.objects.create(name="Org", slug="org")
        Membership.objects.create(user=user, org=org, role=Membership.Role.ADMIN)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.org_soul, DEFAULT_SOUL)

    # -- USER fallbacks --

    def test_anonymous_user_name_fallback(self):
        user = self._user()
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.user_name, DEFAULT_USER_NAME)

    def test_user_name_from_first_last(self):
        user = self._user(first_name="Jane", last_name="Doe")
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.user_name, "Jane Doe")

    def test_user_title_fallback(self):
        user = self._user()
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.user_title, DEFAULT_USER_TITLE)

    def test_user_description_raw(self):
        user = self._user(description="Licensing lead.")
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.user_description, "Licensing lead.")

    # -- ORG fallbacks + flags --

    def test_org_description_boilerplate_when_blank(self):
        user = self._user()
        org = Organization.objects.create(name="Org", slug="org")
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        cust = resolve_agent_customization(user)
        self.assertEqual(cust.org_description, DEFAULT_ORG_DESCRIPTION)

    def test_no_org_flags_and_blank_description(self):
        user = self._user()
        cust = resolve_agent_customization(user)
        self.assertFalse(cust.has_org)
        self.assertFalse(cust.is_org_admin)
        self.assertIsNone(cust.org_name)
        self.assertEqual(cust.org_description, "")
        # A user with no org may still set a personal soul (permissive default).
        self.assertTrue(cust.allow_user_soul)

    def test_admin_flag(self):
        user = self._user()
        org = Organization.objects.create(name="Org", slug="org")
        Membership.objects.create(user=user, org=org, role=Membership.Role.ADMIN)
        cust = resolve_agent_customization(user)
        self.assertTrue(cust.is_org_admin)
        self.assertTrue(cust.has_org)
        self.assertEqual(cust.org_name, "Org")
