"""A sub-agent run reuses one User instance across the tool user-resolution
chokepoints, so per-instance memoization (accounts.models.get_membership /
get_user_preferences_dict) holds for the whole run.

WILFRED-7H (UserSettings) and WILFRED-7W (Membership/Org) were N+1s in
run_subagent_task: every tool loaded a fresh User and re-read both tables.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db import connection
from django.test import TestCase
from django.test.utils import CaptureQueriesContext

from llm.types.context import RunContext

User = get_user_model()


class RunUserCacheTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="rc@example.com", password="x")

    def _ctx(self, cached=None):
        ctx = RunContext.create(user_id=self.user.pk)
        if cached is not None:
            ctx._cached_user = cached
        return ctx

    def test_tool_loops_get_user_prefers_cache(self):
        from chat.tool_loops import _get_user

        ctx = self._ctx(cached=self.user)
        with CaptureQueriesContext(connection) as cap:
            user, err = _get_user(ctx)
        self.assertIsNone(err)
        self.assertIs(user, self.user)
        self.assertEqual(len(cap), 0, "cached user must not hit the DB")

    def test_tool_loops_get_user_falls_back_without_cache(self):
        """The main pipeline never seeds the cache — behaviour is unchanged."""
        from chat.tool_loops import _get_user

        user, err = _get_user(self._ctx())
        self.assertIsNone(err)
        self.assertEqual(user.pk, self.user.pk)

    def test_image_tools_resolve_user_prefers_cache(self):
        from chat.image_tools import _resolve_user

        ctx = self._ctx(cached=self.user)
        with CaptureQueriesContext(connection) as cap:
            user = _resolve_user(ctx)
        self.assertIs(user, self.user)
        self.assertEqual(len(cap), 0)

    def test_tools_get_user_ctx_prefers_cache(self):
        from chat.tools import _get_user_ctx

        ctx = self._ctx(cached=self.user)
        with CaptureQueriesContext(connection) as cap:
            user = _get_user_ctx(ctx)
        self.assertIs(user, self.user)
        self.assertEqual(len(cap), 0)

    def test_membership_reads_memoized_on_shared_instance(self):
        """WILFRED-7W: get_accessible_skills / _org_disabled_info route through the
        memoized get_membership, so repeated calls on one User instance don't re-query
        accounts_membership."""
        from accounts.models import Membership, Organization, get_membership
        from agent_skills.services import get_accessible_skills

        org = Organization.objects.create(name="RCOrg", slug="rc-org")
        Membership.objects.create(user=self.user, org=org, role="admin")

        get_membership(self.user)  # prime the instance cache, as run setup does
        with CaptureQueriesContext(connection) as cap:
            get_accessible_skills(self.user)
            get_accessible_skills(self.user)
        membership_selects = [
            q for q in cap.captured_queries if "accounts_membership" in q["sql"].lower()
        ]
        self.assertEqual(membership_selects, [], "membership must be served from the cache")
