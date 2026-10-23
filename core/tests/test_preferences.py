"""Unit tests for core.preferences cascading resolution."""

from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, override_settings

from accounts.models import Membership, Organization, UserSettings
from unittest.mock import MagicMock

from core.preferences import DEFAULT_MAX_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS, ResolvedPreferences, get_preferences


def _create_user(email="test@example.com", password="testpass123"):
    from django.contrib.auth import get_user_model
    User = get_user_model()
    return User.objects.create_user(email=email, password=password)


class NoOrgPreferencesTest(TestCase):
    """User with no org membership falls back to system defaults."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.5-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_system_defaults(self, mock_registry, mock_allowed):
        # Document tools are skills-gated (section="skills") for the main agent —
        # behind the data_room_tools skill — but always-on for sub-agents via
        # subagent_section="chat".
        doc_tool = SimpleNamespace(
            section="skills", subagent_section="chat", audience="shared",
        )
        mock_registry.return_value.list_tools.return_value = {
            "document_search": doc_tool, "document_read": doc_tool,
        }
        user = _create_user()
        prefs = get_preferences(user)

        self.assertEqual(prefs.top_model, "openai/gpt-5.4")
        self.assertEqual(prefs.mid_model, "gemini/gemini-3.5-flash")
        self.assertEqual(prefs.cheap_model, "openai/gpt-5.4-nano")
        self.assertEqual(prefs.allowed_models, [
            "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
        ])
        self.assertNotIn("document_search", prefs.allowed_tools)
        self.assertIn("document_search", prefs.allowed_subagent_tools)
        self.assertIn("document_read", prefs.allowed_subagent_tools)


class OrgRestrictsModelsTest(TestCase):
    """Org allowed_models restricts the effective allowed list."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "anthropic/claude-sonnet-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_allowed_restricts(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.allowed_models, [
            "openai/gpt-5.4", "anthropic/claude-sonnet-4-6",
        ])


class UserModelOverrideIgnoredTest(TestCase):
    """Stored user model choices do not override organization/system defaults."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_stored_user_choice_is_ignored(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "models": {"primary": "anthropic/claude-sonnet-4-6"},
        }
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.top_model, "openai/gpt-5.4")


class UserPicksOutsideAllowedTest(TestCase):
    """User picks a model not in org's allowed list; falls back."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_outside_allowed_falls_back(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5.4"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "models": {"primary": "anthropic/claude-sonnet-4-6"},
        }
        settings.save()

        prefs = get_preferences(user)
        # Falls back to system default (gpt-5) since user's choice is not in org allowed
        self.assertEqual(prefs.top_model, "openai/gpt-5.4")


class OrgDisablesToolTest(TestCase):
    """Org disabling a tool removes it from allowed_tools."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_disables_tool(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {
            "document_search": None,
            "document_read": None,
        }

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "tools": {"document_read": False},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertIn("document_search", prefs.allowed_tools)
        self.assertNotIn("document_read", prefs.allowed_tools)


class ThemeFromPreferencesTest(TestCase):
    """Theme is read from the preferences JSON."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_theme_from_preferences(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"theme": "dark"}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.theme, "dark")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_theme_defaults_to_system(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        prefs = get_preferences(user)
        self.assertEqual(prefs.theme, "system")


class OrgDefaultModelTest(TestCase):
    """Org can set a default model per tier."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "anthropic/claude-sonnet-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_default_model(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "models": {"primary": "anthropic/claude-sonnet-4-6"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.top_model, "anthropic/claude-sonnet-4-6")


class OrgDefaultRemovedFromAllowedTest(TestCase):
    """Org default model that was removed from allowed list must not be used."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "anthropic/claude-sonnet-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_default_removed_from_allowed_falls_back(self, mock_registry, mock_allowed):
        """If org sets a default but then removes it from allowed_models, fall back."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            # Default is claude, but allowed list only has gpt-5
            "allowed_models": ["openai/gpt-5.4"],
            "models": {"primary": "anthropic/claude-sonnet-4-6"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        # Must NOT resolve to claude since it's not in the org's allowed list
        self.assertNotEqual(prefs.top_model, "anthropic/claude-sonnet-4-6")
        self.assertEqual(prefs.top_model, "openai/gpt-5.4")


class OrgAllowedModelsNoOverlapTest(TestCase):
    """When org's allowed_models has no overlap with system, must not fall back to system default."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_no_overlap_returns_empty_model(self, mock_registry, mock_allowed):
        """Org sets allowed_models with models not in system list — tier should be empty, not system default."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="no-overlap@example.com")
        org = Organization.objects.create(name="NoOverlap", slug="no-overlap", preferences={
            "allowed_models": ["nonexistent/model-a", "nonexistent/model-b"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        # effective_allowed is empty since none of org's models are in system list
        self.assertEqual(prefs.allowed_models, [])
        # Must NOT fall back to system default — that bypasses org restrictions
        self.assertEqual(prefs.top_model, "")
        self.assertEqual(prefs.mid_model, "")
        self.assertEqual(prefs.cheap_model, "")


class SectionAwareToolFilteringTest(TestCase):
    """Processing tools (document_processing section) are excluded from chat allowed_tools."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_processing_tools_excluded_from_allowed_tools(self, mock_registry, mock_allowed):
        chat_tool = MagicMock(section="chat", audience="shared")
        proc_tool = MagicMock(section="document_processing", audience="shared")

        mock_registry.return_value.list_tools.return_value = {
            "document_search": chat_tool,
            "normalize_document": proc_tool,
        }

        user = _create_user(email="section@example.com")
        prefs = get_preferences(user)

        self.assertIn("document_search", prefs.allowed_tools)
        self.assertNotIn("normalize_document", prefs.allowed_tools)


class SubagentSectionOverrideTest(TestCase):
    """``subagent_section`` lets a tool be skills-gated for the main agent yet
    always-on for sub-agents (the documents / web pattern)."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_section_overrides_section_for_subagents(self, mock_registry, mock_allowed):
        # section is set explicitly on every mock so getattr() sees a real value
        # (a bare MagicMock auto-creates truthy attributes, which would defeat
        # the ``subagent_section or section`` fallback).
        always_on = MagicMock(section="chat", subagent_section=None, audience="shared")
        doc_like = MagicMock(section="skills", subagent_section="chat", audience="shared")
        pure_skill = MagicMock(section="skills", subagent_section=None, audience="shared")

        mock_registry.return_value.list_tools.return_value = {
            "always_on": always_on,
            "doc_like": doc_like,
            "pure_skill": pure_skill,
        }

        user = _create_user(email="subsection@example.com")
        prefs = get_preferences(user)

        # Main agent: only the always-on chat tool; both skills tools are gated.
        self.assertIn("always_on", prefs.allowed_tools)
        self.assertNotIn("doc_like", prefs.allowed_tools)
        self.assertNotIn("pure_skill", prefs.allowed_tools)

        # Sub-agents: the always-on tool AND the subagent_section="chat" tool;
        # the pure skills tool stays gated for sub-agents too.
        self.assertIn("always_on", prefs.allowed_subagent_tools)
        self.assertIn("doc_like", prefs.allowed_subagent_tools)
        self.assertNotIn("pure_skill", prefs.allowed_subagent_tools)


class ParallelSubagentsTest(TestCase):
    """parallel_subagents is resolved from org subagent preferences."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_parallel_subagents_default_true(self, mock_registry, mock_allowed):
        """When org has no subagents prefs, parallel_subagents defaults to True."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="parallel-default@example.com")
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={})
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertTrue(prefs.parallel_subagents)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_parallel_subagents_from_org_prefs(self, mock_registry, mock_allowed):
        """When org sets parallel to False, parallel_subagents should be False."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="parallel-false@example.com")
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "subagents": {"parallel": False},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertFalse(prefs.parallel_subagents)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_parallel_subagents_true_from_org_prefs(self, mock_registry, mock_allowed):
        """When org explicitly sets parallel to True, parallel_subagents should be True."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="parallel-true@example.com")
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "subagents": {"parallel": True},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertTrue(prefs.parallel_subagents)


class MaxContextTokensTest(TestCase):
    """max_context_tokens is resolved from org and user preferences."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_default_200k_when_no_prefs(self, mock_registry, mock_allowed):
        """When no org or user setting, max_context_tokens defaults to 200k."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_sets_limit(self, mock_registry, mock_allowed):
        """Org sets max_context_tokens to 100k."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-org@example.com")
        org = Organization.objects.create(name="CtxOrg", slug="ctxorg", preferences={
            "max_context_tokens": 100_000,
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, 100_000)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_value_ignored_org_governs(self, mock_registry, mock_allowed):
        """The individual-user override was removed: a user value is ignored and
        the org limit governs (org 200k + user 150k -> 200k)."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-lower@example.com")
        org = Organization.objects.create(name="CtxOrg2", slug="ctxorg2", preferences={
            "max_context_tokens": 200_000,
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"max_context_tokens": 150_000}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, 200_000)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_value_never_raises_org(self, mock_registry, mock_allowed):
        """A user value (higher or lower) is ignored — org 100k stays 100k even
        when the user stored 200k."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-exceed@example.com")
        org = Organization.objects.create(name="CtxOrg3", slug="ctxorg3", preferences={
            "max_context_tokens": 100_000,
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"max_context_tokens": 200_000}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, 100_000)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_min_floor_enforced(self, mock_registry, mock_allowed):
        """Values below MIN_CONTEXT_TOKENS resolve to the floor."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-floor@example.com")
        org = Organization.objects.create(name="CtxOrg4", slug="ctxorg4", preferences={
            "max_context_tokens": 5_000,
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, MIN_CONTEXT_TOKENS)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_context_budgets_resolved(self, mock_registry, mock_allowed):
        """Per-tier sub-agent budgets come from org prefs; 0 / unknown tiers / bad
        types are dropped, positive ints are kept."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-budget@example.com")
        org = Organization.objects.create(name="CtxOrg5", slug="ctxorg5", preferences={
            "subagents": {"context_budget": {"medium": 120_000, "high": 0, "bogus": 5}},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.subagent_context_budgets, {"medium": 120_000})

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_context_budgets_default_empty(self, mock_registry, mock_allowed):
        """No org budget -> empty dict (every tier unlimited)."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-budget-none@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.subagent_context_budgets, {})
        self.assertEqual(prefs.subagent_reasoning, {})

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_reasoning_resolved(self, mock_registry, mock_allowed):
        """Per-tier sub-agent reasoning entries come from org prefs; malformed
        entries and unknown tiers are dropped."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="sa-reasoning@example.com")
        org = Organization.objects.create(name="ReasonOrg", slug="reasonorg", preferences={
            "subagents": {"reasoning": {
                "medium": {"model": "openai/gpt-6-luna", "level": "medium"},
                "high": {"model": "openai/gpt-6-luna"},
                "bogus": {"model": "openai/gpt-6-luna", "level": "low"},
            }},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(
            prefs.subagent_reasoning,
            {"medium": {"model": "openai/gpt-6-luna", "level": "medium"}},
        )


class TranscriptionModelCascadeTest(TestCase):
    """Transcription model preferences cascade: System -> Org -> User."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_system_default(self, mock_registry, mock_allowed):
        """Transcription model resolves to system default when no org/user prefs."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-mini-transcribe")
        self.assertEqual(
            prefs.allowed_transcription_models,
            ["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
        )

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_restricts_allowed(self, mock_registry, mock_allowed):
        """Org restricts allowed transcription models to a subset."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-org-restrict@example.com")
        org = Organization.objects.create(name="TxOrg1", slug="txorg1", preferences={
            "allowed_transcription_models": ["openai/gpt-4o-transcribe"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.allowed_transcription_models, ["openai/gpt-4o-transcribe"])
        # Default not in allowed => falls back to first allowed
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-transcribe")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_empty_list_disables(self, mock_registry, mock_allowed):
        """Org with explicitly empty allowed_transcription_models disables transcription."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-org-disabled@example.com")
        org = Organization.objects.create(name="TxOrg2", slug="txorg2", preferences={
            "allowed_transcription_models": [],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.allowed_transcription_models, [])
        self.assertEqual(prefs.transcription_model, "")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_stored_user_transcription_model_is_ignored(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-user-picks@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "transcription_models": {"default": "openai/gpt-4o-transcribe"},
        }
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-mini-transcribe")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_picks_disallowed_falls_back(self, mock_registry, mock_allowed):
        """User picks a disallowed model, falls back to system default."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-user-disallowed@example.com")
        org = Organization.objects.create(name="TxOrg3", slug="txorg3", preferences={
            "allowed_transcription_models": ["openai/gpt-4o-mini-transcribe"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "transcription_models": {"default": "openai/gpt-4o-transcribe"},
        }
        settings.save()

        prefs = get_preferences(user)
        # User's choice not in org allowed, so falls back
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-mini-transcribe")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_DEFAULT_MODEL_LIVE="",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_live_model_follows_org_default_when_no_live_override(self, mock_registry, mock_allowed):
        """The live model follows the organization's general default."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-live-follows-default@example.com")
        org = Organization.objects.create(name="TxDefault", slug="tx-default", preferences={
            "allowed_transcription_models": [
                "openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"
            ],
            "transcription_models": {"default": "openai/gpt-4o-transcribe"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-transcribe")
        self.assertEqual(prefs.transcription_model_live, "openai/gpt-4o-transcribe")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_DEFAULT_MODEL_LIVE="",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_explicit_org_live_override_wins_over_default(self, mock_registry, mock_allowed):
        """An explicit organization live override wins over its general default."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-live-override@example.com")
        org = Organization.objects.create(name="TxLive", slug="tx-live", preferences={
            "allowed_transcription_models": [
                "openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"
            ],
            "transcription_models": {
                "default": "openai/gpt-4o-transcribe",
                "live": "openai/gpt-4o-mini-transcribe",
            },
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_model_live, "openai/gpt-4o-mini-transcribe")


@override_settings(
    LLM_DEFAULT_MODEL="openai/gpt-5.4",
    LLM_DEFAULT_MID_MODEL="",
    LLM_DEFAULT_CHEAP_MODEL="",
)
@patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
@patch("llm.tools.registry.get_tool_registry")
class TranscriptionLanguageCascadeTest(TestCase):
    """Default transcription language cascade: System -> Org -> User.

    System default is ``"auto"`` (auto-detect). ``"auto"`` is a first-class
    value a user can pick to override an org language default.
    """

    def _mk(self, mock_registry):
        mock_registry.return_value.list_tools.return_value = {}

    def test_system_default_is_auto(self, mock_registry, mock_allowed):
        self._mk(mock_registry)
        user = _create_user(email="lang-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_language, "auto")

    def test_org_default_applies_to_user(self, mock_registry, mock_allowed):
        self._mk(mock_registry)
        user = _create_user(email="lang-org@example.com")
        org = Organization.objects.create(
            name="LangOrg1", slug="langorg1",
            preferences={"transcription_language": "no"},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_language, "no")

    def test_user_override_wins_over_org(self, mock_registry, mock_allowed):
        self._mk(mock_registry)
        user = _create_user(email="lang-user@example.com")
        org = Organization.objects.create(
            name="LangOrg2", slug="langorg2",
            preferences={"transcription_language": "no"},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"transcription_language": "en"}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_language, "en")

    def test_user_can_override_org_language_back_to_auto(self, mock_registry, mock_allowed):
        self._mk(mock_registry)
        user = _create_user(email="lang-user-auto@example.com")
        org = Organization.objects.create(
            name="LangOrg3", slug="langorg3",
            preferences={"transcription_language": "no"},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"transcription_language": "auto"}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_language, "auto")

    def test_invalid_value_ignored(self, mock_registry, mock_allowed):
        self._mk(mock_registry)
        user = _create_user(email="lang-invalid@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"transcription_language": "klingon"}
        settings.save()

        prefs = get_preferences(user)
        # Garbage is ignored and the cascade falls through to the system default.
        self.assertEqual(prefs.transcription_language, "auto")


class TierConstraintTest(TestCase):
    """Any allowed model may fill any slot; fit only steers automatic picks."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "anthropic/claude-opus-4-6",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_standard_model_accepted_for_cheap_slot(self, mock_registry, mock_allowed):
        """Opus set as the org's cheap model is used as the cheap model."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tier-cheap@example.com")
        org = Organization.objects.create(name="TierOrg", slug="tierorg", preferences={
            "allowed_models": ["openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano", "anthropic/claude-opus-4-6"],
            "models": {"cheap": "anthropic/claude-opus-4-6"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.cheap_model, "anthropic/claude-opus-4-6")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="anthropic/claude-fable-5-1",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "anthropic/claude-fable-5-1",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_system_default_outside_slot_is_honored(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tier-env@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.cheap_model, "anthropic/claude-fable-5-1")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "anthropic/claude-fable-5-1", "openai/gpt-5.4", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_automatic_cheap_pick_prefers_a_cheap_model(self, mock_registry, mock_allowed):
        """With nothing configured, the cheap slot skips Fable for Nano."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tier-auto@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.cheap_model, "openai/gpt-5.4-nano")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_cheap_model_accepted_for_cheap_slot(self, mock_registry, mock_allowed):
        """Nano as cheap model → accepted."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tier-accept@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"models": {"cheap": "openai/gpt-5.4-nano"}}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.cheap_model, "openai/gpt-5.4-nano")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.5-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_stored_user_mid_model_is_ignored(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tier-mid-std@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"models": {"mid": "openai/gpt-5.4"}}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.mid_model, "gemini/gemini-3.5-flash")


class FallbackWarningsTest(TestCase):
    """When a tier has no models, a warning is generated and fallbacks occur."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.5-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_no_cheap_falls_back_to_mid(self, mock_registry, mock_allowed):
        """Org allows only mid + standard → cheap falls back to mid with warning."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="fallback-cheap@example.com")
        org = Organization.objects.create(name="NoCheap", slug="nocheap", preferences={
            "allowed_models": ["openai/gpt-5.4", "gemini/gemini-3.5-flash"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.cheap_model, prefs.mid_model)
        self.assertTrue(any("cheap" in w.lower() for w in prefs.warnings))

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_no_standard_generates_warning(self, mock_registry, mock_allowed):
        """Org allows only cheap → warning about no standard-tier models."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="fallback-std@example.com")
        org = Organization.objects.create(name="NoStd", slug="nostd", preferences={
            "allowed_models": ["openai/gpt-5.4-nano"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertTrue(any("standard" in w.lower() for w in prefs.warnings))


class FeatureModelOverrideTest(TestCase):
    """Per-feature model overrides resolve correctly."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_feature_uses_tier_default_when_no_override(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="feat-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.feature_models["thread_title"], "openai/gpt-5.4-nano")
        self.assertEqual(prefs.feature_models["chat"], "openai/gpt-5.4")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_tiers_use_slot_defaults_when_no_override(self, mock_registry, mock_allowed):
        # No override -> subagent_medium follows the mid slot, subagent_high the
        # primary slot, i.e. identical to the historical resolve_subagent_model.
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="subagent-tier-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.feature_models["subagent_medium"], "openai/gpt-5.4-mini")
        self.assertEqual(prefs.feature_models["subagent_high"], "openai/gpt-5.4")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.6-sol", "openai/gpt-5.6-luna", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_subagent_tier_overrides_resolve(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="subagent-tier-override@example.com")
        org = Organization.objects.create(
            name="SA Override Org",
            slug="sa-override-org",
            preferences={
                "allowed_models": [
                    "openai/gpt-5.4", "openai/gpt-5.6-sol",
                    "openai/gpt-5.6-luna", "openai/gpt-5.4-nano",
                ],
                # sol (4*) clears the top 3-star floor; luna (2*) clears mid's 2-star floor.
                "feature_models": {
                    "subagent_high": "openai/gpt-5.6-sol",
                    "subagent_medium": "openai/gpt-5.6-luna",
                },
            },
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.feature_models["subagent_high"], "openai/gpt-5.6-sol")
        self.assertEqual(prefs.feature_models["subagent_medium"], "openai/gpt-5.6-luna")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.6-luna", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_subagent_high_override_with_few_stars_is_honored(self, mock_registry, mock_allowed):
        # Luna is 2 stars; there is no star floor, so the override is used.
        from llm.model_registry import canonical_model_id

        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="subagent-top-lowstar@example.com")
        org = Organization.objects.create(
            name="SA LowStar Org",
            slug="sa-lowstar-org",
            preferences={
                "allowed_models": [
                    "openai/gpt-5.4", "openai/gpt-5.6-luna", "openai/gpt-5.4-nano",
                ],
                "feature_models": {"subagent_high": "openai/gpt-5.6-luna"},
            },
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(
            canonical_model_id(prefs.feature_models["subagent_high"]),
            canonical_model_id("openai/gpt-5.6-luna"),
        )
        self.assertNotEqual(prefs.feature_models["subagent_high"], prefs.top_model)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.6-sol",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.7-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "gemini/gemini-3.7-flash",
        "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_chat_override_becomes_new_thread_default(self, mock_registry, mock_allowed):
        from core.preferences import resolve_thread_model

        mock_registry.return_value.list_tools.return_value = {}
        user = _create_user(email="org-chat-default@example.com")
        org = Organization.objects.create(
            name="Chat Default Org",
            slug="chat-default-org",
            preferences={
                "allowed_models": [
                    "openai/gpt-5.6-sol",
                    "openai/gpt-5.6-terra",
                    "gemini/gemini-3.7-flash",
                    "openai/gpt-5.4-nano",
                ],
                "feature_models": {"chat": "openai/gpt-5.6-terra"},
            },
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)

        self.assertEqual(
            prefs.feature_models["chat"], "openai/gpt-5.6-terra"
        )
        self.assertEqual(
            resolve_thread_model("", prefs), "openai/gpt-5.6-terra"
        )

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "gemini/gemini-3.1-flash-lite",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_stored_user_feature_override_is_ignored(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="feat-override@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"feature_models": {"thread_title": "gemini/gemini-3.1-flash-lite"}}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.feature_models["thread_title"], "openai/gpt-5.4-nano")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_feature_override_not_in_allowed_falls_back(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="feat-notallowed@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"feature_models": {"thread_title": "not-an-allowed-model"}}
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.feature_models["thread_title"], "openai/gpt-5.4-nano")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_feature_models_dict_is_populated(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="feat-dict@example.com")
        prefs = get_preferences(user)
        from core.preferences import FEATURE_DEFAULTS
        for fkey in FEATURE_DEFAULTS:
            self.assertIn(fkey, prefs.feature_models, f"feature_models missing key: {fkey}")


class FeatureModelPoolTest(TestCase):
    """feature_model_pool(): the one eligibility filter behind resolution, the
    settings rows and override validation."""

    def test_ungated_feature_uses_every_allowed_model(self):
        from core.preferences import FeatureDefault, feature_model_pool

        fdef = FeatureDefault("cheap", "org")
        allowed = ["openai/gpt-6-luna", "custom/plain"]
        self.assertFalse(fdef.gated)
        self.assertEqual(feature_model_pool(fdef, allowed), allowed)

    def test_modality_gate(self):
        from core.preferences import FeatureDefault, feature_model_pool

        fdef = FeatureDefault("mid", "org", required_modality="image")
        self.assertTrue(fdef.gated)
        self.assertEqual(
            feature_model_pool(fdef, ["openai/gpt-6-luna", "custom/plain"]), ["openai/gpt-6-luna"]
        )

    def test_capability_gate(self):
        from core.preferences import FeatureDefault, feature_model_pool

        fdef = FeatureDefault("cheap", "org", required_capability="decisions")
        self.assertTrue(fdef.gated)
        self.assertEqual(
            feature_model_pool(
                fdef, ["openai/gpt-6-sol", "openai/gpt-6-luna", "anthropic/claude-haiku-5-5"]
            ),
            ["openai/gpt-6-luna"],
        )
        self.assertEqual(feature_model_pool(fdef, ["openai/gpt-6-sol"]), [])

    def test_modality_and_capability_combine(self):
        from core.preferences import FeatureDefault, feature_model_pool

        fdef = FeatureDefault("cheap", "org", required_modality="image", required_capability="decisions")
        self.assertEqual(
            feature_model_pool(fdef, ["custom/plain", "openai/gpt-6-luna", "openai/gpt-6-sol"]),
            ["openai/gpt-6-luna"],
        )

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano", "gemini/gemini-3.5-flash",
    ])
    def test_resolve_from_prefs_matches_org_resolver(self, mock_allowed):
        from core.preferences import resolve_feature_model_from_prefs, resolve_org_feature_model

        org = Organization.objects.create(name="PoolOrg", slug="poolorg", preferences={
            "feature_models": {"document_description": "gemini/gemini-3.5-flash"},
        })
        for key in ("document_description", "chat", "guardrails_classifier"):
            with self.subTest(feature=key):
                self.assertEqual(
                    resolve_feature_model_from_prefs(org.preferences, key),
                    resolve_org_feature_model(org.pk, key),
                )
                self.assertEqual(
                    resolve_feature_model_from_prefs(None, key),
                    resolve_org_feature_model(None, key),
                )


class ResolveOrgFeatureReasoningTest(TestCase):
    """A pinned feature reasoning level is bound to the model it was chosen for."""

    def test_level_applies_only_for_the_stored_model(self):
        from core.preferences import resolve_org_feature_reasoning

        prefs = {"feature_reasoning": {"f": {"model": "openai/gpt-6-luna", "level": "low"}}}
        self.assertEqual(resolve_org_feature_reasoning(prefs, "f", "openai/gpt-6-luna"), "low")
        # The feature now runs on another model → that model's default.
        self.assertIsNone(resolve_org_feature_reasoning(prefs, "f", "openai/gpt-6-sol"))
        self.assertIsNone(resolve_org_feature_reasoning(prefs, "other", "openai/gpt-6-luna"))
        self.assertIsNone(resolve_org_feature_reasoning(prefs, "f", ""))
        self.assertIsNone(resolve_org_feature_reasoning(None, "f", "openai/gpt-6-luna"))
        self.assertIsNone(resolve_org_feature_reasoning({}, "f", "openai/gpt-6-luna"))

    def test_level_the_model_no_longer_offers_reverts_to_default(self):
        from core.preferences import resolve_org_feature_reasoning

        prefs = {"feature_reasoning": {"f": {"model": "openai/gpt-6-luna", "level": "ultra"}}}
        self.assertIsNone(resolve_org_feature_reasoning(prefs, "f", "openai/gpt-6-luna"))
        # Malformed entry (not a dict) is ignored, never raises.
        self.assertIsNone(resolve_org_feature_reasoning({"feature_reasoning": {"f": "low"}}, "f", "openai/gpt-6-luna"))
        # Unknown model: no registry levels to validate against.
        prefs = {"feature_reasoning": {"f": {"model": "custom/plain", "level": "low"}}}
        self.assertIsNone(resolve_org_feature_reasoning(prefs, "f", "custom/plain"))


_DECISIONS_SYSTEM_MODELS = [
    "openai/gpt-6-sol", "openai/gpt-6-luna", "anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5",
]


class DecisionsSettingsTest(TestCase):
    """resolve_decisions_settings(): the quick toggle, both pathway models and
    the pinned reasoning level from one org fetch."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-6-sol",
        LLM_DEFAULT_MID_MODEL="anthropic/claude-sonnet-5-5",
        LLM_DEFAULT_CHEAP_MODEL="anthropic/claude-haiku-5-5",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=_DECISIONS_SYSTEM_MODELS)
    def test_system_defaults(self, mock_allowed):
        from core.preferences import resolve_decisions_settings

        s = resolve_decisions_settings(None)
        self.assertTrue(s.quick_enabled)
        # The cheap slot (Haiku) lacks the capability → first capable allowed model.
        self.assertEqual(s.quick_model, "openai/gpt-6-luna")
        self.assertEqual(s.structured_model, "anthropic/claude-haiku-5-5")
        self.assertIsNone(s.structured_thinking_level)
        # A missing org behaves like no org.
        self.assertEqual(resolve_decisions_settings(999_999), s)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-6-sol",
        LLM_DEFAULT_MID_MODEL="anthropic/claude-sonnet-5-5",
        LLM_DEFAULT_CHEAP_MODEL="anthropic/claude-haiku-5-5",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=_DECISIONS_SYSTEM_MODELS)
    def test_org_toggle_overrides_and_pinned_level(self, mock_allowed):
        from core.preferences import resolve_decisions_settings

        org = Organization.objects.create(name="DecOrg", slug="decorg", preferences={
            "decisions_quick_enabled": False,
            "feature_models": {"decisions_structured": "openai/gpt-6-sol"},
            "feature_reasoning": {"decisions_structured": {"model": "openai/gpt-6-sol", "level": "low"}},
        })
        s = resolve_decisions_settings(org.pk)
        self.assertFalse(s.quick_enabled)
        self.assertEqual(s.quick_model, "openai/gpt-6-luna")
        self.assertEqual(s.structured_model, "openai/gpt-6-sol")
        self.assertEqual(s.structured_thinking_level, "low")
        # The level was pinned for Sol; once the feature runs on Sonnet it is inert.
        org.preferences["feature_models"]["decisions_structured"] = "anthropic/claude-sonnet-5-5"
        org.save(update_fields=["preferences"])
        s = resolve_decisions_settings(org.pk)
        self.assertEqual(s.structured_model, "anthropic/claude-sonnet-5-5")
        self.assertIsNone(s.structured_thinking_level)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-6-sol",
        LLM_DEFAULT_MID_MODEL="anthropic/claude-sonnet-5-5",
        LLM_DEFAULT_CHEAP_MODEL="anthropic/claude-haiku-5-5",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=_DECISIONS_SYSTEM_MODELS)
    def test_quick_model_empty_when_org_disallows_every_capable_model(self, mock_allowed):
        from core.preferences import resolve_decisions_settings

        org = Organization.objects.create(name="DecOrg2", slug="decorg2", preferences={
            "allowed_models": ["openai/gpt-6-sol", "anthropic/claude-haiku-5-5"],
        })
        s = resolve_decisions_settings(org.pk)
        self.assertTrue(s.quick_enabled)
        self.assertEqual(s.quick_model, "")
        self.assertEqual(s.structured_model, "anthropic/claude-haiku-5-5")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-6-sol",
        LLM_DEFAULT_MID_MODEL="openai/gpt-6-sol",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-6-sol",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-6-sol"])
    def test_quick_model_empty_when_not_allowed_system_wide(self, mock_allowed):
        from core.preferences import resolve_decisions_settings

        s = resolve_decisions_settings(None)
        self.assertEqual(s.quick_model, "")
        self.assertEqual(s.structured_model, "openai/gpt-6-sol")


class ModalityFeatureResolutionTest(TestCase):
    """Capability-gated features (required_modality) resolve to capable models
    and report availability correctly."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.5-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_image_feature_resolves_to_vision_model(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}
        from core.preferences import feature_is_available

        user = _create_user(email="modality-ok@example.com")
        prefs = get_preferences(user)
        # default_slot="mid" and the mid model is vision-capable.
        self.assertEqual(prefs.feature_models["document_image_description"], "gemini/gemini-3.5-flash")
        self.assertTrue(feature_is_available(user, "document_image_description"))
        # spreadsheet_description shares the same modality gate.
        self.assertEqual(prefs.feature_models["spreadsheet_description"], "gemini/gemini-3.5-flash")
        self.assertTrue(feature_is_available(user, "spreadsheet_description"))

    @override_settings(
        LLM_DEFAULT_MODEL="openai/whisper-1",
        LLM_DEFAULT_MID_MODEL="openai/whisper-1",
        LLM_DEFAULT_CHEAP_MODEL="openai/whisper-1",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/whisper-1"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_image_feature_unavailable_without_vision_model(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}
        from core.preferences import feature_is_available, resolve_org_feature_model

        user = _create_user(email="modality-none@example.com")
        prefs = get_preferences(user)
        # No vision-capable model in the allow-list → feature disabled (empty).
        self.assertEqual(prefs.feature_models["document_image_description"], "")
        self.assertFalse(feature_is_available(user, "document_image_description"))
        self.assertEqual(prefs.feature_models["spreadsheet_description"], "")
        self.assertFalse(feature_is_available(user, "spreadsheet_description"))
        # Features without a required modality are always available.
        self.assertTrue(feature_is_available(user, "document_description"))
        # The org resolver also yields "" when no capable model exists.
        self.assertEqual(resolve_org_feature_model(None, "document_image_description"), "")
        self.assertEqual(resolve_org_feature_model(None, "spreadsheet_description"), "")


class ResolveOrgFeatureModelTest(TestCase):
    """Tests for resolve_org_feature_model()."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    def test_no_org_returns_system_default(self, mock_allowed):
        from core.preferences import resolve_org_feature_model
        # document_description defaults to the mid slot.
        model = resolve_org_feature_model(None, "document_description")
        self.assertEqual(model, "openai/gpt-5.4-mini")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-6-astra",
        LLM_DEFAULT_MID_MODEL="openai/gpt-6-astra",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.6-luna",  # retired alias of gpt-6-luna
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-6-astra", "openai/gpt-6-luna",
    ])
    def test_no_org_canonicalises_a_retired_alias_in_the_system_default(self, mock_allowed):
        # The allow-list is canonical; the env default is not. Without matching,
        # the cheap slot missed the list and fell to the first allowed model
        # (Astra, a flagship) for every no-org caller.
        from core.preferences import resolve_decisions_settings, resolve_org_feature_model

        self.assertEqual(resolve_org_feature_model(None, "guardrails_classifier"), "openai/gpt-6-luna")
        self.assertEqual(resolve_decisions_settings(None).structured_model, "openai/gpt-6-luna")
        # A default that is genuinely not allowed still falls to the first allowed model.
        with override_settings(LLM_DEFAULT_CHEAP_MODEL="anthropic/claude-haiku-5-5"):
            self.assertEqual(resolve_org_feature_model(None, "guardrails_classifier"), "openai/gpt-6-astra")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
        "gemini/gemini-3.5-flash",
    ])
    def test_org_feature_override(self, mock_allowed):
        from core.preferences import resolve_org_feature_model

        # gemini-3.5-flash (→ 3.7-flash, 2 stars) clears
        # document_description's 2-star minimum, so the org override applies.
        org = Organization.objects.create(name="FeatOrg", slug="featorg", preferences={
            "feature_models": {"document_description": "gemini/gemini-3.5-flash"},
        })
        model = resolve_org_feature_model(org.pk, "document_description")
        self.assertEqual(model, "gemini/gemini-3.5-flash")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="gemini/gemini-3.5-flash",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "gemini/gemini-3.5-flash", "openai/gpt-5.4-nano",
        "gemini/gemini-3.1-flash-lite",
    ])
    def test_org_feature_override_with_few_stars_is_honored(self, mock_allowed):
        from core.preferences import resolve_org_feature_model
        from llm.model_registry import canonical_model_id

        # gemini-3.1-flash-lite (→ 3.5-flash-lite, 1 star) on a mid-slot
        # feature: there is no star floor, so the override is used.
        org = Organization.objects.create(name="FeatOrg2", slug="featorg2", preferences={
            "feature_models": {"document_description": "gemini/gemini-3.1-flash-lite"},
        })
        model = resolve_org_feature_model(org.pk, "document_description")
        self.assertEqual(canonical_model_id(model), "gemini/gemini-3.5-flash-lite")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    def test_nonexistent_org_returns_system_default(self, mock_allowed):
        from core.preferences import resolve_org_feature_model
        model = resolve_org_feature_model(99999, "document_description")
        self.assertEqual(model, "openai/gpt-5.4-mini")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5.4-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5.4-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    ])
    def test_unknown_feature_returns_primary_default(self, mock_allowed):
        from core.preferences import resolve_org_feature_model
        model = resolve_org_feature_model(None, "nonexistent_feature")
        self.assertEqual(model, "openai/gpt-5.4")


class AllowAgentAttachSkillsTest(TestCase):
    """The allow_agent_attach_skills user preference."""

    def test_defaults_true(self):
        user = _create_user()
        prefs = get_preferences(user)
        self.assertTrue(prefs.allow_agent_attach_skills)

    def test_user_can_disable(self):
        user = _create_user()
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"allow_agent_attach_skills": False}
        settings.save()
        prefs = get_preferences(user)
        self.assertFalse(prefs.allow_agent_attach_skills)

    def test_user_explicit_true(self):
        user = _create_user()
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {"allow_agent_attach_skills": True}
        settings.save()
        prefs = get_preferences(user)
        self.assertTrue(prefs.allow_agent_attach_skills)

    def test_org_prefs_do_not_override(self):
        """The pref is user-scope only — org prefs do not cascade."""
        user = _create_user()
        org = Organization.objects.create(
            name="Org", preferences={"allow_agent_attach_skills": False},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        prefs = get_preferences(user)
        self.assertTrue(prefs.allow_agent_attach_skills)


class SkillResourceAutoGrantsViewToolsTest(TestCase):
    """Every main/shared skill auto-grants skill_resource_view/load so the agent
    can read a skill's resources the moment it's attached — the author never adds
    them by hand, and it doesn't depend on a resource existing yet. Sub-agent
    specializations don't get them (the tools are main-only)."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_main_skills_grant_view_tools(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        from agent_skills.models import AgentSkill

        user = _create_user(email="autogrant@example.com")
        # A main skill with no resources still gets them (available on attach).
        AgentSkill.objects.create(
            slug="bare", name="Bare", instructions="i", level="user",
            created_by=user, audience="main",
        )
        # A sub-agent specialization must NOT get the main-only tools.
        AgentSkill.objects.create(
            slug="sub", name="Sub", instructions="i", level="user",
            created_by=user, audience="subagent",
        )

        prefs = get_preferences(user)
        main = {e["slug"]: e for e in prefs.allowed_skills}
        self.assertIn("skill_resource_view", main["bare"]["tool_names"])
        self.assertIn("skill_resource_load", main["bare"]["tool_names"])

        specs = {e["slug"]: e for e in prefs.allowed_specializations}
        self.assertNotIn("skill_resource_view", specs["sub"]["tool_names"])

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_can_disable_the_view_tools(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        from accounts.models import Membership, Organization
        from agent_skills.models import AgentSkill

        user = _create_user(email="autogrant-off@example.com")
        org = Organization.objects.create(
            name="O", slug="o",
            preferences={"tools": {"skill_resource_view": False}},
        )
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)
        AgentSkill.objects.create(
            slug="bare", name="Bare", instructions="i", level="user",
            created_by=user, audience="main",
        )
        prefs = get_preferences(user)
        entry = next(e for e in prefs.allowed_skills if e["slug"] == "bare")
        self.assertNotIn("skill_resource_view", entry["tool_names"])


class SkillApprovalInEntriesTests(TestCase):
    """allowed_skills / allowed_specializations entries carry the safety-scan
    verdict, so the chat catalogue and picker never offer what
    chat_skill_attach refuses."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_entries_carry_approved_and_scan_state(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        from agent_skills.models import AgentSkill
        from agent_skills.resources import compute_skill_content_hash

        user = _create_user(email="verdict@example.com")
        ok = AgentSkill.objects.create(
            slug="ok", name="Ok", instructions="i", level="user", created_by=user,
        )
        ok.scan_state = AgentSkill.ScanState.APPROVED
        ok.approved_content_hash = compute_skill_content_hash(ok)
        ok.save(update_fields=["scan_state", "approved_content_hash"])
        AgentSkill.objects.create(
            slug="raw", name="Raw", instructions="i", level="user", created_by=user,
        )
        AgentSkill.objects.create(
            slug="bad", name="Bad", instructions="i", level="user", created_by=user,
            scan_state=AgentSkill.ScanState.BLOCKED,
        )
        AgentSkill.objects.create(
            slug="spec", name="Spec", instructions="i", level="user", created_by=user,
            audience="subagent", scan_state=AgentSkill.ScanState.PENDING,
        )

        with patch("agent_skills.resources._scanning_configured", return_value=True):
            prefs = get_preferences(user)

        by_slug = {e["slug"]: e for e in prefs.allowed_skills}
        self.assertTrue(by_slug["ok"]["approved"])
        self.assertEqual(by_slug["ok"]["scan_state"], "approved")
        self.assertFalse(by_slug["raw"]["approved"])
        self.assertEqual(by_slug["raw"]["scan_state"], "unscanned")
        self.assertFalse(by_slug["bad"]["approved"])
        self.assertEqual(by_slug["bad"]["scan_state"], "blocked")
        specs = {e["slug"]: e for e in prefs.allowed_specializations}
        self.assertFalse(specs["spec"]["approved"])
        self.assertEqual(specs["spec"]["scan_state"], "pending")


class RequiresInEntriesTests(TestCase):
    """allowed_skills entries carry ``kind``, ``requires`` and ``requires_unmet``
    (why a dependent can't be attached: a requirement that isn't available or
    hasn't passed the safety scan). Specializations never carry requirements."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5.4",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5.4"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_entries_carry_requires_and_unmet(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        from agent_skills.models import AgentSkill
        from agent_skills.resources import compute_skill_content_hash

        user = _create_user(email="requires@example.com")
        AgentSkill.objects.all().delete()  # the seeded toolkits would collide
        AgentSkill.objects.create(
            slug="canvas_tools", name="Canvas Tools", instructions="i",
            level="system", kind="toolkit",
        )
        for slug, requires in (
            ("dep", ["canvas_tools"]),
            ("orphan", ["no_such_tools"]),
            ("on-raw", ["raw"]),
            ("chain", ["on-raw"]),
        ):
            s = AgentSkill.objects.create(
                slug=slug, name=slug.title(), instructions="i", level="user",
                created_by=user, requires=requires,
            )
            s.scan_state = AgentSkill.ScanState.APPROVED
            s.approved_content_hash = compute_skill_content_hash(s)
            s.save(update_fields=["scan_state", "approved_content_hash"])
        AgentSkill.objects.create(
            slug="raw", name="Raw", instructions="i", level="user", created_by=user,
        )
        AgentSkill.objects.create(
            slug="spec", name="Spec", instructions="i", level="user", created_by=user,
            audience="subagent", requires=["canvas_tools"],
        )

        with patch("agent_skills.resources._scanning_configured", return_value=True):
            prefs = get_preferences(user)

        by_slug = {e["slug"]: e for e in prefs.allowed_skills}
        self.assertEqual(by_slug["canvas_tools"]["kind"], "toolkit")
        self.assertEqual(by_slug["canvas_tools"]["requires"], [])
        self.assertEqual(by_slug["dep"]["kind"], "playbook")
        self.assertEqual(by_slug["dep"]["requires"], ["canvas_tools"])
        self.assertEqual(by_slug["dep"]["requires_unmet"], [])
        self.assertEqual(
            by_slug["orphan"]["requires_unmet"],
            [{"slug": "no_such_tools", "name": "no_such_tools", "reason": "unavailable"}],
        )
        self.assertEqual(
            by_slug["on-raw"]["requires_unmet"],
            [{"slug": "raw", "name": "Raw", "reason": "unapproved"}],
        )
        # The reason propagates through a chain.
        self.assertEqual(
            by_slug["chain"]["requires_unmet"],
            [{"slug": "on-raw", "name": "On-Raw", "reason": "unapproved"}],
        )
        specs = {e["slug"]: e for e in prefs.allowed_specializations}
        self.assertEqual(specs["spec"]["requires_unmet"], [])
