"""Unit tests for core.preferences cascading resolution."""

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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_system_defaults(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {
            "search_documents": None, "read_document": None,
        }
        user = _create_user()
        prefs = get_preferences(user)

        self.assertEqual(prefs.top_model, "openai/gpt-5")
        self.assertEqual(prefs.mid_model, "openai/gpt-5-mini")
        self.assertEqual(prefs.cheap_model, "openai/gpt-5-nano")
        self.assertEqual(prefs.allowed_models, [
            "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
        ])
        self.assertIn("search_documents", prefs.allowed_tools)
        self.assertIn("read_document", prefs.allowed_tools)


class OrgRestrictsModelsTest(TestCase):
    """Org allowed_models restricts the effective allowed list."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
        "anthropic/claude-sonnet-4-5",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_allowed_restricts(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5", "anthropic/claude-sonnet-4-5"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.allowed_models, [
            "openai/gpt-5", "anthropic/claude-sonnet-4-5",
        ])


class UserPicksModelTest(TestCase):
    """User's choice wins when it's in the effective allowed list."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "anthropic/claude-sonnet-4-5",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_choice_wins(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5", "anthropic/claude-sonnet-4-5"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "models": {"primary": "anthropic/claude-sonnet-4-5"},
        }
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.top_model, "anthropic/claude-sonnet-4-5")


class UserPicksOutsideAllowedTest(TestCase):
    """User picks a model not in org's allowed list; falls back."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "anthropic/claude-sonnet-4-5",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_outside_allowed_falls_back(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "allowed_models": ["openai/gpt-5"],
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "models": {"primary": "anthropic/claude-sonnet-4-5"},
        }
        settings.save()

        prefs = get_preferences(user)
        # Falls back to system default (gpt-5) since user's choice is not in org allowed
        self.assertEqual(prefs.top_model, "openai/gpt-5")


class OrgDisablesToolTest(TestCase):
    """Org disabling a tool removes it from allowed_tools."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_disables_tool(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {
            "search_documents": None,
            "read_document": None,
        }

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "tools": {"read_document": False},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertIn("search_documents", prefs.allowed_tools)
        self.assertNotIn("read_document", prefs.allowed_tools)


class ThemeFromPreferencesTest(TestCase):
    """Theme is read from the preferences JSON."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_theme_defaults_to_light(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        prefs = get_preferences(user)
        self.assertEqual(prefs.theme, "light")


class OrgDefaultModelTest(TestCase):
    """Org can set a default model per tier."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "anthropic/claude-sonnet-4-5",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_default_model(self, mock_registry, mock_allowed):
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            "models": {"primary": "anthropic/claude-sonnet-4-5"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        self.assertEqual(prefs.top_model, "anthropic/claude-sonnet-4-5")


class OrgDefaultRemovedFromAllowedTest(TestCase):
    """Org default model that was removed from allowed list must not be used."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
        "anthropic/claude-sonnet-4-5",
    ])
    @patch("llm.tools.registry.get_tool_registry")
    def test_org_default_removed_from_allowed_falls_back(self, mock_registry, mock_allowed):
        """If org sets a default but then removes it from allowed_models, fall back."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user()
        org = Organization.objects.create(name="TestOrg", slug="testorg", preferences={
            # Default is claude, but allowed list only has gpt-5
            "allowed_models": ["openai/gpt-5"],
            "models": {"primary": "anthropic/claude-sonnet-4-5"},
        })
        Membership.objects.create(user=user, org=org, role=Membership.Role.MEMBER)

        prefs = get_preferences(user)
        # Must NOT resolve to claude since it's not in the org's allowed list
        self.assertNotEqual(prefs.top_model, "anthropic/claude-sonnet-4-5")
        self.assertEqual(prefs.top_model, "openai/gpt-5")


class OrgAllowedModelsNoOverlapTest(TestCase):
    """When org's allowed_models has no overlap with system, must not fall back to system default."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="openai/gpt-5-mini",
        LLM_DEFAULT_CHEAP_MODEL="openai/gpt-5-nano",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_processing_tools_excluded_from_allowed_tools(self, mock_registry, mock_allowed):
        chat_tool = MagicMock(section="chat")
        proc_tool = MagicMock(section="document_processing")

        mock_registry.return_value.list_tools.return_value = {
            "search_documents": chat_tool,
            "normalize_document": proc_tool,
        }

        user = _create_user(email="section@example.com")
        prefs = get_preferences(user)

        self.assertIn("search_documents", prefs.allowed_tools)
        self.assertNotIn("normalize_document", prefs.allowed_tools)


class ParallelSubagentsTest(TestCase):
    """parallel_subagents is resolved from org subagent preferences."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_default_200k_when_no_prefs(self, mock_registry, mock_allowed):
        """When no org or user setting, max_context_tokens defaults to 200k."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="ctx-default@example.com")
        prefs = get_preferences(user)
        self.assertEqual(prefs.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_lowers_below_org(self, mock_registry, mock_allowed):
        """User sets 150k when org allows 200k — resolves to 150k."""
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
        self.assertEqual(prefs.max_context_tokens, 150_000)

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_cannot_exceed_org(self, mock_registry, mock_allowed):
        """User sets 200k when org limits to 100k — resolves to 100k."""
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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


class TranscriptionModelCascadeTest(TestCase):
    """Transcription model preferences cascade: System -> Org -> User."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_user_picks_allowed(self, mock_registry, mock_allowed):
        """User picks an allowed transcription model."""
        mock_registry.return_value.list_tools.return_value = {}

        user = _create_user(email="tx-user-picks@example.com")
        settings = UserSettings.objects.get(user=user)
        settings.preferences = {
            "transcription_models": {"default": "openai/gpt-4o-transcribe"},
        }
        settings.save()

        prefs = get_preferences(user)
        self.assertEqual(prefs.transcription_model, "openai/gpt-4o-transcribe")

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-5",
        LLM_DEFAULT_MID_MODEL="",
        LLM_DEFAULT_CHEAP_MODEL="",
        TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
        TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    )
    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
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
