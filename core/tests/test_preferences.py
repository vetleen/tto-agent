"""Unit tests for core.preferences cascading resolution."""

from unittest.mock import patch

from django.test import TestCase, override_settings

from accounts.models import Membership, Organization, UserSettings
from core.preferences import ResolvedPreferences, get_preferences


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

        self.assertEqual(prefs.primary_model, "openai/gpt-5")
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
        self.assertEqual(prefs.primary_model, "anthropic/claude-sonnet-4-5")


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
        self.assertEqual(prefs.primary_model, "openai/gpt-5")


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
        self.assertEqual(prefs.primary_model, "anthropic/claude-sonnet-4-5")
