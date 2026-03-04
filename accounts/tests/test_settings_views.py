"""Tests for accounts settings views (theme_update, preferences, org settings)."""
import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization, UserSettings

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class ThemeUpdateViewTests(TestCase):
    def setUp(self) -> None:
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="tester@example.com",
            password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:theme_update")

    def test_requires_login(self) -> None:
        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_rejects_get_request(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)

    def test_set_theme_to_dark(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "dark")
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.theme, "dark")

    def test_set_theme_to_light(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        # First set to dark
        self.client.post(self.url, {"theme": "dark"})
        # Then switch back to light
        response = self.client.post(self.url, {"theme": "light"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "light")
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.theme, "light")

    def test_invalid_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "invalid"})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)

    def test_empty_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": ""})
        self.assertEqual(response.status_code, 400)

    def test_missing_theme_returns_400(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {})
        self.assertEqual(response.status_code, 400)

    def test_theme_value_is_stripped_and_lowercased(self) -> None:
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(self.url, {"theme": "  DARK  "})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["theme"], "dark")

    def test_creates_settings_if_not_exists(self) -> None:
        """theme_update should create UserSettings via get_or_create if missing."""
        self.client.login(email=self.user.email, password=self.password)
        # Delete any auto-created settings from signal
        UserSettings.objects.filter(user=self.user).delete()
        self.assertFalse(UserSettings.objects.filter(user=self.user).exists())

        response = self.client.post(self.url, {"theme": "dark"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(UserSettings.objects.filter(user=self.user).exists())
        self.assertEqual(UserSettings.objects.get(user=self.user).theme, "dark")

    def test_theme_dual_write_to_preferences(self) -> None:
        """theme_update should write theme to preferences JSON too."""
        self.client.login(email=self.user.email, password=self.password)
        self.client.post(self.url, {"theme": "dark"})
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.preferences.get("theme"), "dark")


@override_settings(ALLOWED_HOSTS=["testserver"])
class SettingsPageTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="settings@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:settings")

    def test_requires_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    @patch("core.preferences.get_preferences")
    def test_renders_settings_page(self, mock_prefs):
        from core.preferences import ResolvedPreferences
        mock_prefs.return_value = ResolvedPreferences(
            primary_model="openai/gpt-5",
            mid_model="openai/gpt-5-mini",
            cheap_model="",
            allowed_models=["openai/gpt-5", "openai/gpt-5-mini"],
            allowed_tools=["search_documents"],
            theme="light",
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI Models")


@override_settings(ALLOWED_HOSTS=["testserver"])
class PreferencesModelsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="prefs@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:preferences_models_update")

    @patch("core.preferences.get_preferences")
    def test_update_model_preference(self, mock_prefs):
        from core.preferences import ResolvedPreferences
        mock_prefs.return_value = ResolvedPreferences(
            primary_model="openai/gpt-5",
            mid_model="",
            cheap_model="",
            allowed_models=["openai/gpt-5", "anthropic/claude-sonnet-4-5"],
            allowed_tools=[],
            theme="light",
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"tier": "primary", "model": "anthropic/claude-sonnet-4-5"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])

        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.preferences["models"]["primary"], "anthropic/claude-sonnet-4-5")

    @patch("core.preferences.get_preferences")
    def test_reject_model_not_in_allowed(self, mock_prefs):
        from core.preferences import ResolvedPreferences
        mock_prefs.return_value = ResolvedPreferences(
            primary_model="openai/gpt-5",
            mid_model="",
            cheap_model="",
            allowed_models=["openai/gpt-5"],
            allowed_tools=[],
            theme="light",
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"tier": "primary", "model": "not-allowed-model"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_tier_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"tier": "invalid", "model": "any"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_clear_model_preference(self):
        """Sending empty model clears the preference."""
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"tier": "primary", "model": ""}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        settings = UserSettings.objects.get(user=self.user)
        self.assertIsNone(settings.preferences["models"]["primary"])


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgSettingsAccessTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="admin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.member_user = User.objects.create_user(
            email="member@example.com", password=self.password,
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TestOrg", slug="testorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_settings")

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_admin_can_access_org_settings(self, mock_reg, mock_models):
        mock_reg.return_value.list_tools.return_value = {}
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "TestOrg")

    def test_member_gets_403(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 403)

    def test_requires_login(self):
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 302)


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgAllowedModelsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="orgadmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TestOrg", slug="testorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        self.url = reverse("accounts:org_allowed_models_update")

    @patch("llm.service.policies.get_allowed_models", return_value=[
        "openai/gpt-5", "anthropic/claude-sonnet-4-5",
    ])
    def test_admin_sets_allowed_models(self, mock_models):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_models": ["openai/gpt-5"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["allowed_models"], ["openai/gpt-5"])

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    def test_reject_model_not_in_system(self, mock_models):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_models": ["not-a-system-model"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgToolsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="tooladmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TestOrg", slug="testorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        self.url = reverse("accounts:org_tools_update")

    def test_admin_can_toggle_tool(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"name": "read_document", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertFalse(self.org.preferences["tools"]["read_document"])
