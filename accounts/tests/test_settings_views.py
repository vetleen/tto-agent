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
            top_model="openai/gpt-5",
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
            top_model="openai/gpt-5",
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
            top_model="openai/gpt-5",
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

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_tools_grouped_by_section(self, mock_reg, mock_models):
        from unittest.mock import MagicMock
        chat_tool = MagicMock(section="chat", description="Fetch a URL")
        mock_reg.return_value.list_tools.return_value = {
            "web_fetch": chat_tool,
        }
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wilfred Chat")

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_skills_section_tools_excluded_from_tool_sections(self, mock_reg, mock_models):
        from unittest.mock import MagicMock
        chat_tool = MagicMock(section="chat", description="Fetch a URL")
        skills_tool = MagicMock(section="skills", description="Create a skill")
        mock_reg.return_value.list_tools.return_value = {
            "web_fetch": chat_tool,
            "create_skill": skills_tool,
        }
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        tool_sections = response.context["tool_sections"]
        self.assertIn("chat", tool_sections)
        self.assertNotIn("skills", tool_sections)

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_json_serialized_context_vars(self, mock_reg, mock_models):
        mock_reg.return_value.list_tools.return_value = {}
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        # Verify JSON-serialized vars are in context
        self.assertIn("org_tools_json", response.context)
        self.assertIn("skills_data_json", response.context)
        self.assertIn("org_models_json", response.context)
        # Verify they are valid JSON strings
        import json
        json.loads(response.context["org_tools_json"])
        json.loads(response.context["skills_data_json"])
        json.loads(response.context["org_models_json"])

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_globally_disabled_tool_in_org_tools_json(self, mock_reg, mock_models):
        mock_reg.return_value.list_tools.return_value = {}
        self.org.preferences = {"tools": {"create_skill": False}}
        self.org.save(update_fields=["preferences"])
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        import json
        org_tools = json.loads(response.context["org_tools_json"])
        self.assertFalse(org_tools["create_skill"])

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


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgSkillsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="skilladmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.member_user = User.objects.create_user(
            email="skillmember@example.com", password=self.password,
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="SkillOrg", slug="skillorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_skills_update")

    def test_admin_can_toggle_skill(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"slug": "skill-creator", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertFalse(self.org.preferences["skills"]["skill-creator"]["enabled"])

    def test_admin_can_toggle_per_tool(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"slug": "skill-creator", "tool": "create_skill", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertFalse(
            self.org.preferences["skills"]["skill-creator"]["tools"]["create_skill"]
        )

    def test_non_admin_gets_403(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"slug": "skill-creator", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"slug": "skill-creator", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)

    def test_missing_slug_returns_400(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


@override_settings(ALLOWED_HOSTS=["testserver"])
class PreferencesSkillCascadeTests(TestCase):
    """Test that the preferences resolver respects skill toggles."""

    def setUp(self):
        from agent_skills.models import AgentSkill

        AgentSkill.objects.all().delete()

        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="cascade@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="CascadeOrg", slug="cascadeorg")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)

        self.skill = AgentSkill.objects.create(
            slug="test-skill", name="Test Skill", instructions="Inst.",
            level="system", tool_names=["create_skill", "edit_skill"],
        )

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_disabled_skill_excluded(self, mock_reg, mock_models):
        from unittest.mock import MagicMock

        mock_reg.return_value.list_tools.return_value = {}
        # Disable the skill in org prefs
        self.org.preferences = {"skills": {"test-skill": {"enabled": False}}}
        self.org.save(update_fields=["preferences"])

        from core.preferences import get_preferences

        prefs = get_preferences(self.user)
        skill_slugs = [s["slug"] for s in prefs.allowed_skills]
        self.assertNotIn("test-skill", skill_slugs)

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_enabled_skill_included(self, mock_reg, mock_models):
        from unittest.mock import MagicMock

        mock_reg.return_value.list_tools.return_value = {}

        from core.preferences import get_preferences

        prefs = get_preferences(self.user)
        skill_slugs = [s["slug"] for s in prefs.allowed_skills]
        self.assertIn("test-skill", skill_slugs)

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_per_tool_toggle(self, mock_reg, mock_models):
        from unittest.mock import MagicMock

        mock_reg.return_value.list_tools.return_value = {}
        self.org.preferences = {
            "skills": {"test-skill": {"tools": {"create_skill": False}}}
        }
        self.org.save(update_fields=["preferences"])

        from core.preferences import get_preferences

        prefs = get_preferences(self.user)
        skill_entry = next(s for s in prefs.allowed_skills if s["slug"] == "test-skill")
        self.assertNotIn("create_skill", skill_entry["tool_names"])
        self.assertIn("edit_skill", skill_entry["tool_names"])

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_skill_tools_gated_by_tool_names(self, mock_reg, mock_models):
        from unittest.mock import MagicMock

        mock_reg.return_value.list_tools.return_value = {}

        from core.preferences import get_preferences

        prefs = get_preferences(self.user)
        # Skill tools should NOT be in the base allowed_tools — they are
        # only injected at chat time when the skill is active
        self.assertNotIn("create_skill", prefs.allowed_tools)
        self.assertNotIn("edit_skill", prefs.allowed_tools)
        # But they should be listed in the skill's allowed_skills entry
        skill_entry = next(s for s in prefs.allowed_skills if s["slug"] == "test-skill")
        self.assertIn("create_skill", skill_entry["tool_names"])
        self.assertIn("edit_skill", skill_entry["tool_names"])
