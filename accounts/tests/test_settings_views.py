"""Tests for accounts settings views (theme_update, preferences, org settings)."""
import json
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization, UserSettings
from agent_skills.models import AgentSkill

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
        self.assertContains(response, f"{settings.ASSISTANT_NAME} Chat")

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

    @patch("llm.service.policies.get_allowed_models", return_value=["openai/gpt-5"])
    @patch("llm.tools.registry.get_tool_registry")
    def test_disabled_system_skill_still_listed_for_admin(self, mock_reg, mock_models):
        """Org admins need to see disabled skills so they can re-enable them.

        The overview page and chat flow hide these (covered elsewhere); the
        settings page intentionally does not.
        """
        from agent_skills.models import AgentSkill

        mock_reg.return_value.list_tools.return_value = {}
        AgentSkill.objects.filter(slug="disabled-demo").delete()
        AgentSkill.objects.create(
            slug="disabled-demo", name="Disabled Demo",
            instructions="i", level="system",
        )
        self.org.preferences = {"skills": {"disabled-demo": {"enabled": False}}}
        self.org.save(update_fields=["preferences"])

        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        skills = response.context["skills_data"]
        row = next((s for s in skills if s["slug"] == "disabled-demo"), None)
        self.assertIsNotNone(row, "Disabled skill must remain visible on the settings page")
        self.assertFalse(row["enabled"])


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


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgSubagentsUpdateViewTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="subadmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.member_user = User.objects.create_user(
            email="submember@example.com", password=self.password,
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="SubOrg", slug="suborg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_subagents_update")

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"parallel": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_requires_admin(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"parallel": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_set_parallel_false(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"parallel": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["parallel"])
        self.org.refresh_from_db()
        self.assertFalse(self.org.preferences["subagents"]["parallel"])

    def test_set_parallel_true(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"parallel": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["parallel"])
        self.org.refresh_from_db()
        self.assertTrue(self.org.preferences["subagents"]["parallel"])

    def test_invalid_json_returns_400(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            "not valid json{{{",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


@override_settings(ALLOWED_HOSTS=["testserver"])
class OrgMaxContextUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="ctxadmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.member_user = User.objects.create_user(
            email="ctxmember@example.com", password=self.password,
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="CtxOrg", slug="ctxorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_max_context_update")

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 100_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_requires_admin(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 100_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_sets_max_context(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 100_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["max_context_tokens"], 100_000)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["max_context_tokens"], 100_000)

    def test_below_minimum_returns_400(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 5_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_non_integer_returns_400(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": "not a number"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_clear_value(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        # First set a value
        self.org.preferences = {"max_context_tokens": 100_000}
        self.org.save(update_fields=["preferences"])
        # Then clear it
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": None}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertNotIn("max_context_tokens", self.org.preferences)


@override_settings(ALLOWED_HOSTS=["testserver"])
class PreferencesMaxContextUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="ctxuser@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(
            name="CtxOrg2", slug="ctxorg2",
            preferences={"max_context_tokens": 200_000},
        )
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:preferences_max_context_update")

    def test_user_sets_max_context(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 150_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["max_context_tokens"], 150_000)
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.preferences["max_context_tokens"], 150_000)

    def test_user_exceeds_org_limit_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 300_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_user_clears_value(self):
        self.client.login(email=self.user.email, password=self.password)
        # Set a value first
        settings, _ = UserSettings.objects.get_or_create(user=self.user)
        settings.preferences = {"max_context_tokens": 100_000}
        settings.save()
        # Clear it
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": None}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        settings.refresh_from_db()
        self.assertNotIn("max_context_tokens", settings.preferences)

    def test_below_minimum_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"max_context_tokens": 5_000}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
)
class OrgAllowedTranscriptionModelsUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="txadmin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.member_user = User.objects.create_user(
            email="txmember@example.com", password=self.password,
        )
        self.member_user.email_verified = True
        self.member_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TxOrg", slug="txorg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member_user, org=self.org, role=Membership.Role.MEMBER)
        self.url = reverse("accounts:org_allowed_transcription_models_update")

    def test_admin_sets_allowed_transcription_models(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_transcription_models": ["openai/gpt-4o-transcribe"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["allowed_transcription_models"], ["openai/gpt-4o-transcribe"])

    def test_reject_model_not_in_system(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_transcription_models": ["not-a-model"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_requires_admin(self):
        self.client.login(email=self.member_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_transcription_models": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 403)

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"allowed_transcription_models": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)

    def test_empty_list_disables_transcription(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"allowed_transcription_models": []}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["allowed_transcription_models"], [])


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
)
class OrgTranscriptionModelUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.admin_user = User.objects.create_user(
            email="txdefault-admin@example.com", password=self.password,
        )
        self.admin_user.email_verified = True
        self.admin_user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="TxDefOrg", slug="txdeforg")
        Membership.objects.create(user=self.admin_user, org=self.org, role=Membership.Role.ADMIN)
        self.url = reverse("accounts:org_transcription_model_update")

    def test_admin_sets_default(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "openai/gpt-4o-transcribe"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertEqual(self.org.preferences["transcription_models"]["default"], "openai/gpt-4o-transcribe")

    def test_admin_clears_default(self):
        self.client.login(email=self.admin_user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": ""}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.org.refresh_from_db()
        self.assertIsNone(self.org.preferences["transcription_models"]["default"])

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"model": "openai/gpt-4o-transcribe"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)


@override_settings(
    ALLOWED_HOSTS=["testserver"],
    TRANSCRIPTION_ALLOWED_MODELS=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
    TRANSCRIPTION_DEFAULT_MODEL="openai/gpt-4o-mini-transcribe",
)
class PreferencesTranscriptionModelUpdateTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="txuser@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:preferences_transcription_model_update")

    @patch("core.preferences.get_preferences")
    def test_user_sets_transcription_model(self, mock_prefs):
        from core.preferences import ResolvedPreferences
        mock_prefs.return_value = ResolvedPreferences(
            top_model="openai/gpt-5",
            mid_model="",
            cheap_model="",
            allowed_models=[],
            allowed_tools=[],
            theme="light",
            allowed_transcription_models=["openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"],
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "openai/gpt-4o-transcribe"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        settings = UserSettings.objects.get(user=self.user)
        self.assertEqual(settings.preferences["transcription_models"]["default"], "openai/gpt-4o-transcribe")

    @patch("core.preferences.get_preferences")
    def test_reject_model_not_allowed(self, mock_prefs):
        from core.preferences import ResolvedPreferences
        mock_prefs.return_value = ResolvedPreferences(
            top_model="openai/gpt-5",
            mid_model="",
            cheap_model="",
            allowed_models=[],
            allowed_tools=[],
            theme="light",
            allowed_transcription_models=["openai/gpt-4o-mini-transcribe"],
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": "openai/gpt-4o-transcribe"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_clear_model(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"model": ""}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        settings = UserSettings.objects.get(user=self.user)
        self.assertIsNone(settings.preferences["transcription_models"]["default"])

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"model": "openai/gpt-4o-transcribe"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)


class MeetingSummarizerSkillPrefTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="msp@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:preferences_meeting_summarizer_skill_update")
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        self.skill = AgentSkill.objects.create(
            slug="meeting-summarizer",
            name="Meeting Summarizer",
            instructions="x",
            level="system",
            tool_names=["save_meeting_minutes"],
        )

    def test_set_default_skill(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"skill_id": str(self.skill.id)}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])
        us = UserSettings.objects.get(user=self.user)
        self.assertEqual(
            us.preferences["meetings"]["summarizer_skill_id"],
            str(self.skill.id),
        )

    def test_clear_default_skill(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"skill_id": ""}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        us = UserSettings.objects.get(user=self.user)
        self.assertIsNone(us.preferences["meetings"]["summarizer_skill_id"])

    def test_rejects_ineligible_skill(self):
        other = AgentSkill.objects.create(
            slug="other", name="Other", instructions="x",
            level="system", tool_names=["some_tool"],
        )
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"skill_id": str(other.id)}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_nonexistent_skill(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url,
            json.dumps({"skill_id": "00000000-0000-0000-0000-000000000000"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_requires_login(self):
        response = self.client.post(
            self.url,
            json.dumps({"skill_id": str(self.skill.id)}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)


@override_settings(ALLOWED_HOSTS=["testserver"])
class AgentAttachSkillsPreferenceViewTests(TestCase):
    def setUp(self):
        self.password = "test-pass-123"
        self.user = User.objects.create_user(
            email="attachpref@example.com", password=self.password,
        )
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.url = reverse("accounts:preferences_agent_attach_skills_update")

    def test_requires_login(self):
        response = self.client.post(
            self.url, json.dumps({"enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 302)

    def test_rejects_get(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 405)

    def test_set_false(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({"enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["enabled"])
        settings_obj = UserSettings.objects.get(user=self.user)
        self.assertFalse(settings_obj.preferences["allow_agent_attach_skills"])

    def test_set_true(self):
        self.client.login(email=self.user.email, password=self.password)
        # First disable so we can observe a re-enable
        self.client.post(
            self.url, json.dumps({"enabled": False}),
            content_type="application/json",
        )
        response = self.client.post(
            self.url, json.dumps({"enabled": True}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["enabled"])
        settings_obj = UserSettings.objects.get(user=self.user)
        self.assertTrue(settings_obj.preferences["allow_agent_attach_skills"])

    def test_invalid_json_returns_400(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url, "not-json", content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_missing_enabled_defaults_true(self):
        self.client.login(email=self.user.email, password=self.password)
        response = self.client.post(
            self.url, json.dumps({}), content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["enabled"])
