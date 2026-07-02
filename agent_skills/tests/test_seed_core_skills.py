"""Seeding checks for the five core capability skills (canvas / image / loop /
data room / web) that gate always-on tool groups behind a main-agent skill."""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from core.preferences import get_preferences

User = get_user_model()


class SeedCoreSkillsTest(TestCase):
    """Each core skill seeds at level="system", audience="main", carrying its
    tool group. (Seeding runs via the agent_skills post_migrate hook, so the
    rows exist in the test DB without explicit creation.)"""

    def _get(self, slug):
        return AgentSkill.objects.filter(slug=slug, level="system").first()

    def test_image_generator_seeded(self):
        skill = self._get("image_generator")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        self.assertIn("chat_generate_image", skill.tool_names)

    def test_assistant_loop_tools_seeded(self):
        skill = self._get("assistant_loop_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in ("chat_loop_create", "chat_loop_list", "chat_loop_edit", "chat_loop_stop"):
            self.assertIn(tool, skill.tool_names)

    def test_canvas_collaborator_seeded(self):
        skill = self._get("canvas_collaborator")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in (
            "canvas_write", "canvas_edit", "canvas_activate", "canvas_delete",
            "canvas_save_to_document",
        ):
            self.assertIn(tool, skill.tool_names)

    def test_data_room_tools_seeded(self):
        skill = self._get("data_room_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in ("document_search", "document_read", "document_edit", "document_status"):
            self.assertIn(tool, skill.tool_names)

    def test_web_research_tools_seeded(self):
        skill = self._get("web_research_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in ("web_search", "web_fetch", "web_search_read"):
            self.assertIn(tool, skill.tool_names)


class ImageGeneratorGateTest(TestCase):
    """The image-model gate moved from allowed_tools into skill-tool resolution:
    chat_generate_image survives in the enabled image_generator skill only when
    an image model resolves."""

    def setUp(self):
        self.user = User.objects.create_user(email="imggate@example.com", password="pw")
        self.org = Organization.objects.create(
            name="ImgOrg", slug="imgorg",
            preferences={"skills": {"image_generator": {"enabled": True}}},
        )
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

    def _image_tools(self, prefs):
        entry = next(
            (s for s in prefs.allowed_skills if s["slug"] == "image_generator"), None
        )
        self.assertIsNotNone(entry, "image_generator should be enabled and visible")
        return entry["tool_names"]

    @override_settings(
        IMAGE_ALLOWED_MODELS=["openai/gpt-image-1"],
        IMAGE_DEFAULT_MODEL="openai/gpt-image-1",
    )
    def test_tool_present_when_image_model_resolves(self):
        prefs = get_preferences(self.user)
        self.assertIn("chat_generate_image", self._image_tools(prefs))

    @override_settings(IMAGE_ALLOWED_MODELS=[], IMAGE_DEFAULT_MODEL="")
    def test_tool_stripped_when_no_image_model(self):
        prefs = get_preferences(self.user)
        self.assertNotIn("chat_generate_image", self._image_tools(prefs))
