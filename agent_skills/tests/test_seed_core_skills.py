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
        for tool in ("web_search", "web_fetch", "web_search_read", "web_image_view"):
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
        IMAGE_ALLOWED_MODELS=["gemini/gemini-3.1-flash-lite-image"],
        IMAGE_DEFAULT_MODEL="gemini/gemini-3.1-flash-lite-image",
    )
    def test_tool_present_when_image_model_resolves(self):
        prefs = get_preferences(self.user)
        self.assertIn("chat_generate_image", self._image_tools(prefs))

    @override_settings(IMAGE_ALLOWED_MODELS=[], IMAGE_DEFAULT_MODEL="")
    def test_tool_stripped_when_no_image_model(self):
        prefs = get_preferences(self.user)
        self.assertNotIn("chat_generate_image", self._image_tools(prefs))


class SeedTextTemplatesTest(TestCase):
    """Seeded text templates are first-party content: READY on creation (never
    scanned, like seeded files), hashed and token-counted like any other text
    resource, and repaired on re-seed if an older loader left a row PENDING —
    a PENDING row renders as "still processing" in the skill's resource list."""

    def test_seeded_templates_are_ready(self):
        from agent_skills.models import SkillResource
        from agent_skills.seed_skills import SYSTEM_SKILLS

        seeded = [s for s in SYSTEM_SKILLS if s.get("templates")]
        self.assertTrue(seeded)
        for data in seeded:
            skill = AgentSkill.objects.get(slug=data["slug"], level="system")
            rows = list(skill.templates.filter(original_filename=""))
            self.assertEqual(len(rows), len(data["templates"]), data["slug"])
            for tmpl in rows:
                self.assertEqual(tmpl.status, SkillResource.Status.READY, tmpl.name)
                self.assertTrue(tmpl.content_sha256, tmpl.name)
                self.assertGreater(tmpl.token_count, 0, tmpl.name)

    def test_reseed_repairs_pending_template(self):
        from agent_skills.models import SkillResource
        from agent_skills.seed_skills import seed_system_skills

        skill = AgentSkill.objects.get(slug="patent-searcher", level="system")
        tmpl = skill.templates.get(name="Patent Search Report (general)")
        SkillResource.objects.filter(pk=tmpl.pk).update(
            status=SkillResource.Status.PENDING, content_sha256="", token_count=0,
        )

        seed_system_skills()

        tmpl.refresh_from_db()
        self.assertEqual(tmpl.status, SkillResource.Status.READY)
        self.assertTrue(tmpl.content_sha256)
        self.assertGreater(tmpl.token_count, 0)
