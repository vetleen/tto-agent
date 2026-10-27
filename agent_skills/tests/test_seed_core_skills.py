"""Seeding checks for the toolkits (canvas / image / loop / data room / web /
slide deck) that gate tool groups behind a main-agent skill."""

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

    def test_image_generation_tools_seeded(self):
        skill = self._get("image_generation_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        self.assertIn("chat_generate_image", skill.tool_names)

    def test_assistant_loop_tools_seeded(self):
        skill = self._get("assistant_loop_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in ("chat_loop_create", "chat_loop_list", "chat_loop_edit", "chat_loop_stop"):
            self.assertIn(tool, skill.tool_names)

    def test_canvas_tools_seeded(self):
        skill = self._get("canvas_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        for tool in (
            "canvas_write", "canvas_edit", "canvas_read", "canvas_delete",
            "canvas_save_to_document",
        ):
            self.assertIn(tool, skill.tool_names)
        # Activation is gone: canvases are listed and read on demand.
        self.assertNotIn("canvas_activate", skill.tool_names)
        self.assertIn("# Reading canvases", skill.instructions)
        self.assertIn("canvas_name", skill.instructions)

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

    def test_skill_management_tools_seeded(self):
        skill = self._get("skill_management_tools")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "main")
        self.assertEqual(skill.kind, AgentSkill.Kind.TOOLKIT)
        for tool in (
            "skill_create", "skill_edit", "skill_delete", "skill_field_save",
            "skill_field_load", "skill_resource_list", "skill_resource_save",
            "skill_resource_attach", "skill_resource_update", "skill_resource_delete",
            "skill_tool_list", "skill_tool_inspect",
        ):
            self.assertIn(tool, skill.tool_names)

    def test_skill_creator_requires_toolkits_not_tools(self):
        """Skill Creator gets the skill tools through the toolkit, not tool_names."""
        skill = self._get("skill-creator")
        self.assertEqual(skill.tool_names, [])
        self.assertEqual(skill.requires, ["canvas_tools", "skill_management_tools"])


class SeedGraphTests(TestCase):
    """Seed-level invariants of the toolkit / ``requires`` model: toolkits are
    main-audience tool unlocks, ``requires`` only names seed slugs on
    main-audience playbooks, and the dependency graph has no cycles."""

    def _seeds(self):
        from agent_skills.seed_skills import SYSTEM_SKILLS

        return SYSTEM_SKILLS

    def test_toolkit_seeds(self):
        toolkits = [s for s in self._seeds() if s.get("kind") == "toolkit"]
        self.assertEqual(
            sorted(s["slug"] for s in toolkits),
            sorted([
                "assistant_loop_tools", "canvas_tools", "data_room_tools",
                "image_generation_tools", "web_research_tools", "slide_deck_tools",
                "skill_management_tools",
            ]),
        )
        for data in toolkits:
            self.assertTrue(data["slug"].endswith("_tools"), data["slug"])
            self.assertEqual(data.get("audience", "main"), "main", data["slug"])
            self.assertTrue(data["tool_names"], data["slug"])
            self.assertFalse(data.get("requires"), data["slug"])
            row = AgentSkill.objects.get(slug=data["slug"], level="system")
            self.assertEqual(row.kind, AgentSkill.Kind.TOOLKIT, data["slug"])

    def test_playbook_seeds_default_kind(self):
        for data in self._seeds():
            if data.get("kind") == "toolkit":
                continue
            row = AgentSkill.objects.get(slug=data["slug"], level="system")
            self.assertEqual(row.kind, AgentSkill.Kind.PLAYBOOK, data["slug"])

    def test_requires_name_seed_slugs_on_main_playbooks(self):
        seeds = self._seeds()
        slugs = {s["slug"] for s in seeds}
        for data in seeds:
            requires = data.get("requires") or []
            for req in requires:
                self.assertIn(req, slugs, f"{data['slug']} requires unknown {req}")
                self.assertNotEqual(req, data["slug"])
            if requires:
                self.assertEqual(data.get("audience", "main"), "main", data["slug"])
            row = AgentSkill.objects.get(slug=data["slug"], level="system")
            self.assertEqual(row.requires, list(requires), data["slug"])

    def test_canvas_drafting_seeds_require_canvas_tools(self):
        for slug in ("skill-creator", "web-deep-researcher-2", "web-deep-researcher",
                     "meeting-summarizer"):
            row = AgentSkill.objects.get(slug=slug, level="system")
            self.assertIn("canvas_tools", row.requires, slug)
        for slug in ("researcher", "web-researcher", "patent-searcher"):
            row = AgentSkill.objects.get(slug=slug, level="system")
            self.assertEqual(row.requires, [], slug)

    def test_specialization_availability(self):
        # Generic specializations are always listed; the patent searcher only
        # through a playbook that names it.
        for slug in ("researcher", "web-researcher"):
            row = AgentSkill.objects.get(slug=slug, level="system")
            self.assertTrue(row.always_available, slug)
        row = AgentSkill.objects.get(slug="patent-searcher", level="system")
        self.assertFalse(row.always_available)

    def test_seed_subagents_name_known_specializations(self):
        seeds = self._seeds()
        specs = {s["slug"] for s in seeds if s.get("audience") in ("subagent", "shared")}
        for data in seeds:
            for slug in data.get("subagents") or []:
                self.assertIn(slug, specs, f"{data['slug']} names unknown {slug}")
            if data.get("subagents"):
                self.assertNotEqual(data.get("audience", "main"), "subagent", data["slug"])
                self.assertNotEqual(data.get("kind", "playbook"), "toolkit", data["slug"])

    def test_requires_graph_is_acyclic(self):
        graph = {s["slug"]: list(s.get("requires") or []) for s in self._seeds()}

        def visit(slug, stack):
            self.assertNotIn(slug, stack, f"requires cycle through {slug}")
            for req in graph.get(slug, []):
                visit(req, stack + [slug])

        for slug in graph:
            visit(slug, [])


class ImageGeneratorGateTest(TestCase):
    """The image-model gate moved from allowed_tools into skill-tool resolution:
    chat_generate_image survives in the enabled image_generation_tools skill only
    when an image model resolves."""

    def setUp(self):
        self.user = User.objects.create_user(email="imggate@example.com", password="pw")
        self.org = Organization.objects.create(
            name="ImgOrg", slug="imgorg",
            preferences={"skills": {"image_generation_tools": {"enabled": True}}},
        )
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

    def _image_tools(self, prefs):
        entry = next(
            (s for s in prefs.allowed_skills if s["slug"] == "image_generation_tools"),
            None,
        )
        self.assertIsNotNone(entry, "image_generation_tools should be enabled and visible")
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
