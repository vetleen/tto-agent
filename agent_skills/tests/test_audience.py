"""Tests for the skill/tool audience axis and sub-agent specializations."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from agent_skills.models import AgentSkill
from agent_skills.services import (
    filter_to_skill_tools,
    fork_skill,
    get_available_skills,
    get_subagent_skills,
    skill_tool_audience_ok,
)
from core.preferences import get_preferences

User = get_user_model()


class SkillToolAudienceOkTests(TestCase):
    def test_shared_tool_allowed_everywhere(self):
        for skill_aud in ("main", "subagent", "shared"):
            self.assertTrue(skill_tool_audience_ok("shared", skill_aud))

    def test_same_audience_allowed(self):
        self.assertTrue(skill_tool_audience_ok("main", "main"))
        self.assertTrue(skill_tool_audience_ok("subagent", "subagent"))

    def test_cross_audience_rejected(self):
        self.assertFalse(skill_tool_audience_ok("main", "subagent"))
        self.assertFalse(skill_tool_audience_ok("subagent", "main"))

    def test_shared_skill_only_carries_shared_tools(self):
        self.assertTrue(skill_tool_audience_ok("shared", "shared"))
        self.assertFalse(skill_tool_audience_ok("main", "shared"))
        self.assertFalse(skill_tool_audience_ok("subagent", "shared"))


class FilterToSkillToolsAudienceTests(TestCase):
    def test_keeps_shared_tool_for_subagent_skill(self):
        # skill_template_view is a section="skills", audience="shared" tool.
        self.assertEqual(
            filter_to_skill_tools(["skill_template_view"], skill_audience="subagent"),
            ["skill_template_view"],
        )

    def test_drops_main_tool_for_subagent_skill(self):
        # skill_create is section="skills", audience="main".
        self.assertEqual(
            filter_to_skill_tools(["skill_create"], skill_audience="subagent"),
            [],
        )

    def test_no_audience_keeps_section_skills_tools(self):
        # Back-compat: without skill_audience, only the section gate applies.
        self.assertEqual(
            filter_to_skill_tools(["skill_create"]),
            ["skill_create"],
        )


class AudiencePartitioningTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="aud@example.com", password="pass")
        self.main_skill = AgentSkill.objects.create(
            slug="main-one", name="Main One", level="user",
            created_by=self.user, audience="main",
        )
        self.sub_skill = AgentSkill.objects.create(
            slug="sub-one", name="Sub One", level="user",
            created_by=self.user, audience="subagent",
        )
        self.shared_skill = AgentSkill.objects.create(
            slug="shared-one", name="Shared One", level="system",
            audience="shared",
        )

    def test_main_surface_excludes_subagent(self):
        slugs = {s.slug for s in get_available_skills(self.user)}
        self.assertIn("main-one", slugs)
        self.assertIn("shared-one", slugs)  # shared shows on both
        self.assertNotIn("sub-one", slugs)

    def test_subagent_surface_excludes_main(self):
        slugs = {s.slug for s in get_subagent_skills(self.user)}
        self.assertIn("sub-one", slugs)
        self.assertIn("shared-one", slugs)  # shared shows on both
        self.assertNotIn("main-one", slugs)


class ForkAudienceTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="fork@example.com", password="pass")

    def test_fork_inherits_source_audience(self):
        src = AgentSkill.objects.create(
            slug="sub-src", name="Sub Src", level="system", audience="subagent",
        )
        copy = fork_skill(self.user, src)
        self.assertEqual(copy.audience, "subagent")

    def test_fork_override_audience(self):
        src = AgentSkill.objects.create(
            slug="main-src", name="Main Src", level="system", audience="main",
        )
        copy = fork_skill(self.user, src, audience="subagent")
        self.assertEqual(copy.audience, "subagent")

    def test_fork_shared_source_drops_to_main(self):
        # A personal copy is never "shared" (seed-only).
        src = AgentSkill.objects.create(
            slug="shared-src", name="Shared Src", level="system", audience="shared",
        )
        copy = fork_skill(self.user, src)
        self.assertEqual(copy.audience, "main")


class SeedWebResearcherTests(TestCase):
    def test_web_researcher_seeded_as_subagent(self):
        skill = AgentSkill.objects.filter(slug="web-researcher", level="system").first()
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "subagent")
        # Carries skill_template_view (shared) and a report template.
        self.assertIn("skill_template_view", skill.tool_names)
        self.assertTrue(skill.templates.filter(name="Research Findings Report").exists())


class PreferenceAudienceTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="prefs@example.com", password="pass")

    def test_main_only_tool_excluded_from_subagent_bucket(self):
        prefs = get_preferences(self.user)
        # chat_subagent_create is audience="main" (always-on); chat_task_update is
        # shared. (canvas_write is now skills-gated, so no longer a good probe here.)
        self.assertIn("chat_subagent_create", prefs.allowed_tools)
        self.assertNotIn("chat_subagent_create", prefs.allowed_subagent_tools)
        self.assertIn("chat_task_update", prefs.allowed_tools)
        self.assertIn("chat_task_update", prefs.allowed_subagent_tools)

    def test_specialization_surfaced_with_filtered_tools(self):
        AgentSkill.objects.create(
            slug="my-spec", name="My Spec", level="user", created_by=self.user,
            audience="subagent",
            tool_names=["skill_template_view", "skill_create"],
        )
        prefs = get_preferences(self.user)
        specs = {s["slug"]: s for s in prefs.allowed_specializations}
        self.assertIn("my-spec", specs)
        # skill_template_view (shared) kept; skill_create (main) dropped.
        self.assertEqual(specs["my-spec"]["tool_names"], ["skill_template_view"])
        # And it must not leak into the main-agent skills list.
        self.assertNotIn("my-spec", {s["slug"] for s in prefs.allowed_skills})


@override_settings(ALLOWED_HOSTS=["testserver"])
class AudienceViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="audview@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.main_skill = AgentSkill.objects.create(
            slug="m-x", name="Main X", instructions="i", level="user",
            created_by=self.user, audience="main",
        )
        self.sub_skill = AgentSkill.objects.create(
            slug="s-x", name="Sub X", instructions="i", level="user",
            created_by=self.user, audience="subagent",
        )

    def test_list_partitions_by_audience(self):
        resp = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(resp.status_code, 200)
        main_slugs = {r["skill"].slug for r in resp.context["main_user_rows"]}
        sub_slugs = {r["skill"].slug for r in resp.context["sub_user_rows"]}
        self.assertIn("m-x", main_slugs)
        self.assertNotIn("s-x", main_slugs)
        self.assertIn("s-x", sub_slugs)
        self.assertNotIn("m-x", sub_slugs)

    def test_create_from_subagent_tab_sets_audience(self):
        resp = self.client.post(
            reverse("agent_skills_create"),
            {"name": "New Spec", "audience": "subagent"},
        )
        self.assertEqual(resp.status_code, 302)
        skill = AgentSkill.objects.get(name="New Spec")
        self.assertEqual(skill.audience, "subagent")

    def test_copy_from_subagent_tab_sets_audience(self):
        # Copy a (main) skill via the sub-agent tab → personal sub-agent skill.
        resp = self.client.post(
            reverse("agent_skills_copy", kwargs={"skill_id": self.main_skill.id}),
            {"audience": "subagent"},
        )
        self.assertEqual(resp.status_code, 302)
        copy = (
            AgentSkill.objects.filter(created_by=self.user)
            .exclude(pk__in=[self.main_skill.pk, self.sub_skill.pk])
            .order_by("-created_at")
            .first()
        )
        self.assertIsNotNone(copy)
        self.assertEqual(copy.audience, "subagent")
