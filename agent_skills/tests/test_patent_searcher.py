"""Tests for the patent-searcher seed subagent skill and its tool resolution.

The EPO tools only auto-register when OPS credentials are set, so these tests
register them into the process-wide registry in setUp and restore it in
tearDown — otherwise skill-tool filtering (which checks the registry) would drop
them.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.test import TestCase

from agent_skills.models import AgentSkill
from agent_skills.services import (
    filter_to_skill_tools,
    get_available_skills,
    get_subagent_skills,
)
from core.preferences import get_preferences
from llm.tools.epo_ops import (
    PatentEpoOpsFamilyTool,
    PatentEpoOpsGetTool,
    PatentEpoOpsSearchTool,
)
from llm.tools.registry import get_tool_registry

User = get_user_model()

_PATENT_TOOLS = ["patent_epoops_search", "patent_epoops_get", "patent_epoops_family"]


class PatentSearcherSeedTests(TestCase):
    def setUp(self):
        from accounts.models import Membership, Organization

        self.registry = get_tool_registry()
        self._added: list[str] = []
        for cls in (PatentEpoOpsSearchTool, PatentEpoOpsGetTool, PatentEpoOpsFamilyTool):
            tool = cls()
            self.registry.register_tool(tool)
            self._added.append(tool.name)
        self.user = User.objects.create_user(email="patent@example.com", password="pw")
        # System seed skills are opt-in per org: an org admin enables them via
        # org.preferences["skills"][slug]["enabled"] (see core.preferences
        # _resolve_skill_entries). Enable patent-searcher so it resolves into
        # this member's specializations.
        self.org = Organization.objects.create(
            name="Org",
            slug="org-patent",
            preferences={"skills": {"patent-searcher": {"enabled": True}}},
        )
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)

    def tearDown(self):
        for name in self._added:
            self.registry._tools.pop(name, None)

    def test_seeded_as_subagent(self):
        skill = AgentSkill.objects.filter(slug="patent-searcher", level="system").first()
        self.assertIsNotNone(skill)
        self.assertEqual(skill.audience, "subagent")
        for name in _PATENT_TOOLS + ["skill_template_view"]:
            self.assertIn(name, skill.tool_names)
        self.assertTrue(skill.templates.filter(name="Patent Search Report").exists())

    def test_filter_keeps_patent_tools_for_subagent(self):
        kept = filter_to_skill_tools(
            _PATENT_TOOLS + ["skill_template_view"], skill_audience="subagent"
        )
        for name in _PATENT_TOOLS + ["skill_template_view"]:
            self.assertIn(name, kept)

    def test_surfaces_as_subagent_specialization_only(self):
        sub_slugs = {s.slug for s in get_subagent_skills(self.user)}
        main_slugs = {s.slug for s in get_available_skills(self.user)}
        self.assertIn("patent-searcher", sub_slugs)
        self.assertNotIn("patent-searcher", main_slugs)

    def test_preferences_specialization_carries_tools(self):
        prefs = get_preferences(self.user)
        specs = {s["slug"]: s for s in prefs.allowed_specializations}
        self.assertIn("patent-searcher", specs)
        for name in _PATENT_TOOLS:
            self.assertIn(name, specs["patent-searcher"]["tool_names"])
        # Must not leak into the main-agent skill list.
        self.assertNotIn("patent-searcher", {s["slug"] for s in prefs.allowed_skills})

    def test_resolve_subagent_tools_includes_patent_tools(self):
        from chat.subagent_service import resolve_subagent_tools

        prefs = get_preferences(self.user)
        tools = resolve_subagent_tools(prefs, [], "patent-searcher")
        for name in _PATENT_TOOLS:
            self.assertIn(name, tools)
