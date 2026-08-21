"""Lightweight regression tests for the slide-deck consumer + skill wiring."""

from __future__ import annotations

from django.test import SimpleTestCase


class SlidesUpdatedToolsMembershipTests(SimpleTestCase):
    def test_deck_mutating_tools_are_listed(self):
        """Any deck-mutating tool MUST be in SLIDES_UPDATED_TOOLS or its deck never
        refreshes in the UI (mirror of the CANVAS_UPDATED_TOOLS assertion)."""
        from chat.consumers import SLIDES_UPDATED_TOOLS

        for name in ("slide_canvas_write", "slide_canvas_edit", "slides_add_slide"):
            self.assertIn(name, SLIDES_UPDATED_TOOLS)


class SkillWiringTests(SimpleTestCase):
    def test_skill_tools_registered_and_audience_compatible(self):
        from agent_skills.seed_skills.slide_deck_collaborator import SLIDE_DECK_COLLABORATOR
        from llm.tools import get_tool_registry

        reg = get_tool_registry()
        for name in SLIDE_DECK_COLLABORATOR["tool_names"]:
            tool = reg.get_tool(name)
            self.assertIsNotNone(tool, f"{name} not registered")
            self.assertEqual(tool.section, "skills", name)
            self.assertIn(tool.audience, ("main", "shared"), name)

    def test_skill_in_system_seed_list(self):
        from agent_skills.seed_skills import SYSTEM_SKILLS

        slugs = {s["slug"] for s in SYSTEM_SKILLS}
        self.assertIn("slide_deck_collaborator", slugs)
