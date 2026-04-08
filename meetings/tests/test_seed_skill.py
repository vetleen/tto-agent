"""Test that the meeting summarizer system skill is registered and seeded."""
from __future__ import annotations

from django.test import TestCase

from agent_skills.models import AgentSkill
from agent_skills.seed_skills import SYSTEM_SKILLS, seed_system_skills


class MeetingSummarizerSeedTests(TestCase):
    def test_seed_list_includes_meeting_summarizer(self):
        slugs = {s["slug"] for s in SYSTEM_SKILLS}
        self.assertIn("meeting-summarizer", slugs)

    def test_seed_creates_system_skill(self):
        AgentSkill.objects.filter(slug="meeting-summarizer").delete()
        seed_system_skills()
        skill = AgentSkill.objects.get(slug="meeting-summarizer", level="system")
        self.assertEqual(skill.name, "Meeting Summarizer")
        self.assertEqual(skill.tool_names, ["save_meeting_minutes"])
        self.assertIn("transcript", skill.instructions.lower())

    def test_seed_is_idempotent(self):
        seed_system_skills()
        seed_system_skills()
        self.assertEqual(
            AgentSkill.objects.filter(slug="meeting-summarizer", level="system").count(),
            1,
        )
