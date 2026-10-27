"""A skill's Automatic model preference (category + reasoning level)."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from agent_skills.services import (
    clean_model_prefs,
    export_skill,
    fork_skill,
    import_skill,
    parse_skill_export,
)

User = get_user_model()


class CleanModelPrefsTests(TestCase):
    def test_validates_category_and_level(self):
        self.assertEqual(clean_model_prefs("writing", "high"), ("writing", "high"))
        self.assertEqual(clean_model_prefs("writing", "bogus"), ("writing", ""))
        # No level without a category; unknown categories clear both.
        self.assertEqual(clean_model_prefs("nope", "high"), ("", ""))
        self.assertEqual(clean_model_prefs("", "high"), ("", ""))
        self.assertEqual(clean_model_prefs(None, None), ("", ""))

    def test_seed_preferences_are_valid(self):
        from agent_skills.seed_skills import SYSTEM_SKILLS

        seeded = {s["slug"]: s.get("preferred_model_category", "") for s in SYSTEM_SKILLS}
        for slug, category in seeded.items():
            self.assertEqual(clean_model_prefs(category, "")[0], category, slug)
        self.assertEqual(seeded["researcher"], "light_analysis")
        self.assertEqual(seeded["meeting-summarizer"], "writing")


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillModelPrefsTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="modelprefs@example.com", password="pass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Org")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)
        self.skill = AgentSkill.objects.create(
            slug="writer", name="Writer", instructions="x", level="user", created_by=self.user,
        )
        self.client.force_login(self.user)

    def _save(self, **fields):
        return self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
            {"action": "save", "name": "Writer", "instructions": "x", **fields},
        )

    def test_detail_page_offers_categories(self):
        response = self.client.get(reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id}))
        self.assertTrue(response.context["show_model"])
        keys = [c["key"] for c in response.context["model_categories"]]
        self.assertIn("writing", keys)
        self.assertContains(response, 'id="skill-model-category"')

    def test_toolkit_has_no_model_section(self):
        toolkit = AgentSkill.objects.create(
            slug="canvas_tools", name="Canvas Tools", instructions="i", level="system", kind="toolkit",
        )
        self.org.preferences = {"skills": {"canvas_tools": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])
        response = self.client.get(reverse("agent_skills_detail", kwargs={"skill_id": toolkit.id}))
        self.assertFalse(response.context["show_model"])

    def test_save_sets_validates_and_leaves_alone(self):
        self._save(preferred_model_category="writing", preferred_reasoning_level="high")
        self.skill.refresh_from_db()
        self.assertEqual((self.skill.preferred_model_category, self.skill.preferred_reasoning_level), ("writing", "high"))

        # A save without the field (read-only select, other forms) changes nothing.
        self._save()
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.preferred_model_category, "writing")

        self._save(preferred_model_category="bogus", preferred_reasoning_level="high")
        self.skill.refresh_from_db()
        self.assertEqual((self.skill.preferred_model_category, self.skill.preferred_reasoning_level), ("", ""))

    def test_fork_and_export_import_carry_the_preference(self):
        self.skill.preferred_model_category = "slide_design"
        self.skill.preferred_reasoning_level = "max"
        self.skill.save()

        copy = fork_skill(self.user, self.skill, rescan=False)
        self.assertEqual((copy.preferred_model_category, copy.preferred_reasoning_level), ("slide_design", "max"))

        data = export_skill(self.skill)
        self.assertEqual(data["preferred_model_category"], "slide_design")
        [payload] = parse_skill_export(json.dumps({"skills": [data]}))
        imported = import_skill(self.user, payload)
        self.assertEqual((imported.preferred_model_category, imported.preferred_reasoning_level), ("slide_design", "max"))
