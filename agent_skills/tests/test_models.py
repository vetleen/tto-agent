"""Tests for AgentSkill and SkillTemplate models."""

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Organization
from agent_skills.models import AgentSkill, SkillTemplate

User = get_user_model()


class AgentSkillModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="test@example.com", password="pass")
        self.org = Organization.objects.create(name="Test Org", slug="test-org")

    def test_create_system_skill(self):
        skill = AgentSkill.objects.create(
            slug="test-sys",
            name="Test System Skill",
            instructions="Do the thing.",
            level="system",
        )
        self.assertEqual(skill.level, "system")
        self.assertIsNone(skill.organization)
        self.assertIsNone(skill.created_by)

    def test_create_org_skill(self):
        skill = AgentSkill.objects.create(
            slug="test-org",
            name="Test Org Skill",
            instructions="Do the org thing.",
            level="org",
            organization=self.org,
        )
        self.assertEqual(skill.level, "org")
        self.assertEqual(skill.organization, self.org)

    def test_create_user_skill(self):
        skill = AgentSkill.objects.create(
            slug="test-user",
            name="Test User Skill",
            instructions="Do the user thing.",
            level="user",
            created_by=self.user,
        )
        self.assertEqual(skill.level, "user")
        self.assertEqual(skill.created_by, self.user)

    def test_str_output(self):
        skill = AgentSkill.objects.create(
            slug="demo",
            name="Demo Skill",
            instructions="Instructions.",
            level="system",
        )
        self.assertEqual(str(skill), "Demo Skill (System)")

    def test_tool_names_json_round_trip(self):
        skill = AgentSkill.objects.create(
            slug="tools-test",
            name="Tools Test",
            instructions="Instructions.",
            level="system",
            tool_names=["search_documents", "read_document"],
        )
        skill.refresh_from_db()
        self.assertEqual(skill.tool_names, ["search_documents", "read_document"])

    def test_tool_names_default_empty_list(self):
        skill = AgentSkill.objects.create(
            slug="no-tools",
            name="No Tools",
            instructions="Instructions.",
            level="system",
        )
        skill.refresh_from_db()
        self.assertEqual(skill.tool_names, [])

    def test_unique_system_slug(self):
        AgentSkill.objects.create(
            slug="unique-test",
            name="First",
            instructions="Instructions.",
            level="system",
        )
        # Duplicate system slug should fail
        with self.assertRaises(IntegrityError):
            AgentSkill.objects.create(
                slug="unique-test",
                name="Second",
                instructions="Instructions.",
                level="system",
            )

    def test_same_slug_different_levels(self):
        """Same slug can exist at different levels."""
        AgentSkill.objects.create(
            slug="multi-level",
            name="System",
            instructions="Sys.",
            level="system",
        )
        AgentSkill.objects.create(
            slug="multi-level",
            name="Org",
            instructions="Org.",
            level="org",
            organization=self.org,
        )
        AgentSkill.objects.create(
            slug="multi-level",
            name="User",
            instructions="User.",
            level="user",
            created_by=self.user,
        )
        self.assertEqual(AgentSkill.objects.filter(slug="multi-level").count(), 3)

    def test_is_active_default_true(self):
        skill = AgentSkill.objects.create(
            slug="active-test",
            name="Active",
            instructions="Instructions.",
            level="system",
        )
        self.assertTrue(skill.is_active)

    def test_description_blank(self):
        skill = AgentSkill.objects.create(
            slug="no-desc",
            name="No Desc",
            instructions="Instructions.",
            level="system",
            description="",
        )
        self.assertEqual(skill.description, "")


class SkillTemplateModelTests(TestCase):
    def setUp(self):
        self.skill = AgentSkill.objects.create(
            slug="tmpl-skill",
            name="Template Skill",
            instructions="Inst.",
            level="system",
        )

    def test_create_template(self):
        tmpl = SkillTemplate.objects.create(
            skill=self.skill,
            name="Patent Claim",
            content="1. A method comprising...",
        )
        tmpl.refresh_from_db()
        self.assertEqual(tmpl.name, "Patent Claim")
        self.assertEqual(tmpl.content, "1. A method comprising...")
        self.assertEqual(tmpl.skill, self.skill)

    def test_str_output(self):
        tmpl = SkillTemplate.objects.create(
            skill=self.skill, name="Report", content="",
        )
        self.assertIn("Report", str(tmpl))
        self.assertIn(self.skill.name, str(tmpl))

    def test_unique_constraint_per_skill(self):
        SkillTemplate.objects.create(
            skill=self.skill, name="Unique Name", content="A",
        )
        with self.assertRaises(IntegrityError):
            SkillTemplate.objects.create(
                skill=self.skill, name="Unique Name", content="B",
            )

    def test_same_name_different_skills(self):
        other_skill = AgentSkill.objects.create(
            slug="other-skill",
            name="Other Skill",
            instructions="Inst.",
            level="system",
        )
        SkillTemplate.objects.create(skill=self.skill, name="Shared", content="A")
        SkillTemplate.objects.create(skill=other_skill, name="Shared", content="B")
        self.assertEqual(SkillTemplate.objects.filter(name="Shared").count(), 2)

    def test_cascade_delete(self):
        SkillTemplate.objects.create(skill=self.skill, name="Doomed", content="X")
        self.assertEqual(SkillTemplate.objects.count(), 1)
        self.skill.delete()
        self.assertEqual(SkillTemplate.objects.count(), 0)

    def test_blank_content(self):
        tmpl = SkillTemplate.objects.create(
            skill=self.skill, name="Empty", content="",
        )
        self.assertEqual(tmpl.content, "")


class SeedSystemSkillsTests(TestCase):
    def test_seed_creates_all_system_skills(self):
        from agent_skills.seed import seed_system_skills

        seed_system_skills()

        self.assertTrue(
            AgentSkill.objects.filter(slug="skill-creator", level="system").exists()
        )
        self.assertTrue(
            AgentSkill.objects.filter(
                slug="written-assignment-writer", level="system"
            ).exists()
        )

    def test_seed_is_idempotent(self):
        from agent_skills.seed import seed_system_skills

        seed_system_skills()
        seed_system_skills()

        self.assertEqual(
            AgentSkill.objects.filter(level="system").count(), 2
        )

    def test_written_assignment_writer_fields(self):
        from agent_skills.seed import seed_system_skills

        seed_system_skills()

        skill = AgentSkill.objects.get(
            slug="written-assignment-writer", level="system"
        )
        self.assertEqual(skill.name, "Written Assignment Writer")
        self.assertIn("college-level", skill.description)
        self.assertIn("Decode the prompt", skill.instructions)
        self.assertEqual(skill.tool_names, [])
