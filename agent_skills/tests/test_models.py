"""Tests for AgentSkill model."""

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Organization
from agent_skills.models import AgentSkill

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
