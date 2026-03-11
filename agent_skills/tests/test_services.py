"""Tests for agent_skills.services."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from agent_skills.services import get_available_skills, get_skill_for_user

User = get_user_model()


class GetAvailableSkillsTests(TestCase):
    def setUp(self):
        # Remove seeded skills so tests have a clean slate
        AgentSkill.objects.all().delete()

        self.user = User.objects.create_user(email="user@example.com", password="pass")
        self.org = Organization.objects.create(name="My Org", slug="my-org")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

        self.other_user = User.objects.create_user(email="other@example.com", password="pass")

    def test_system_skills_visible_to_all(self):
        AgentSkill.objects.create(
            slug="sys-skill", name="System Skill",
            instructions="Inst.", level="system",
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].slug, "sys-skill")

    def test_org_skills_visible_to_members(self):
        AgentSkill.objects.create(
            slug="org-skill", name="Org Skill",
            instructions="Inst.", level="org", organization=self.org,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)

        # Non-member should not see org skill
        skills_other = get_available_skills(self.other_user)
        self.assertEqual(len(skills_other), 0)

    def test_user_skills_private(self):
        AgentSkill.objects.create(
            slug="user-skill", name="User Skill",
            instructions="Inst.", level="user", created_by=self.user,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)

        skills_other = get_available_skills(self.other_user)
        self.assertEqual(len(skills_other), 0)

    def test_shadowing_user_overrides_org(self):
        AgentSkill.objects.create(
            slug="shared", name="Org Version",
            instructions="Org inst.", level="org", organization=self.org,
        )
        AgentSkill.objects.create(
            slug="shared", name="User Version",
            instructions="User inst.", level="user", created_by=self.user,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "User Version")

    def test_shadowing_org_overrides_system(self):
        AgentSkill.objects.create(
            slug="shared", name="System Version",
            instructions="Sys inst.", level="system",
        )
        AgentSkill.objects.create(
            slug="shared", name="Org Version",
            instructions="Org inst.", level="org", organization=self.org,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "Org Version")

    def test_shadowing_user_overrides_system(self):
        AgentSkill.objects.create(
            slug="shared", name="System Version",
            instructions="Sys inst.", level="system",
        )
        AgentSkill.objects.create(
            slug="shared", name="User Version",
            instructions="User inst.", level="user", created_by=self.user,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].name, "User Version")

    def test_inactive_skills_excluded(self):
        AgentSkill.objects.create(
            slug="inactive", name="Inactive",
            instructions="Inst.", level="system", is_active=False,
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 0)

    def test_sorted_by_name(self):
        AgentSkill.objects.create(
            slug="b-skill", name="Bravo",
            instructions="Inst.", level="system",
        )
        AgentSkill.objects.create(
            slug="a-skill", name="Alpha",
            instructions="Inst.", level="system",
        )
        skills = get_available_skills(self.user)
        self.assertEqual([s.name for s in skills], ["Alpha", "Bravo"])


class GetSkillForUserTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="pass")
        self.other_user = User.objects.create_user(email="other@example.com", password="pass")
        self.org = Organization.objects.create(name="My Org", slug="my-org")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def test_system_skill_accessible_by_anyone(self):
        skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="Inst.", level="system",
        )
        self.assertIsNotNone(get_skill_for_user(self.user, str(skill.pk)))
        self.assertIsNotNone(get_skill_for_user(self.other_user, str(skill.pk)))

    def test_org_skill_accessible_by_member(self):
        skill = AgentSkill.objects.create(
            slug="org", name="Org", instructions="Inst.",
            level="org", organization=self.org,
        )
        self.assertIsNotNone(get_skill_for_user(self.user, str(skill.pk)))
        self.assertIsNone(get_skill_for_user(self.other_user, str(skill.pk)))

    def test_user_skill_accessible_by_creator(self):
        skill = AgentSkill.objects.create(
            slug="usr", name="Usr", instructions="Inst.",
            level="user", created_by=self.user,
        )
        self.assertIsNotNone(get_skill_for_user(self.user, str(skill.pk)))
        self.assertIsNone(get_skill_for_user(self.other_user, str(skill.pk)))

    def test_nonexistent_skill_returns_none(self):
        import uuid
        self.assertIsNone(get_skill_for_user(self.user, str(uuid.uuid4())))

    def test_inactive_skill_returns_none(self):
        skill = AgentSkill.objects.create(
            slug="off", name="Off", instructions="Inst.",
            level="system", is_active=False,
        )
        self.assertIsNone(get_skill_for_user(self.user, str(skill.pk)))
