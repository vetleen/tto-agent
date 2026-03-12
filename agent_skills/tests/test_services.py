"""Tests for agent_skills.services."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    can_edit_skill,
    create_user_skill,
    fork_skill,
    get_available_skills,
    get_editable_skill_for_user,
    get_skill_for_user,
)

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


class CanEditSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="editor@example.com", password="pass")
        self.other_user = User.objects.create_user(email="other@example.com", password="pass")
        self.org = Organization.objects.create(name="Edit Org", slug="edit-org")
        self.admin_membership = Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN,
        )

    def test_system_skill_not_editable(self):
        skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="Inst.", level="system",
        )
        self.assertFalse(can_edit_skill(self.user, skill))

    def test_org_skill_editable_by_admin(self):
        skill = AgentSkill.objects.create(
            slug="org", name="Org", instructions="Inst.",
            level="org", organization=self.org,
        )
        self.assertTrue(can_edit_skill(self.user, skill))

    def test_org_skill_not_editable_by_non_admin(self):
        member = User.objects.create_user(email="member@example.com", password="pass")
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        skill = AgentSkill.objects.create(
            slug="org", name="Org", instructions="Inst.",
            level="org", organization=self.org,
        )
        self.assertFalse(can_edit_skill(member, skill))

    def test_user_skill_editable_by_creator(self):
        skill = AgentSkill.objects.create(
            slug="usr", name="Usr", instructions="Inst.",
            level="user", created_by=self.user,
        )
        self.assertTrue(can_edit_skill(self.user, skill))

    def test_user_skill_not_editable_by_other(self):
        skill = AgentSkill.objects.create(
            slug="usr", name="Usr", instructions="Inst.",
            level="user", created_by=self.user,
        )
        self.assertFalse(can_edit_skill(self.other_user, skill))


class CreateUserSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="creator@example.com", password="pass")

    def test_creates_with_auto_slug(self):
        skill = create_user_skill(self.user, "My Cool Skill")
        self.assertEqual(skill.slug, "my-cool-skill")
        self.assertEqual(skill.name, "My Cool Skill")
        self.assertEqual(skill.level, "user")
        self.assertEqual(skill.created_by, self.user)
        self.assertEqual(skill.instructions, "")
        self.assertEqual(skill.description, "")

    def test_deduplicates_slug(self):
        create_user_skill(self.user, "Dupe Skill")
        skill2 = create_user_skill(self.user, "Dupe Skill")
        self.assertEqual(skill2.slug, "dupe-skill-1")

    def test_custom_slug(self):
        skill = create_user_skill(self.user, "Custom", slug="custom-slug")
        self.assertEqual(skill.slug, "custom-slug")


class GetEditableSkillForUserTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="ed@example.com", password="pass")
        self.org = Organization.objects.create(name="Ed Org", slug="ed-org")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.ADMIN)

    def test_returns_editable_user_skill(self):
        AgentSkill.objects.create(
            slug="my-skill", name="Mine", instructions="Inst.",
            level="user", created_by=self.user,
        )
        result = get_editable_skill_for_user(self.user, "my-skill")
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "Mine")

    def test_returns_none_for_system_skill(self):
        AgentSkill.objects.create(
            slug="sys-skill", name="Sys", instructions="Inst.", level="system",
        )
        result = get_editable_skill_for_user(self.user, "sys-skill")
        self.assertIsNone(result)

    def test_returns_none_for_nonexistent(self):
        result = get_editable_skill_for_user(self.user, "no-such-skill")
        self.assertIsNone(result)


class ForkSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="fork@example.com", password="pass")

    def test_forks_skill_with_templates(self):
        source = AgentSkill.objects.create(
            slug="source", name="Source", instructions="Do stuff.",
            description="A source skill.", level="system",
            tool_names=["search_documents"],
        )
        SkillTemplate.objects.create(skill=source, name="Template A", content="Content A")
        SkillTemplate.objects.create(skill=source, name="Template B", content="Content B")

        forked = fork_skill(self.user, source)

        self.assertEqual(forked.slug, "source")
        self.assertEqual(forked.name, "Source")
        self.assertEqual(forked.instructions, "Do stuff.")
        self.assertEqual(forked.description, "A source skill.")
        self.assertEqual(forked.tool_names, ["search_documents"])
        self.assertEqual(forked.level, "user")
        self.assertEqual(forked.created_by, self.user)
        self.assertEqual(forked.parent, source)

        templates = list(forked.templates.order_by("name"))
        self.assertEqual(len(templates), 2)
        self.assertEqual(templates[0].name, "Template A")
        self.assertEqual(templates[0].content, "Content A")

    def test_fork_deduplicates_slug(self):
        source = AgentSkill.objects.create(
            slug="dup", name="Dup", instructions="Inst.", level="system",
        )
        fork1 = fork_skill(self.user, source)
        self.assertEqual(fork1.slug, "dup")
        fork2 = fork_skill(self.user, source)
        self.assertEqual(fork2.slug, "dup-1")
