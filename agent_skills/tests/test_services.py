"""Tests for agent_skills.services."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization, UserSettings
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    can_edit_skill,
    create_org_skill,
    create_user_skill,
    fork_skill,
    get_available_skills,
    get_editable_skill_for_user,
    get_skill_for_user,
    promote_skill_to_org,
    set_user_skill_selection,
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

    def test_fork_name_suffix_starts_with_paren_one(self):
        source = AgentSkill.objects.create(
            slug="research", name="Deep Research",
            instructions="Inst.", level="system",
        )
        first = fork_skill(self.user, source)
        self.assertEqual(first.name, "Deep Research")
        second = fork_skill(self.user, source)
        self.assertEqual(second.name, "Deep Research (1)")
        third = fork_skill(self.user, source)
        self.assertEqual(third.name, "Deep Research (2)")

    def test_fork_name_suffix_skips_used_numbers(self):
        source = AgentSkill.objects.create(
            slug="research", name="Deep Research",
            instructions="Inst.", level="system",
        )
        # Pre-seed with an existing user skill named "Deep Research (3)".
        AgentSkill.objects.create(
            slug="other", name="Deep Research (3)",
            instructions="Inst.", level="user", created_by=self.user,
        )
        first = fork_skill(self.user, source)
        self.assertEqual(first.name, "Deep Research")
        second = fork_skill(self.user, source)
        self.assertEqual(second.name, "Deep Research (1)")
        third = fork_skill(self.user, source)
        self.assertEqual(third.name, "Deep Research (2)")
        fourth = fork_skill(self.user, source)
        self.assertEqual(fourth.name, "Deep Research (4)")  # 3 already used

    def test_fork_user_skill_inherits_grandparent(self):
        org = Organization.objects.create(name="O", slug="o")
        Membership.objects.create(user=self.user, org=org)
        org_source = AgentSkill.objects.create(
            slug="rcn", name="RCN", instructions="Inst.",
            level="org", organization=org,
        )
        # Copy org skill → user skill (parent should be org_source).
        first = fork_skill(self.user, org_source)
        self.assertEqual(first.parent, org_source)

        # Copy that user skill → another user skill. Parent should be the
        # grandparent (org_source), not the intermediate user copy.
        second = fork_skill(self.user, first)
        self.assertEqual(second.parent, org_source)

    def test_fork_user_skill_with_no_parent_keeps_none(self):
        original = create_user_skill(self.user, "Hand-rolled")
        copied = fork_skill(self.user, original)
        self.assertIsNone(copied.parent)


class GetAvailableSkillsWithSelectionTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="sel@example.com", password="pass")
        self.org = Organization.objects.create(name="Sel Org", slug="sel-org")
        Membership.objects.create(user=self.user, org=self.org)

    def _set_pref(self, slug, selected_skill_id):
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        prefs = us.preferences or {}
        skills = prefs.get("skills") or {}
        skills[slug] = {"selected_skill_id": selected_skill_id}
        prefs["skills"] = skills
        us.preferences = prefs
        us.save(update_fields=["preferences"])

    def test_selection_overrides_shadowing(self):
        org_skill = AgentSkill.objects.create(
            slug="research", name="Org Research", instructions="o",
            level="org", organization=self.org,
        )
        user_skill = AgentSkill.objects.create(
            slug="research", name="My Research", instructions="u",
            level="user", created_by=self.user,
        )
        # Default: user skill wins (shadowing).
        self.assertEqual(get_available_skills(self.user)[0].pk, user_skill.pk)
        # Explicitly select the org version.
        self._set_pref("research", str(org_skill.id))
        self.assertEqual(get_available_skills(self.user)[0].pk, org_skill.pk)

    def test_explicit_disable_hides_skill(self):
        AgentSkill.objects.create(
            slug="research", name="Org Research", instructions="o",
            level="org", organization=self.org,
        )
        self._set_pref("research", None)
        self.assertEqual(get_available_skills(self.user), [])

    def test_stale_selection_falls_back_to_default(self):
        org_skill = AgentSkill.objects.create(
            slug="research", name="Org Research", instructions="o",
            level="org", organization=self.org,
        )
        # Selection points at a UUID that doesn't correspond to any skill.
        self._set_pref("research", "00000000-0000-0000-0000-000000000000")
        result = get_available_skills(self.user)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pk, org_skill.pk)

    def test_unrelated_prefs_preserved_when_writing_selection(self):
        skill = AgentSkill.objects.create(
            slug="research", name="Research", instructions="x", level="system",
        )
        # Pre-existing unrelated pref.
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {"theme_overrides": {"foo": "bar"}}
        us.save(update_fields=["preferences"])

        set_user_skill_selection(self.user, skill, enabled=False)

        us.refresh_from_db()
        self.assertEqual(us.preferences.get("theme_overrides"), {"foo": "bar"})
        self.assertIsNone(us.preferences["skills"]["research"]["selected_skill_id"])


class SetUserSkillSelectionTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="tog@example.com", password="pass")
        self.org = Organization.objects.create(name="Tog Org", slug="tog-org")
        Membership.objects.create(user=self.user, org=self.org)

    def test_enable_replaces_default_active(self):
        org_skill = AgentSkill.objects.create(
            slug="x", name="Org X", instructions="o",
            level="org", organization=self.org,
        )
        user_skill = AgentSkill.objects.create(
            slug="x", name="My X", instructions="u",
            level="user", created_by=self.user,
        )
        # Active by default = user_skill (shadowing). Now explicitly enable
        # org_skill — replaced should report the user version.
        result = set_user_skill_selection(self.user, org_skill, enabled=True)
        self.assertTrue(result["now_active"])
        self.assertIsNotNone(result["replaced"])
        self.assertEqual(result["replaced"]["id"], str(user_skill.id))
        self.assertEqual(result["replaced"]["level"], "user")

    def test_enable_with_no_prior_active_returns_no_replacement(self):
        skill = AgentSkill.objects.create(
            slug="solo", name="Solo", instructions="s",
            level="user", created_by=self.user,
        )
        result = set_user_skill_selection(self.user, skill, enabled=True)
        self.assertTrue(result["now_active"])
        self.assertIsNone(result["replaced"])

    def test_disable_writes_null_selection(self):
        skill = AgentSkill.objects.create(
            slug="solo", name="Solo", instructions="s",
            level="user", created_by=self.user,
        )
        set_user_skill_selection(self.user, skill, enabled=False)
        us = UserSettings.objects.get(user=self.user)
        self.assertIsNone(us.preferences["skills"]["solo"]["selected_skill_id"])
        self.assertNotIn(skill, get_available_skills(self.user))


class CreateOrgSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pass")
        self.member = User.objects.create_user(email="mem@example.com", password="pass")
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)

    def test_admin_can_create(self):
        skill = create_org_skill(self.admin, "New Skill", self.org)
        self.assertEqual(skill.level, "org")
        self.assertEqual(skill.organization, self.org)
        self.assertEqual(skill.slug, "new-skill")

    def test_member_cannot_create(self):
        with self.assertRaises(PermissionError):
            create_org_skill(self.member, "Forbidden", self.org)

    def test_dedupes_slug(self):
        create_org_skill(self.admin, "Foo Bar", self.org)
        again = create_org_skill(self.admin, "Foo Bar", self.org)
        self.assertEqual(again.slug, "foo-bar-1")


class PromoteSkillToOrgTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm2@example.com", password="pass")
        self.member = User.objects.create_user(email="mem2@example.com", password="pass")
        self.org = Organization.objects.create(name="Co2", slug="co2")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)

    def test_admin_promotes_user_skill(self):
        source = AgentSkill.objects.create(
            slug="my-skill", name="My Skill", instructions="i",
            description="d", level="user", created_by=self.admin,
            tool_names=["search"],
        )
        SkillTemplate.objects.create(skill=source, name="T", content="C")

        promoted = promote_skill_to_org(self.admin, source, self.org)
        self.assertEqual(promoted.level, "org")
        self.assertEqual(promoted.organization, self.org)
        self.assertEqual(promoted.instructions, "i")
        self.assertEqual(promoted.description, "d")
        self.assertEqual(promoted.tool_names, ["search"])
        self.assertEqual(promoted.parent, source)
        self.assertEqual(list(promoted.templates.values_list("name", flat=True)), ["T"])

    def test_member_cannot_promote(self):
        source = AgentSkill.objects.create(
            slug="x", name="X", instructions="i",
            level="user", created_by=self.member,
        )
        with self.assertRaises(PermissionError):
            promote_skill_to_org(self.member, source, self.org)

    def test_promote_org_skill_in_same_org_is_noop(self):
        existing = AgentSkill.objects.create(
            slug="existing", name="Existing", instructions="i",
            level="org", organization=self.org,
        )
        result = promote_skill_to_org(self.admin, existing, self.org)
        self.assertEqual(result.pk, existing.pk)
