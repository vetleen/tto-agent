"""Tests for agent_skills.services."""

from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.test import TestCase

from accounts.models import Membership, Organization, UserSettings
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    _next_free_slug,
    _save_with_free_slug,
    can_edit_skill,
    create_org_skill,
    create_user_skill,
    filter_to_skill_tools,
    fork_skill,
    get_accessible_skills,
    get_available_skills,
    get_editable_skill_for_user,
    get_skill_for_user,
    migrate_skill_slug_prefs,
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

    def _enable(self, *slugs):
        prefs = self.org.preferences or {}
        skills = prefs.get("skills", {})
        for slug in slugs:
            skills.setdefault(slug, {})["enabled"] = True
        prefs["skills"] = skills
        self.org.preferences = prefs
        self.org.save(update_fields=["preferences"])

    def test_system_skills_visible_when_enabled(self):
        AgentSkill.objects.create(
            slug="sys-skill", name="System Skill",
            instructions="Inst.", level="system",
        )
        self._enable("sys-skill")
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0].slug, "sys-skill")

    def test_system_skills_hidden_by_default(self):
        AgentSkill.objects.create(
            slug="sys-skill", name="System Skill",
            instructions="Inst.", level="system",
        )
        skills = get_available_skills(self.user)
        self.assertEqual(len(skills), 0)

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
        self._enable("a-skill", "b-skill")
        skills = get_available_skills(self.user)
        self.assertEqual([s.name for s in skills], ["Alpha", "Bravo"])


class GetSkillForUserTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="user@example.com", password="pass")
        self.other_user = User.objects.create_user(email="other@example.com", password="pass")
        self.org = Organization.objects.create(name="My Org", slug="my-org")
        self.membership = Membership.objects.create(user=self.user, org=self.org)

    def _enable(self, *slugs):
        prefs = self.org.preferences or {}
        skills = prefs.get("skills", {})
        for slug in slugs:
            skills.setdefault(slug, {})["enabled"] = True
        prefs["skills"] = skills
        self.org.preferences = prefs
        self.org.save(update_fields=["preferences"])

    def test_system_skill_accessible_when_org_enabled(self):
        skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="Inst.", level="system",
        )
        self._enable("sys")
        self.assertIsNotNone(get_skill_for_user(self.user, str(skill.pk)))

    def test_system_skill_accessible_without_membership(self):
        skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="Inst.", level="system",
        )
        self.assertIsNotNone(get_skill_for_user(self.other_user, str(skill.pk)))

    def test_system_skill_hidden_by_default_for_org_member(self):
        skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="Inst.", level="system",
        )
        self.assertIsNone(get_skill_for_user(self.user, str(skill.pk)))

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

    def test_falls_back_to_owned_when_slug_disabled(self):
        """'Edit my skill X' must work even when the user toggled X off —
        a disabled slug vanishes from get_available_skills entirely."""
        mine = AgentSkill.objects.create(
            slug="toggled", name="Toggled", instructions="Inst.",
            level="user", created_by=self.user,
        )
        set_user_skill_selection(self.user, mine, enabled=False)
        result = get_editable_skill_for_user(self.user, "toggled")
        self.assertIsNotNone(result)
        self.assertEqual(result.pk, mine.pk)

    def test_falls_back_to_owned_when_org_tier_selected(self):
        """A non-admin member selected the org version (not editable by
        them); the fallback must surface the member's own version."""
        member = User.objects.create_user(email="ed-mem@example.com", password="pass")
        Membership.objects.create(
            user=member, org=self.org, role=Membership.Role.MEMBER
        )
        org_skill = AgentSkill.objects.create(
            slug="dual", name="Org Dual", instructions="Inst.",
            level="org", organization=self.org,
        )
        mine = AgentSkill.objects.create(
            slug="dual", name="My Dual", instructions="Inst.",
            level="user", created_by=member,
        )
        set_user_skill_selection(member, org_skill, enabled=True)
        result = get_editable_skill_for_user(member, "dual")
        self.assertIsNotNone(result)
        self.assertEqual(result.pk, mine.pk)

    def test_fallback_returns_none_when_user_owns_nothing(self):
        """Visible-but-not-editable with no owned skill behind it → None."""
        member = User.objects.create_user(email="ed-mem2@example.com", password="pass")
        Membership.objects.create(
            user=member, org=self.org, role=Membership.Role.MEMBER
        )
        AgentSkill.objects.create(
            slug="org-only", name="Org Only", instructions="Inst.",
            level="org", organization=self.org,
        )
        self.assertIsNone(get_editable_skill_for_user(member, "org-only"))


class ForkSkillTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="fork@example.com", password="pass")

    def test_forks_skill_with_templates(self):
        source = AgentSkill.objects.create(
            slug="source", name="Source", emoji="🧪", instructions="Do stuff.",
            description="A source skill.", level="system",
            tool_names=["search_documents"],
        )
        SkillTemplate.objects.create(skill=source, name="Template A", content="Content A")
        SkillTemplate.objects.create(skill=source, name="Template B", content="Content B")

        forked = fork_skill(self.user, source)

        self.assertEqual(forked.slug, "source")
        self.assertEqual(forked.name, "Source")
        self.assertEqual(forked.emoji, "🧪")
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
            slug="my-skill", name="My Skill", emoji="🧪", instructions="i",
            description="d", level="user", created_by=self.admin,
            tool_names=["search"],
        )
        SkillTemplate.objects.create(skill=source, name="T", content="C")

        promoted = promote_skill_to_org(self.admin, source, self.org)
        self.assertEqual(promoted.level, "org")
        self.assertEqual(promoted.organization, self.org)
        self.assertEqual(promoted.emoji, "🧪")
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


class FilterToSkillToolsTests(TestCase):
    """The allow-list keeps only registered skills-section tools.

    Relies on the real tool registry, populated at startup: agent_skills.tools
    registers the skills-section tools (view_template, ...) and chat.tools
    registers the chat-section doc tools (search_documents, read_document).
    """

    def test_keeps_skills_section_tools(self):
        self.assertEqual(
            filter_to_skill_tools(["view_template", "load_template_to_canvas"]),
            ["view_template", "load_template_to_canvas"],
        )

    def test_drops_chat_section_and_unknown_tools(self):
        result = filter_to_skill_tools(
            ["view_template", "search_documents", "read_document", "totally_made_up"]
        )
        self.assertEqual(result, ["view_template"])

    def test_preserves_order(self):
        self.assertEqual(
            filter_to_skill_tools(["load_template_to_canvas", "view_template"]),
            ["load_template_to_canvas", "view_template"],
        )

    def test_handles_none_and_non_strings(self):
        self.assertEqual(filter_to_skill_tools(None), [])
        self.assertEqual(filter_to_skill_tools([]), [])
        # Non-str entries are coerced to str, then dropped as unknown.
        self.assertEqual(filter_to_skill_tools([123, {"x": 1}]), [])


class OrgDisabledSkillVisibilityTests(TestCase):
    """An org admin's disable flag hides the skill from every access helper.

    The org settings page queries AgentSkill directly and is covered by a
    separate test in accounts/tests/test_settings_views.py.
    """

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.org = Organization.objects.create(name="Co", slug="co")
        self.member = User.objects.create_user(email="m@example.com", password="pass")
        self.admin = User.objects.create_user(email="a@example.com", password="pass", is_superuser=True)
        self.outsider = User.objects.create_user(email="o@example.com", password="pass")
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)

    def _disable(self, slug):
        self.org.preferences = {"skills": {slug: {"enabled": False}}}
        self.org.save(update_fields=["preferences"])

    def test_accessible_skills_hides_disabled_system_skill(self):
        AgentSkill.objects.create(
            slug="research", name="Research", instructions="i", level="system",
        )
        self._disable("research")
        self.assertEqual(get_accessible_skills(self.member), [])

    def test_accessible_skills_hides_for_superuser_member(self):
        AgentSkill.objects.create(
            slug="research", name="Research", instructions="i", level="system",
        )
        self._disable("research")
        # is_superuser gets no special treatment — still a member of the org.
        self.assertEqual(get_accessible_skills(self.admin), [])

    def test_accessible_skills_unaffected_for_user_without_membership(self):
        AgentSkill.objects.create(
            slug="research", name="Research", instructions="i", level="system",
        )
        self._disable("research")
        skills = get_accessible_skills(self.outsider)
        self.assertEqual([s.slug for s in skills], ["research"])

    def test_accessible_skills_hides_user_fork_with_disabled_slug(self):
        # Preserves existing preferences.py semantics: disable is by slug.
        AgentSkill.objects.create(
            slug="research", name="Sys Research", instructions="i", level="system",
        )
        AgentSkill.objects.create(
            slug="research", name="My Research", instructions="i",
            level="user", created_by=self.member,
        )
        self._disable("research")
        self.assertEqual(get_accessible_skills(self.member), [])

    def test_skill_for_user_returns_none_when_org_disabled(self):
        skill = AgentSkill.objects.create(
            slug="research", name="Research", instructions="i", level="system",
        )
        self._disable("research")
        self.assertIsNone(get_skill_for_user(self.member, str(skill.pk)))
        self.assertIsNone(get_skill_for_user(self.admin, str(skill.pk)))
        # Non-member unaffected.
        self.assertIsNotNone(get_skill_for_user(self.outsider, str(skill.pk)))

    def test_re_enabling_restores_visibility(self):
        skill = AgentSkill.objects.create(
            slug="research", name="Research", instructions="i", level="system",
        )
        self._disable("research")
        self.assertEqual(get_accessible_skills(self.member), [])
        self.org.preferences = {"skills": {"research": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])
        self.assertEqual([s.pk for s in get_accessible_skills(self.member)], [skill.pk])


class SlugDedupHelperTests(TestCase):
    """_next_free_slug / _save_with_free_slug — the shared dedup + race retry."""

    def test_next_free_slug_returns_base_when_free(self):
        self.assertEqual(_next_free_slug("base", lambda s: False), "base")

    def test_next_free_slug_increments_past_taken(self):
        taken = {"base", "base-1"}
        self.assertEqual(_next_free_slug("base", lambda s: s in taken), "base-2")

    def test_next_free_slug_suffix_fits_64_chars(self):
        base = "a" * 64
        result = _next_free_slug(base, lambda s: s == base)
        self.assertEqual(result, "a" * 62 + "-1")
        self.assertLessEqual(len(result), 64)

    def test_retries_with_recomputed_slug_on_integrity_error(self):
        """Simulated race: a concurrent request claims the slug between the
        existence check and the INSERT. The retry recomputes and succeeds."""
        now_taken = set()
        calls = []

        def persist(slug):
            calls.append(slug)
            if len(calls) == 1:
                now_taken.add(slug)  # the concurrent winner owns it now
                raise IntegrityError("simulated concurrent claim")
            return slug

        result = _save_with_free_slug("race", lambda s: s in now_taken, persist)
        self.assertEqual(calls, ["race", "race-1"])
        self.assertEqual(result, "race-1")

    def test_reraises_after_max_attempts(self):
        def persist(slug):
            raise IntegrityError("always loses the race")

        with self.assertRaises(IntegrityError):
            _save_with_free_slug("doomed", lambda s: False, persist)


class MigrateSkillSlugPrefsTests(TestCase):
    """Slug-keyed user prefs must follow a skill across a slug rename."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="mig@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="old-slug", name="Mig", instructions="i",
            level="user", created_by=self.user,
        )

    def _set_prefs(self, user, skills_dict):
        us, _ = UserSettings.objects.get_or_create(user=user)
        us.preferences = {"skills": skills_dict}
        us.save(update_fields=["preferences"])
        return us

    def test_moves_matching_entry(self):
        us = self._set_prefs(
            self.user, {"old-slug": {"selected_skill_id": str(self.skill.id)}}
        )
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        us.refresh_from_db()
        skills = us.preferences["skills"]
        self.assertNotIn("old-slug", skills)
        self.assertEqual(skills["new-slug"]["selected_skill_id"], str(self.skill.id))

    def test_leaves_disable_entries_alone(self):
        """A slug-wide disable (selected_skill_id: None) isn't a selection of
        this skill — it stays keyed under the old slug."""
        us = self._set_prefs(self.user, {"old-slug": {"selected_skill_id": None}})
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        us.refresh_from_db()
        skills = us.preferences["skills"]
        self.assertIn("old-slug", skills)
        self.assertNotIn("new-slug", skills)

    def test_leaves_other_skill_entries_alone(self):
        other_owner = User.objects.create_user(email="oth@example.com", password="pass")
        other = AgentSkill.objects.create(
            slug="old-slug", name="Other's", instructions="i",
            level="user", created_by=other_owner,
        )
        us = self._set_prefs(
            self.user, {"old-slug": {"selected_skill_id": str(other.id)}}
        )
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        us.refresh_from_db()
        skills = us.preferences["skills"]
        self.assertEqual(skills["old-slug"]["selected_skill_id"], str(other.id))
        self.assertNotIn("new-slug", skills)

    def test_does_not_clobber_existing_target_entry(self):
        target = AgentSkill.objects.create(
            slug="new-slug", name="Target", instructions="i",
            level="user", created_by=self.user,
        )
        us = self._set_prefs(self.user, {
            "old-slug": {"selected_skill_id": str(self.skill.id)},
            "new-slug": {"selected_skill_id": str(target.id)},
        })
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        us.refresh_from_db()
        skills = us.preferences["skills"]
        # Old key is removed, but the user's explicit choice under the new
        # slug wins — never clobbered.
        self.assertNotIn("old-slug", skills)
        self.assertEqual(skills["new-slug"]["selected_skill_id"], str(target.id))

    def test_noop_when_slug_unchanged(self):
        us = self._set_prefs(
            self.user, {"old-slug": {"selected_skill_id": str(self.skill.id)}}
        )
        migrate_skill_slug_prefs(self.skill, "old-slug", "old-slug")
        us.refresh_from_db()
        self.assertIn("old-slug", us.preferences["skills"])

    def test_migrates_every_user_with_matching_entry(self):
        """An org skill rename must follow every member's selection."""
        second = User.objects.create_user(email="mig2@example.com", password="pass")
        us1 = self._set_prefs(
            self.user, {"old-slug": {"selected_skill_id": str(self.skill.id)}}
        )
        us2 = self._set_prefs(
            second, {"old-slug": {"selected_skill_id": str(self.skill.id)}}
        )
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        for us in (us1, us2):
            us.refresh_from_db()
            self.assertEqual(
                us.preferences["skills"]["new-slug"]["selected_skill_id"],
                str(self.skill.id),
            )

    def test_unrelated_prefs_untouched(self):
        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {
            "theme_overrides": {"foo": "bar"},
            "skills": {"old-slug": {"selected_skill_id": str(self.skill.id)}},
        }
        us.save(update_fields=["preferences"])
        migrate_skill_slug_prefs(self.skill, "old-slug", "new-slug")
        us.refresh_from_db()
        self.assertEqual(us.preferences.get("theme_overrides"), {"foo": "bar"})
