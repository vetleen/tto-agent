"""Tests for promote/demote as in-place level moves (type changes)."""

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    get_accessible_skills,
    move_skill_to_org,
    move_skill_to_personal,
)

User = get_user_model()


class MoveSkillToOrgTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.member = User.objects.create_user(email="mem@example.com", password="pw")
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)

    def _personal(self, owner, slug="mine", name="Mine"):
        skill = AgentSkill.objects.create(
            slug=slug, name=name, emoji="🧪", instructions="i", description="d",
            level="user", created_by=owner, tool_names=["search"],
        )
        SkillTemplate.objects.create(skill=skill, name="T", content="C")
        return skill

    def test_promote_moves_in_place(self):
        skill = self._personal(self.admin)
        original_id = skill.id

        result = move_skill_to_org(self.admin, skill, self.org)

        self.assertEqual(result.id, original_id)  # same row, not a copy
        self.assertEqual(result.level, "org")
        self.assertEqual(result.organization, self.org)
        self.assertIsNone(result.created_by)
        # Templates preserved on the same row.
        self.assertEqual(list(result.templates.values_list("name", flat=True)), ["T"])
        # Nothing left behind at the personal tier.
        self.assertFalse(AgentSkill.objects.filter(level="user", name="Mine").exists())
        self.assertEqual(AgentSkill.objects.count(), 1)

    def test_promote_requires_admin(self):
        skill = self._personal(self.member)
        with self.assertRaises(PermissionError):
            move_skill_to_org(self.member, skill, self.org)

    def test_promote_rejects_non_user_skill(self):
        sys_skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        with self.assertRaises(ValueError):
            move_skill_to_org(self.admin, sys_skill, self.org)

    def test_promote_dedupes_slug(self):
        AgentSkill.objects.create(
            slug="mine", name="Existing Org", instructions="i",
            level="org", organization=self.org,
        )
        skill = self._personal(self.admin, slug="mine")
        result = move_skill_to_org(self.admin, skill, self.org)
        self.assertEqual(result.slug, "mine-1")


class MoveSkillToPersonalTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.member = User.objects.create_user(email="mem@example.com", password="pw")
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)

    def _org_skill(self, slug="shared", name="Shared"):
        skill = AgentSkill.objects.create(
            slug=slug, name=name, emoji="🧪", instructions="i", description="d",
            level="org", organization=self.org, tool_names=["search"],
        )
        SkillTemplate.objects.create(skill=skill, name="T", content="C")
        return skill

    def test_demote_moves_in_place(self):
        skill = self._org_skill()
        original_id = skill.id

        result = move_skill_to_personal(self.admin, skill)

        self.assertEqual(result.id, original_id)  # same row
        self.assertEqual(result.level, "user")
        self.assertEqual(result.created_by, self.admin)
        self.assertIsNone(result.organization)
        self.assertEqual(list(result.templates.values_list("name", flat=True)), ["T"])
        self.assertEqual(AgentSkill.objects.count(), 1)

    def test_demote_removes_from_other_members(self):
        skill = self._org_skill()
        # Member can see it as an org skill beforehand.
        self.assertIn(skill.id, [s.id for s in get_accessible_skills(self.member)])
        move_skill_to_personal(self.admin, skill)
        # After demote it is the admin's personal skill — member loses access.
        self.assertNotIn(skill.id, [s.id for s in get_accessible_skills(self.member)])

    def test_demote_requires_admin(self):
        skill = self._org_skill()
        with self.assertRaises(PermissionError):
            move_skill_to_personal(self.member, skill)

    def test_demote_rejects_non_org_skill(self):
        personal = AgentSkill.objects.create(
            slug="p", name="P", instructions="i", level="user", created_by=self.admin,
        )
        with self.assertRaises(ValueError):
            move_skill_to_personal(self.admin, personal)

    def test_demote_dedupes_slug(self):
        AgentSkill.objects.create(
            slug="shared", name="Admin Personal", instructions="i",
            level="user", created_by=self.admin,
        )
        skill = self._org_skill(slug="shared")
        result = move_skill_to_personal(self.admin, skill)
        self.assertEqual(result.slug, "shared-1")


@override_settings(ALLOWED_HOSTS=["testserver"])
class DemoteViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.admin.email_verified = True
        self.admin.save(update_fields=["email_verified"])
        self.member = User.objects.create_user(email="mem@example.com", password="pw")
        self.member.email_verified = True
        self.member.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        Membership.objects.create(user=self.member, org=self.org, role=Membership.Role.MEMBER)
        self.org_skill = AgentSkill.objects.create(
            slug="shared", name="Shared", instructions="i",
            level="org", organization=self.org,
        )

    def test_admin_demote(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("agent_skills_demote", kwargs={"skill_id": self.org_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        self.org_skill.refresh_from_db()
        self.assertEqual(self.org_skill.level, "user")
        self.assertEqual(self.org_skill.created_by, self.admin)
        self.assertIsNone(self.org_skill.organization)

    def test_member_cannot_demote(self):
        self.client.force_login(self.member)
        response = self.client.post(
            reverse("agent_skills_demote", kwargs={"skill_id": self.org_skill.id})
        )
        self.assertEqual(response.status_code, 403)
        self.org_skill.refresh_from_db()
        self.assertEqual(self.org_skill.level, "org")


@override_settings(ALLOWED_HOSTS=["testserver"])
class CopyToOrgViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.admin.email_verified = True
        self.admin.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        self.sys_skill = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", level="system",
        )
        self.org.preferences = {"skills": {"sys": {"enabled": True}}}
        self.org.save(update_fields=["preferences"])

    def test_copy_system_skill_to_org(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("agent_skills_copy_to_org", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        # A new org copy exists, and the system skill is untouched.
        self.assertTrue(
            AgentSkill.objects.filter(level="org", organization=self.org, name="Sys").exists()
        )
        self.assertTrue(AgentSkill.objects.filter(level="system", slug="sys").exists())

    def test_member_cannot_copy_to_org(self):
        member = User.objects.create_user(email="m2@example.com", password="pw")
        member.email_verified = True
        member.save(update_fields=["email_verified"])
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        self.client.force_login(member)
        response = self.client.post(
            reverse("agent_skills_copy_to_org", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 403)


@override_settings(ALLOWED_HOSTS=["testserver"])
class DetailActionMoveTests(TestCase):
    """The detail-page Promote/Demote buttons save form edits then change type."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.admin.email_verified = True
        self.admin.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Co", slug="co")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)

    def _post(self, skill, action, **extra):
        data = {
            "action": action,
            "name": extra.get("name", skill.name),
            "description": extra.get("description", ""),
            "instructions": extra.get("instructions", "edited body"),
            "tool_names_json": "[]",
            "templates_json": "[]",
        }
        return self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": skill.id}), data
        )

    def test_detail_promote_applies_edits_then_moves(self):
        skill = AgentSkill.objects.create(
            slug="mine", name="Mine", instructions="old", level="user", created_by=self.admin,
        )
        self.client.force_login(self.admin)
        response = self._post(skill, "promote", instructions="new body")
        self.assertEqual(response.status_code, 302)
        skill.refresh_from_db()
        self.assertEqual(skill.level, "org")
        self.assertEqual(skill.organization, self.org)
        self.assertEqual(skill.instructions, "new body")

    def test_detail_demote_applies_edits_then_moves(self):
        skill = AgentSkill.objects.create(
            slug="shared", name="Shared", instructions="old",
            level="org", organization=self.org,
        )
        self.client.force_login(self.admin)
        response = self._post(skill, "demote", instructions="new body")
        self.assertEqual(response.status_code, 302)
        skill.refresh_from_db()
        self.assertEqual(skill.level, "user")
        self.assertEqual(skill.created_by, self.admin)
        self.assertEqual(skill.instructions, "new body")
