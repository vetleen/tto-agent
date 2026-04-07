"""View tests for the agent_skills UI."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization, UserSettings
from agent_skills.models import AgentSkill, SkillTemplate

User = get_user_model()


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsListViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="u@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Acme", slug="acme")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        AgentSkill.objects.create(
            slug="sys-x", name="Sys X", instructions="i", level="system",
        )
        AgentSkill.objects.create(
            slug="org-x", name="Org X", instructions="i",
            level="org", organization=self.org,
        )
        AgentSkill.objects.create(
            slug="usr-x", name="My X", instructions="i",
            level="user", created_by=self.user,
        )

    def test_requires_login(self):
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 302)

    def test_lists_three_sections(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your skills")
        self.assertContains(response, "Acme")
        self.assertContains(response, "Built-in skills")
        self.assertContains(response, "Sys X")
        self.assertContains(response, "Org X")
        self.assertContains(response, "My X")

    def test_non_member_sees_no_org_section(self):
        outsider = User.objects.create_user(email="out@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self.client.get(reverse("agent_skills_list"))
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Acme")
        self.assertNotContains(response, "Org X")


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsCreateViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="c@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="Co", slug="co")

    def test_creates_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create"), {"name": "Built one"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            AgentSkill.objects.filter(
                level="user", created_by=self.user, name="Built one"
            ).exists()
        )

    def test_member_cannot_create_org_skill(self):
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.MEMBER
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create_org"), {"name": "Forbidden"}
        )
        self.assertEqual(response.status_code, 403)

    def test_admin_creates_org_skill(self):
        Membership.objects.create(
            user=self.user, org=self.org, role=Membership.Role.ADMIN
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_create_org"), {"name": "Admin made"}
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            AgentSkill.objects.filter(
                level="org", organization=self.org, name="Admin made"
            ).exists()
        )


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsDetailViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="d@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill = AgentSkill.objects.create(
            slug="my", name="My Skill", instructions="hello",
            level="user", created_by=self.user,
        )

    def test_owner_sees_editable_form(self):
        self.client.force_login(self.user)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "My Skill")
        self.assertNotContains(response, "readonly")

    def test_non_owner_redirected(self):
        outsider = User.objects.create_user(email="o@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self.client.get(
            reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(response.status_code, 302)


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsSaveViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="s@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill = AgentSkill.objects.create(
            slug="my", name="My Skill", instructions="hello",
            level="user", created_by=self.user,
        )

    def _post(self, action, **extra):
        payload = {
            "action": action,
            "name": extra.get("name", "Updated"),
            "description": extra.get("description", "desc"),
            "instructions": extra.get("instructions", "new instructions"),
            "tool_names_json": extra.get("tool_names_json", "[]"),
            "templates_json": extra.get("templates_json", "[]"),
        }
        return self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": self.skill.id}),
            payload,
        )

    def test_save_updates_in_place(self):
        self.client.force_login(self.user)
        response = self._post("save")
        self.assertEqual(response.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "Updated")
        self.assertEqual(self.skill.instructions, "new instructions")

    def test_save_as_user_creates_copy(self):
        self.client.force_login(self.user)
        before = AgentSkill.objects.filter(level="user").count()
        response = self._post("save_as_user", name="Forked")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(AgentSkill.objects.filter(level="user").count(), before + 1)
        # Original is unchanged.
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "My Skill")

    def test_save_creates_new_template(self):
        templates_json = json.dumps([{"id": None, "name": "T1", "content": "Hello"}])
        self.client.force_login(self.user)
        self._post("save", templates_json=templates_json)
        self.assertEqual(self.skill.templates.count(), 1)
        self.assertEqual(self.skill.templates.first().name, "T1")

    def test_save_deletes_missing_template(self):
        SkillTemplate.objects.create(skill=self.skill, name="Old", content="x")
        self.client.force_login(self.user)
        self._post("save", templates_json="[]")
        self.assertEqual(self.skill.templates.count(), 0)

    def test_non_owner_save_forbidden(self):
        outsider = User.objects.create_user(email="o2@example.com", password="pw")
        outsider.email_verified = True
        outsider.save(update_fields=["email_verified"])
        self.client.force_login(outsider)
        response = self._post("save")
        # get_skill_for_user returns None → redirect to list, not 403.
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], reverse("agent_skills_list"))


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsCopyDeleteToggleTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="cdt@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="O", slug="o")
        Membership.objects.create(user=self.user, org=self.org, role=Membership.Role.MEMBER)
        self.sys_skill = AgentSkill.objects.create(
            slug="sys-only", name="Sys Only", instructions="i", level="system",
        )
        self.user_skill = AgentSkill.objects.create(
            slug="my-only", name="Mine", instructions="i",
            level="user", created_by=self.user,
        )

    def test_copy_creates_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_copy", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        copies = AgentSkill.objects.filter(
            level="user", created_by=self.user, name="Sys Only"
        )
        self.assertEqual(copies.count(), 1)
        self.assertEqual(copies.first().parent, self.sys_skill)

    def test_delete_user_skill(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_delete", kwargs={"skill_id": self.user_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertFalse(AgentSkill.objects.filter(pk=self.user_skill.pk).exists())

    def test_delete_system_skill_forbidden(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_delete", kwargs={"skill_id": self.sys_skill.id})
        )
        self.assertEqual(response.status_code, 403)
        self.assertTrue(AgentSkill.objects.filter(pk=self.sys_skill.pk).exists())

    def test_toggle_disable(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_toggle", kwargs={"skill_id": self.user_skill.id}),
            {"enabled": "0"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["ok"])
        self.assertFalse(data["now_active"])
        self.assertIsNone(data["replaced"])

        us = UserSettings.objects.get(user=self.user)
        self.assertIsNone(us.preferences["skills"]["my-only"]["selected_skill_id"])

    def test_toggle_enable_replaces_default(self):
        # Create org skill with the same slug as user_skill so user_skill
        # currently shadows it. Enable the org version explicitly.
        org_skill = AgentSkill.objects.create(
            slug="my-only", name="Org Mine", instructions="i",
            level="org", organization=self.org,
        )
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_toggle", kwargs={"skill_id": org_skill.id}),
            {"enabled": "1"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["now_active"])
        self.assertIsNotNone(data["replaced"])
        self.assertEqual(data["replaced"]["id"], str(self.user_skill.id))


@override_settings(ALLOWED_HOSTS=["testserver"])
class SkillsPromoteViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.admin = User.objects.create_user(email="adm@example.com", password="pw")
        self.admin.email_verified = True
        self.admin.save(update_fields=["email_verified"])
        self.org = Organization.objects.create(name="P", slug="p")
        Membership.objects.create(user=self.admin, org=self.org, role=Membership.Role.ADMIN)
        self.user_skill = AgentSkill.objects.create(
            slug="great", name="Great", instructions="i",
            level="user", created_by=self.admin,
        )

    def test_admin_promote(self):
        self.client.force_login(self.admin)
        response = self.client.post(
            reverse("agent_skills_promote", kwargs={"skill_id": self.user_skill.id})
        )
        self.assertEqual(response.status_code, 302)
        self.assertTrue(
            AgentSkill.objects.filter(
                level="org", organization=self.org, name="Great"
            ).exists()
        )

    def test_member_cannot_promote(self):
        member = User.objects.create_user(email="mem@example.com", password="pw")
        member.email_verified = True
        member.save(update_fields=["email_verified"])
        Membership.objects.create(user=member, org=self.org, role=Membership.Role.MEMBER)
        own_skill = AgentSkill.objects.create(
            slug="x", name="X", instructions="i",
            level="user", created_by=member,
        )
        self.client.force_login(member)
        response = self.client.post(
            reverse("agent_skills_promote", kwargs={"skill_id": own_skill.id})
        )
        self.assertEqual(response.status_code, 403)
