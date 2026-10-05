"""Tests for "Share with organization": a member shares a personal skill, an
admin approves it in place, and the member keeps editing it as maintainer."""

import json
import uuid
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.test import RequestFactory, TestCase, override_settings
from django.urls import reverse

from accounts.context_processors import nav_context
from accounts.models import Membership, Organization
from agent_skills import resources as res
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.services import (
    approve_skill_share,
    can_delete_skill,
    can_edit_skill,
    decline_skill_share,
    get_editable_skill_for_user,
    get_pending_share_for_admin,
    get_skill_for_user,
    move_skill_to_personal,
    pending_skill_shares,
    request_skill_share,
    revoke_skill_maintainer,
    withdraw_skill_share,
)
from agent_skills.tools import DeleteSkillTool, EditSkillTool
from llm.types import RunContext

User = get_user_model()


class _ShareBase(TestCase):
    def setUp(self):
        # No scanning configured: skills are usable unscanned (tests that need
        # the gate patch ``_scanning_configured`` back on).
        patcher = patch.object(res, "_scanning_configured_for_org", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        AgentSkill.objects.all().delete()
        self.org = Organization.objects.create(name="Co", slug="co")
        self.admin = self._user("adm@example.com", Membership.Role.ADMIN)
        self.member = self._user("mem@example.com", Membership.Role.MEMBER)
        self.other = self._user("oth@example.com", Membership.Role.MEMBER)
        self.skill = AgentSkill.objects.create(
            slug="mine", name="Mine", instructions="i", description="d",
            level="user", created_by=self.member,
        )
        SkillTemplate.objects.create(skill=self.skill, name="T", content="C")

    def _user(self, email, role, org=None):
        user = User.objects.create_user(email=email, password="pw")
        user.email_verified = True
        user.save(update_fields=["email_verified"])
        Membership.objects.create(user=user, org=org or self.org, role=role)
        return user

    def _fresh(self, user):
        # Drop per-instance memos (membership, prefs) between steps.
        return User.objects.get(pk=user.pk)

    def _approved(self):
        request_skill_share(self.member, self.skill)
        approve_skill_share(self.admin, self.skill)
        self.skill.refresh_from_db()
        return self.skill


class RequestShareTests(_ShareBase):
    def test_request_marks_pending_for_org(self):
        request_skill_share(self.member, self.skill)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.share_requested_org, self.org)
        self.assertIsNotNone(self.skill.share_requested_at)
        self.assertEqual(self.skill.level, "user")
        self.assertEqual(list(pending_skill_shares(self.org)), [self.skill])

    def test_request_is_idempotent(self):
        request_skill_share(self.member, self.skill)
        first = AgentSkill.objects.get(pk=self.skill.pk).share_requested_at
        request_skill_share(self.member, self.skill)
        self.assertEqual(AgentSkill.objects.get(pk=self.skill.pk).share_requested_at, first)

    def test_only_owner_can_share(self):
        with self.assertRaises(PermissionError):
            request_skill_share(self.other, self.skill)

    def test_admin_cannot_share(self):
        own = AgentSkill.objects.create(
            slug="a", name="A", instructions="i", level="user", created_by=self.admin,
        )
        with self.assertRaises(PermissionError):
            request_skill_share(self.admin, own)

    def test_user_without_org_cannot_share(self):
        loner = User.objects.create_user(email="solo@example.com", password="pw")
        skill = AgentSkill.objects.create(
            slug="s", name="S", instructions="i", level="user", created_by=loner,
        )
        with self.assertRaises(PermissionError):
            request_skill_share(loner, skill)

    def test_withdraw_clears_request(self):
        request_skill_share(self.member, self.skill)
        withdraw_skill_share(self.member, self.skill)
        self.skill.refresh_from_db()
        self.assertIsNone(self.skill.share_requested_org)
        self.assertIsNone(self.skill.share_requested_at)

    def test_withdraw_requires_owner(self):
        request_skill_share(self.member, self.skill)
        with self.assertRaises(PermissionError):
            withdraw_skill_share(self.other, self.skill)

    def test_pending_excludes_submitter_who_left(self):
        request_skill_share(self.member, self.skill)
        Membership.objects.filter(user=self.member).delete()
        self.assertFalse(pending_skill_shares(self.org).exists())

    def test_pending_excludes_soft_deleted(self):
        from agent_skills.services import soft_delete_skill

        request_skill_share(self.member, self.skill)
        soft_delete_skill(self.skill)
        self.assertFalse(pending_skill_shares(self.org).exists())

    def test_check_constraint_rejects_request_on_org_skill(self):
        with self.assertRaises(IntegrityError), transaction.atomic():
            AgentSkill.objects.create(
                slug="o", name="O", instructions="i", level="org",
                organization=self.org, share_requested_org=self.org,
            )


class ReviewAccessTests(_ShareBase):
    def test_admin_reads_pending_share_but_cannot_attach_it(self):
        request_skill_share(self.member, self.skill)
        admin = self._fresh(self.admin)
        self.assertEqual(get_pending_share_for_admin(admin, str(self.skill.id)), self.skill)
        # Not reachable through the chat/attach read gate.
        self.assertIsNone(get_skill_for_user(admin, str(self.skill.id)))

    def test_member_cannot_review(self):
        request_skill_share(self.member, self.skill)
        self.assertIsNone(
            get_pending_share_for_admin(self._fresh(self.other), str(self.skill.id))
        )

    def test_other_org_admin_cannot_review(self):
        org2 = Organization.objects.create(name="Other", slug="other")
        admin2 = self._user("adm2@example.com", Membership.Role.ADMIN, org=org2)
        request_skill_share(self.member, self.skill)
        self.assertIsNone(get_pending_share_for_admin(admin2, str(self.skill.id)))
        with self.assertRaises(PermissionError):
            approve_skill_share(admin2, self.skill)

    def test_not_pending_is_not_reviewable(self):
        self.assertIsNone(
            get_pending_share_for_admin(self._fresh(self.admin), str(self.skill.id))
        )


class ApproveShareTests(_ShareBase):
    def test_approve_moves_in_place_with_maintainer(self):
        original_id = self.skill.id
        skill = self._approved()
        self.assertEqual(skill.id, original_id)
        self.assertEqual(skill.level, "org")
        self.assertEqual(skill.organization, self.org)
        self.assertIsNone(skill.created_by)
        self.assertEqual(skill.maintainer, self.member)
        self.assertIsNone(skill.share_requested_org)
        self.assertIsNone(skill.share_requested_at)
        self.assertEqual(list(skill.templates.values_list("name", flat=True)), ["T"])

    def test_approved_skill_visible_to_other_members(self):
        self._approved()
        self.assertIsNotNone(get_skill_for_user(self._fresh(self.other), str(self.skill.id)))

    def test_thread_attachment_survives(self):
        from chat.models import ChatThread, ChatThreadSkill

        thread = ChatThread.objects.create(created_by=self.member)
        ChatThreadSkill.objects.create(thread=thread, skill=self.skill)
        self._approved()
        self.assertTrue(ChatThreadSkill.objects.filter(thread=thread, skill=self.skill).exists())

    def test_slug_dedupes_against_existing_org_skill(self):
        AgentSkill.objects.create(
            slug="mine", name="Org mine", instructions="i", level="org",
            organization=self.org,
        )
        skill = self._approved()
        self.assertEqual(skill.slug, "mine-1")

    def test_slug_avoids_org_disabled_slug(self):
        self.org.preferences = {"skills": {"mine": {"enabled": False}}}
        self.org.save(update_fields=["preferences"])
        skill = self._approved()
        self.assertEqual(skill.slug, "mine-1")
        # On by default: visible to members.
        self.assertIsNotNone(get_skill_for_user(self._fresh(self.other), str(skill.id)))

    def test_approve_requires_admin(self):
        request_skill_share(self.member, self.skill)
        with self.assertRaises(PermissionError):
            approve_skill_share(self.other, self.skill)

    def test_approve_requires_pending_request(self):
        with self.assertRaises(ValueError):
            approve_skill_share(self.admin, self.skill)

    def test_approve_refused_when_submitter_left(self):
        request_skill_share(self.member, self.skill)
        Membership.objects.filter(user=self.member).delete()
        with self.assertRaises(ValueError):
            approve_skill_share(self.admin, self.skill)

    def test_approve_refused_until_scan_passes(self):
        request_skill_share(self.member, self.skill)
        with patch.object(res, "_scanning_configured", return_value=True):
            with self.assertRaises(ValueError):
                approve_skill_share(self.admin, self.skill)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.level, "user")

    def test_approve_refused_when_blocked(self):
        request_skill_share(self.member, self.skill)
        AgentSkill.objects.filter(pk=self.skill.pk).update(
            scan_state=AgentSkill.ScanState.BLOCKED
        )
        self.skill.refresh_from_db()
        with self.assertRaises(ValueError):
            approve_skill_share(self.admin, self.skill)

    def test_decline_keeps_personal_skill(self):
        request_skill_share(self.member, self.skill)
        decline_skill_share(self.admin, self.skill)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.level, "user")
        self.assertEqual(self.skill.created_by, self.member)
        self.assertIsNone(self.skill.share_requested_org)

    def test_decline_requires_admin(self):
        request_skill_share(self.member, self.skill)
        with self.assertRaises(PermissionError):
            decline_skill_share(self.other, self.skill)


class MaintainerTests(_ShareBase):
    def test_maintainer_can_edit_but_not_delete(self):
        skill = self._approved()
        self.assertTrue(can_edit_skill(self.member, skill))
        self.assertFalse(can_delete_skill(self.member, skill))
        self.assertTrue(can_delete_skill(self.admin, skill))
        self.assertFalse(can_edit_skill(self.other, skill))

    def test_maintainer_loses_rights_after_leaving_org(self):
        skill = self._approved()
        Membership.objects.filter(user=self.member).delete()
        self.assertFalse(can_edit_skill(self.member, skill))
        self.assertTrue(can_edit_skill(self.admin, skill))

    def test_deleting_maintainer_keeps_org_skill(self):
        skill = self._approved()
        self.member.delete()
        skill.refresh_from_db()
        self.assertEqual(skill.level, "org")
        self.assertIsNone(skill.maintainer)

    def test_revoke_removes_editing_rights(self):
        skill = self._approved()
        revoke_skill_maintainer(self.admin, skill)
        skill.refresh_from_db()
        self.assertIsNone(skill.maintainer)
        self.assertFalse(can_edit_skill(self.member, skill))

    def test_revoke_requires_admin(self):
        skill = self._approved()
        with self.assertRaises(PermissionError):
            revoke_skill_maintainer(self.member, skill)

    def test_demote_clears_maintainer(self):
        skill = self._approved()
        move_skill_to_personal(self.admin, skill)
        skill.refresh_from_db()
        self.assertIsNone(skill.maintainer)

    def test_editable_lookup_falls_back_to_maintained_skill(self):
        skill = self._approved()
        member = self._fresh(self.member)
        # The member explicitly disabled the slug, so it isn't in their list.
        from accounts.models import UserSettings

        us, _ = UserSettings.objects.get_or_create(user=member)
        us.preferences = {"skills": {skill.slug: {"selected_skill_id": None}}}
        us.save(update_fields=["preferences"])
        self.assertEqual(get_editable_skill_for_user(self._fresh(member), skill.slug), skill)

    def test_scan_runs_as_maintainer(self):
        skill = self._approved()
        self.assertEqual(res._scan_user_for_skill(skill, None), self.member)


def _ctx(user):
    return RunContext.create(user_id=user.pk, conversation_id=str(uuid.uuid4()))


class MaintainerToolTests(_ShareBase):
    def test_maintainer_edits_via_chat_tool_in_place(self):
        skill = self._approved()
        tool = EditSkillTool()
        tool.context = _ctx(self.member)
        out = json.loads(tool._run(
            skill_slug=skill.slug, updates={"name": "Renamed"}, text_edits=[],
        ))
        self.assertEqual(out["status"], "ok")
        skill.refresh_from_db()
        self.assertEqual(skill.name, "Renamed")
        # Org slugs don't follow renames.
        self.assertEqual(skill.slug, "mine")
        # No personal fork was made.
        self.assertFalse(AgentSkill.objects.filter(level="user", created_by=self.member).exists())

    def test_maintainer_cannot_change_org_slug(self):
        skill = self._approved()
        tool = EditSkillTool()
        tool.context = _ctx(self.member)
        out = json.loads(tool._run(skill_slug=skill.slug, updates={"new_slug": "x"}))
        self.assertEqual(out["status"], "error")
        skill.refresh_from_db()
        self.assertEqual(skill.slug, "mine")

    def test_maintainer_cannot_delete_via_tool(self):
        skill = self._approved()
        tool = DeleteSkillTool()
        tool.context = _ctx(self.member)
        out = json.loads(tool._run(skill_slug=skill.slug))
        self.assertEqual(out["status"], "error")
        skill.refresh_from_db()
        self.assertIsNone(skill.deleted_at)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ShareViewTests(_ShareBase):
    def test_share_and_withdraw_views(self):
        self.client.force_login(self.member)
        r = self.client.post(reverse("agent_skills_share", kwargs={"skill_id": self.skill.id}))
        self.assertEqual(r.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.share_requested_org, self.org)
        r = self.client.post(
            reverse("agent_skills_share_withdraw", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(r.status_code, 302)
        self.skill.refresh_from_db()
        self.assertIsNone(self.skill.share_requested_org)

    def test_list_shows_pending_to_admin_and_submitter_only(self):
        request_skill_share(self.member, self.skill)
        url = reverse("agent_skills_list")

        self.client.force_login(self.admin)
        r = self.client.get(url)
        self.assertContains(r, "Needs approval")
        self.assertContains(r, reverse("agent_skills_share_approve", kwargs={"skill_id": self.skill.id}))
        org_ids = [row["skill"].id for row in r.context["main_org_rows"]]
        self.assertIn(self.skill.id, org_ids)

        self.client.force_login(self.member)
        r = self.client.get(url)
        self.assertContains(r, "Needs approval")
        self.assertIn(self.skill.id, [row["skill"].id for row in r.context["main_org_rows"]])
        self.assertNotIn(self.skill.id, [row["skill"].id for row in r.context["main_user_rows"]])

        self.client.force_login(self.other)
        r = self.client.get(url)
        self.assertNotContains(r, "Needs approval")

    def test_admin_detail_is_read_only_review(self):
        request_skill_share(self.member, self.skill)
        self.client.force_login(self.admin)
        r = self.client.get(reverse("agent_skills_detail", kwargs={"skill_id": self.skill.id}))
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.context["is_review"])
        self.assertFalse(r.context["editable"])

    def test_approve_view(self):
        request_skill_share(self.member, self.skill)
        self.client.force_login(self.admin)
        r = self.client.post(
            reverse("agent_skills_share_approve", kwargs={"skill_id": self.skill.id})
        )
        self.assertEqual(r.status_code, 302)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.level, "org")
        self.assertEqual(self.skill.maintainer, self.member)

    def test_member_cannot_approve_view(self):
        request_skill_share(self.member, self.skill)
        self.client.force_login(self.other)
        self.client.post(
            reverse("agent_skills_share_approve", kwargs={"skill_id": self.skill.id})
        )
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.level, "user")

    def test_decline_view(self):
        request_skill_share(self.member, self.skill)
        self.client.force_login(self.admin)
        self.client.post(
            reverse("agent_skills_share_decline", kwargs={"skill_id": self.skill.id})
        )
        self.skill.refresh_from_db()
        self.assertIsNone(self.skill.share_requested_org)

    def test_maintainer_saves_form_but_cannot_delete(self):
        skill = self._approved()
        self.client.force_login(self.member)
        r = self.client.post(
            reverse("agent_skills_save", kwargs={"skill_id": skill.id}),
            {
                "action": "save", "name": "Mine", "description": "",
                "instructions": "edited", "tool_names_json": "[]",
                "templates_json": "[]",
            },
        )
        self.assertEqual(r.status_code, 302)
        skill.refresh_from_db()
        self.assertEqual(skill.instructions, "edited")
        r = self.client.post(reverse("agent_skills_delete", kwargs={"skill_id": skill.id}))
        self.assertEqual(r.status_code, 403)

    def test_maintainer_edit_queues_rescan(self):
        skill = self._approved()
        self.client.force_login(self.member)
        with patch("agent_skills.resources.request_skill_rescan") as rescan:
            self.client.post(
                reverse("agent_skills_save", kwargs={"skill_id": skill.id}),
                {
                    "action": "save", "name": "Mine", "description": "",
                    "instructions": "edited again", "tool_names_json": "[]",
                    "templates_json": "[]",
                },
            )
        rescan.assert_called_once()

    def test_revoke_view(self):
        skill = self._approved()
        self.client.force_login(self.admin)
        self.client.post(
            reverse("agent_skills_revoke_maintainer", kwargs={"skill_id": skill.id})
        )
        skill.refresh_from_db()
        self.assertIsNone(skill.maintainer)


class NavBadgeTests(_ShareBase):
    def _ctx(self, user):
        request = RequestFactory().get("/")
        request.user = self._fresh(user)
        return nav_context(request)

    def test_admin_sees_pending_count(self):
        request_skill_share(self.member, self.skill)
        self.assertEqual(self._ctx(self.admin)["pending_skill_shares_count"], 1)

    def test_member_sees_no_count(self):
        request_skill_share(self.member, self.skill)
        self.assertEqual(self._ctx(self.member)["pending_skill_shares_count"], 0)
