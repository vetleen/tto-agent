"""Tests for agent_skills.tools.AttachSkillsTool."""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from agent_skills.tools import AttachSkillsTool
from chat.models import ChatThread
from llm.types import RunContext

User = get_user_model()


def _ctx(user, thread):
    return RunContext.create(user_id=user.pk, conversation_id=str(thread.id))


class AttachSkillsToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="attach@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="my-skill", name="My Skill", instructions="Do the thing.",
            description="Does the thing.", level="user", created_by=self.user,
        )
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.tool = AttachSkillsTool()
        self.tool.context = _ctx(self.user, self.thread)

    def test_attach_single_slug_updates_thread(self):
        result = json.loads(self.tool._run(skill_slugs=["my-skill"]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["attached_skill_name"], "My Skill")
        self.assertEqual(result["attached_skill_id"], str(self.skill.id))
        self.assertFalse(result.get("detached"))
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.skill_id, self.skill.id)

    def test_empty_list_detaches(self):
        self.thread.skill = self.skill
        self.thread.save(update_fields=["skill"])
        result = json.loads(self.tool._run(skill_slugs=[]))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["detached"])
        self.assertIsNone(result["attached_skill_id"])
        self.assertEqual(result["previous_skill_name"], "My Skill")
        self.thread.refresh_from_db()
        self.assertIsNone(self.thread.skill_id)

    def test_empty_list_when_nothing_attached_is_noop(self):
        result = json.loads(self.tool._run(skill_slugs=[]))
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["detached"])
        self.assertIsNone(result["attached_skill_id"])
        self.thread.refresh_from_db()
        self.assertIsNone(self.thread.skill_id)

    def test_len_greater_than_one_returns_error(self):
        other = AgentSkill.objects.create(
            slug="other", name="Other", instructions="x",
            level="user", created_by=self.user,
        )
        result = json.loads(self.tool._run(skill_slugs=["my-skill", "other"]))
        self.assertEqual(result["status"], "error")
        self.thread.refresh_from_db()
        self.assertIsNone(self.thread.skill_id)
        # Touch var so linters don't complain
        self.assertIsNotNone(other.id)

    def test_unknown_slug_returns_error_with_available_slugs(self):
        result = json.loads(self.tool._run(skill_slugs=["does-not-exist"]))
        self.assertEqual(result["status"], "error")
        self.assertIn("my-skill", result["available_slugs"])
        self.thread.refresh_from_db()
        self.assertIsNone(self.thread.skill_id)

    def test_same_slug_reattach_is_noop(self):
        self.thread.skill = self.skill
        self.thread.save(update_fields=["skill"])
        result = json.loads(self.tool._run(skill_slugs=["my-skill"]))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result.get("no_change"))
        self.assertEqual(result["attached_skill_id"], str(self.skill.id))

    def test_other_users_user_skill_not_attachable(self):
        other_user = User.objects.create_user(email="other@example.com", password="pass")
        AgentSkill.objects.create(
            slug="private", name="Private", instructions="x",
            level="user", created_by=other_user,
        )
        result = json.loads(self.tool._run(skill_slugs=["private"]))
        self.assertEqual(result["status"], "error")
        self.thread.refresh_from_db()
        self.assertIsNone(self.thread.skill_id)

    def test_org_disabled_slug_not_attachable(self):
        org = Organization.objects.create(name="Org")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)
        AgentSkill.objects.create(
            slug="org-skill", name="Org Skill", instructions="x",
            level="org", organization=org,
        )
        org.preferences = {"skills": {"org-skill": {"enabled": False}}}
        org.save(update_fields=["preferences"])
        result = json.loads(self.tool._run(skill_slugs=["org-skill"]))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("org-skill", result["available_slugs"])

    def test_missing_context_returns_error(self):
        self.tool.context = None
        result = json.loads(self.tool._run(skill_slugs=["my-skill"]))
        self.assertEqual(result["status"], "error")

    def test_thread_belonging_to_other_user_not_attachable(self):
        other_user = User.objects.create_user(email="other2@example.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user, title="t")
        self.tool.context = RunContext.create(
            user_id=self.user.pk, conversation_id=str(other_thread.id),
        )
        result = json.loads(self.tool._run(skill_slugs=["my-skill"]))
        self.assertEqual(result["status"], "error")

    def test_attach_then_swap(self):
        second = AgentSkill.objects.create(
            slug="second", name="Second", instructions="x",
            level="user", created_by=self.user,
        )
        json.loads(self.tool._run(skill_slugs=["my-skill"]))
        result = json.loads(self.tool._run(skill_slugs=["second"]))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["detached"])
        self.assertEqual(result["previous_skill_name"], "My Skill")
        self.assertEqual(result["attached_skill_id"], str(second.id))
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.skill_id, second.id)

    def test_whitespace_in_slug_stripped(self):
        result = json.loads(self.tool._run(skill_slugs=["  my-skill  "]))
        self.assertEqual(result["status"], "ok")
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.skill_id, self.skill.id)

    def test_none_slugs_treated_as_empty(self):
        result = json.loads(self.tool._run(skill_slugs=None))
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["detached"])

    def test_tool_registered(self):
        from llm.tools.registry import get_tool_registry
        self.assertIn("attach_skills", get_tool_registry().list_tools())
