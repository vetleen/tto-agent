"""Tests for agent_skills.tools.AttachSkillsTool (declarative multi-skill set)."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import MAX_THREAD_SKILLS, AgentSkill
from agent_skills.tools import AttachSkillsTool
from chat.models import ChatThread, ChatThreadSkill
from llm.types import RunContext

User = get_user_model()


def _ctx(user, thread):
    return RunContext.create(user_id=user.pk, conversation_id=str(thread.id))


def _attached_ids(thread):
    """Attached skill ids in attach order (the through-model ordering)."""
    return [
        str(sid)
        for sid in ChatThreadSkill.objects.filter(thread=thread).values_list(
            "skill_id", flat=True
        )
    ]


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

    def _attach(self, *slugs):
        return json.loads(self.tool._run(skill_slugs=list(slugs)))

    def test_attach_single_slug_updates_thread(self):
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["skills"], [
            {"id": str(self.skill.id), "name": "My Skill", "emoji": self.skill.emoji},
        ])
        self.assertEqual(result["added"], [str(self.skill.id)])
        self.assertEqual(result["removed"], [])
        self.assertFalse(result["no_change"])
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_attach_multiple_up_to_cap(self):
        slugs = ["my-skill"]
        ids = [str(self.skill.id)]
        for i in range(1, MAX_THREAD_SKILLS):
            s = AgentSkill.objects.create(
                slug=f"s{i}", name=f"S{i}", instructions="x",
                level="user", created_by=self.user,
            )
            slugs.append(s.slug)
            ids.append(str(s.id))
        result = self._attach(*slugs)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([s["id"] for s in result["skills"]], ids)
        self.assertEqual(_attached_ids(self.thread), ids)

    def test_over_cap_rejected(self):
        slugs = ["my-skill"]
        for i in range(MAX_THREAD_SKILLS):  # one too many overall
            s = AgentSkill.objects.create(
                slug=f"x{i}", name=f"X{i}", instructions="x",
                level="user", created_by=self.user,
            )
            slugs.append(s.slug)
        result = self._attach(*slugs)
        self.assertEqual(result["status"], "error")
        self.assertEqual(_attached_ids(self.thread), [])

    def test_empty_list_detaches_all(self):
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        result = self._attach()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["skills"], [])
        self.assertEqual(result["removed"], [str(self.skill.id)])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_empty_list_when_nothing_attached_is_noop(self):
        result = self._attach()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])
        self.assertEqual(result["skills"], [])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_declarative_replace_diffs_added_and_removed(self):
        second = AgentSkill.objects.create(
            slug="second", name="Second", instructions="x",
            level="user", created_by=self.user,
        )
        self._attach("my-skill")
        result = self._attach("second")  # full replace, not additive
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["added"], [str(second.id)])
        self.assertEqual(result["removed"], [str(self.skill.id)])
        self.assertEqual(_attached_ids(self.thread), [str(second.id)])

    def test_same_set_is_noop(self):
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])
        self.assertEqual([s["id"] for s in result["skills"]], [str(self.skill.id)])

    def test_duplicate_slugs_deduped(self):
        result = self._attach("my-skill", "my-skill")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_unknown_slug_returns_error_with_available_slugs(self):
        result = self._attach("does-not-exist")
        self.assertEqual(result["status"], "error")
        self.assertIn("my-skill", result["available_slugs"])
        self.assertEqual(_attached_ids(self.thread), [])

    def test_one_unknown_slug_rejects_whole_set(self):
        result = self._attach("my-skill", "nope")
        self.assertEqual(result["status"], "error")
        # Nothing persisted — the set is rejected atomically.
        self.assertEqual(_attached_ids(self.thread), [])

    def test_other_users_user_skill_not_attachable(self):
        other_user = User.objects.create_user(email="other@example.com", password="pass")
        AgentSkill.objects.create(
            slug="private", name="Private", instructions="x",
            level="user", created_by=other_user,
        )
        result = self._attach("private")
        self.assertEqual(result["status"], "error")
        self.assertEqual(_attached_ids(self.thread), [])

    def test_org_disabled_slug_not_attachable(self):
        org = Organization.objects.create(name="Org")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)
        AgentSkill.objects.create(
            slug="org-skill", name="Org Skill", instructions="x",
            level="org", organization=org,
        )
        org.preferences = {"skills": {"org-skill": {"enabled": False}}}
        org.save(update_fields=["preferences"])
        result = self._attach("org-skill")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("org-skill", result["available_slugs"])

    def test_missing_context_returns_error(self):
        self.tool.context = None
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")

    def test_thread_belonging_to_other_user_not_attachable(self):
        other_user = User.objects.create_user(email="other2@example.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other_user, title="t")
        self.tool.context = RunContext.create(
            user_id=self.user.pk, conversation_id=str(other_thread.id),
        )
        result = self._attach("my-skill")
        self.assertEqual(result["status"], "error")

    def test_whitespace_in_slug_stripped(self):
        result = json.loads(self.tool._run(skill_slugs=["  my-skill  "]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(_attached_ids(self.thread), [str(self.skill.id)])

    def test_none_slugs_treated_as_empty(self):
        result = json.loads(self.tool._run(skill_slugs=None))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["no_change"])

    def test_tool_registered(self):
        from llm.tools.registry import get_tool_registry
        self.assertIn("chat_skill_attach", get_tool_registry().list_tools())
