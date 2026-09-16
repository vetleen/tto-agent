"""Tests for agent_skills.tools.DetachSkillsTool (the agent may only detach its own)."""

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from agent_skills.models import AgentSkill
from agent_skills.tools import DetachSkillsTool
from chat.models import ChatThread, ChatThreadSkill
from llm.types import RunContext

User = get_user_model()


def _attached(thread):
    return [
        (r.skill.slug, r.attached_by)
        for r in ChatThreadSkill.objects.filter(thread=thread).select_related("skill")
    ]


class DetachSkillsToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="detach@example.com", password="pass")
        self.thread = ChatThread.objects.create(created_by=self.user, title="t")
        self.mine = AgentSkill.objects.create(
            slug="agent-pick", name="Agent Pick", instructions="x",
            level="user", created_by=self.user,
        )
        self.theirs = AgentSkill.objects.create(
            slug="users-pick", name="Users Pick", instructions="x",
            level="user", created_by=self.user,
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.theirs)  # default: user
        ChatThreadSkill.objects.create(
            thread=self.thread, skill=self.mine, attached_by="agent"
        )
        self.tool = DetachSkillsTool()
        self.tool.context = RunContext.create(
            user_id=self.user.pk, conversation_id=str(self.thread.id)
        )

    def _detach(self, *slugs):
        return json.loads(self.tool._run(skill_slugs=list(slugs)))

    def test_detaches_agent_attached_skill(self):
        result = self._detach("agent-pick")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["no_change"])
        self.assertEqual(result["removed"], ["agent-pick"])
        self.assertEqual(result["removed_names"], ["Agent Pick"])
        self.assertEqual(
            [(s["slug"], s["attached_by"]) for s in result["skills"]],
            [("users-pick", "user")],
        )
        self.assertEqual(_attached(self.thread), [("users-pick", "user")])

    def test_user_attached_skill_is_protected(self):
        result = self._detach("users-pick")
        self.assertEqual(result["status"], "error")
        self.assertIn("attached by the user", result["message"])
        self.assertIn("users-pick", result["message"])
        self.assertEqual(len(_attached(self.thread)), 2)

    def test_mixed_request_is_refused_atomically(self):
        result = self._detach("agent-pick", "users-pick")
        self.assertEqual(result["status"], "error")
        self.assertIn("users-pick", result["message"])
        self.assertNotIn("agent-pick", result["message"])
        self.assertEqual(len(_attached(self.thread)), 2)

    def test_unknown_slug_lists_attached(self):
        result = self._detach("nope")
        self.assertEqual(result["status"], "error")
        self.assertIn("not attached", result["message"])
        self.assertEqual(sorted(result["attached_slugs"]), ["agent-pick", "users-pick"])
        self.assertEqual(len(_attached(self.thread)), 2)

    def test_empty_list_is_noop(self):
        for value in ([], None):
            result = json.loads(self.tool._run(skill_slugs=value))
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["no_change"])
            self.assertEqual(result["removed"], [])
            self.assertEqual(len(result["skills"]), 2)
        self.assertEqual(len(_attached(self.thread)), 2)

    def test_duplicate_and_whitespace_slugs_deduped(self):
        result = self._detach(" agent-pick ", "agent-pick")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["removed"], ["agent-pick"])

    def test_missing_context_returns_error(self):
        self.tool.context = None
        self.assertEqual(self._detach("agent-pick")["status"], "error")

    def test_other_users_thread_not_touchable(self):
        other = User.objects.create_user(email="other@example.com", password="pass")
        other_thread = ChatThread.objects.create(created_by=other, title="t")
        ChatThreadSkill.objects.create(
            thread=other_thread, skill=self.mine, attached_by="agent"
        )
        self.tool.context = RunContext.create(
            user_id=self.user.pk, conversation_id=str(other_thread.id)
        )
        result = self._detach("agent-pick")
        self.assertEqual(result["status"], "error")
        self.assertEqual(_attached(other_thread), [("agent-pick", "agent")])

    def test_leaves_same_turn_state_untouched(self):
        """No mid-turn revocation exists in the pipeline; detach must not
        pretend otherwise by touching the turn's context slots."""
        self.tool.context.skill_tool_map = {"agent-pick": ["skill_resource_view"]}
        self._detach("agent-pick")
        self.assertEqual(self.tool.context.added_tool_names, [])
        self.assertEqual(self.tool.context.pending_skill_instructions, [])

    def test_protection_is_checked_after_the_lock(self):
        """A concurrent write that re-marks the row as user-attached just before
        we get the lock is honoured: the check reads under the lock."""
        def flip(thread_id):
            ChatThreadSkill.objects.filter(thread=self.thread, skill=self.mine).update(
                attached_by="user"
            )

        with patch("chat.thread_skills.lock_thread", side_effect=flip):
            result = self._detach("agent-pick")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(_attached(self.thread)), 2)

    def test_end_label_for_result(self):
        label = self.tool.end_label_for_result
        self.assertIsNone(label({"status": "error"}))
        self.assertEqual(label({"status": "ok", "removed": []}), "No skills detached")
        self.assertEqual(
            label({"status": "ok", "removed": ["a"], "removed_names": ["Alpha"]}),
            "Detached skill: Alpha",
        )
        self.assertEqual(
            label({"status": "ok", "removed": ["a", "b"]}), "Detached 2 skills"
        )

    def test_registered_main_only_chat_section(self):
        from core.preferences import get_preferences
        from llm.tools.registry import get_tool_registry

        registry = get_tool_registry()
        self.assertIn("chat_skill_detach", registry.list_tools())
        tool = registry.get_tool("chat_skill_detach")
        self.assertEqual(tool.audience, "main")
        self.assertEqual(tool.section, "chat")
        prefs = get_preferences(self.user)
        self.assertIn("chat_skill_detach", prefs.allowed_tools)
        self.assertNotIn("chat_skill_detach", prefs.allowed_subagent_tools)
