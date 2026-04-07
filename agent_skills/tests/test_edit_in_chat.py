"""Tests for the "Edit skill in chat" flow.

Covers:
- ``skills_edit_in_chat`` view (thread + canvases + seed message)
- ``resolve_skill_for_thread_edit`` (fork-on-write routing)
- End-to-end through the relevant tools
"""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill
from agent_skills.tools import (
    EditSkillTool,
    SaveCanvasToSkillFieldTool,
    resolve_skill_for_thread_edit,
)
from chat.models import ChatCanvas, ChatMessage, ChatThread
from llm.types import RunContext

User = get_user_model()


def _seed_skill_creator():
    """Create the system Skill Creator skill that the view looks up."""
    return AgentSkill.objects.create(
        slug="skill-creator",
        name="Skill Creator",
        instructions="Help the user create and edit skills.",
        level="system",
    )


def _make_canvas(thread, title, content):
    """Create a canvas with content for tests that need to save to a field."""
    return ChatCanvas.objects.create(thread=thread, title=title, content=content)


# ----- View tests -------------------------------------------------------


@override_settings(ALLOWED_HOSTS=["testserver"])
class EditInChatViewTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="eic@example.com", password="pw")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.skill_creator = _seed_skill_creator()
        self.user_skill = AgentSkill.objects.create(
            slug="user-x",
            name="User X",
            description="A user skill description.",
            instructions="A user skill instructions.",
            level="user",
            created_by=self.user,
        )
        self.system_skill = AgentSkill.objects.create(
            slug="sys-x",
            name="Sys X",
            description="System description.",
            instructions="System instructions.",
            level="system",
        )

    def test_requires_login(self):
        response = self.client.post(
            reverse("agent_skills_edit_in_chat", args=[self.user_skill.id])
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response["Location"])

    def test_user_skill_owner_creates_thread(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_edit_in_chat", args=[self.user_skill.id])
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("thread=", response["Location"])

        thread = ChatThread.objects.get(created_by=self.user)
        self.assertEqual(thread.skill_id, self.skill_creator.id)
        self.assertEqual(thread.title, "Editing User X")
        self.assertEqual(thread.metadata.get("source_skill_id"), str(self.user_skill.id))
        self.assertTrue(thread.metadata.get("pending_initial_turn"))

        canvases = list(thread.canvases.all())
        titles = sorted(c.title for c in canvases)
        self.assertEqual(titles, ["User X \u2014 description", "User X \u2014 instructions"])

        # Instructions canvas should be the active one (loaded last).
        thread.refresh_from_db()
        self.assertIsNotNone(thread.active_canvas_id)
        self.assertEqual(thread.active_canvas.title, "User X \u2014 instructions")

        # A hidden seed message exists with the skill name in it.
        seed = ChatMessage.objects.get(thread=thread, is_hidden_from_user=True)
        self.assertEqual(seed.role, "user")
        self.assertIn("User X", seed.content)
        self.assertIn("description", seed.content)
        self.assertIn("instructions", seed.content)

    def test_system_skill_visible_to_anyone_creates_thread(self):
        """Non-editable skills are still openable — fork happens on save."""
        other = User.objects.create_user(email="other@example.com", password="pw")
        other.email_verified = True
        other.save(update_fields=["email_verified"])
        self.client.force_login(other)

        response = self.client.post(
            reverse("agent_skills_edit_in_chat", args=[self.system_skill.id])
        )
        self.assertEqual(response.status_code, 302)

        thread = ChatThread.objects.get(created_by=other)
        self.assertEqual(thread.metadata.get("source_skill_id"), str(self.system_skill.id))
        # System skill is unchanged.
        self.system_skill.refresh_from_db()
        self.assertEqual(self.system_skill.name, "Sys X")

    def test_unknown_skill_redirects(self):
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_edit_in_chat", args=[uuid.uuid4()])
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("agent_skills_list"), response["Location"])
        self.assertEqual(ChatThread.objects.count(), 0)

    def test_missing_skill_creator_fails_gracefully(self):
        self.skill_creator.delete()
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("agent_skills_edit_in_chat", args=[self.user_skill.id])
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse("agent_skills_list"), response["Location"])
        self.assertEqual(ChatThread.objects.count(), 0)


# ----- Fork-on-write tests ----------------------------------------------


class ResolveSkillForThreadEditTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="resolve@example.com", password="pw")
        self.skill_creator = _seed_skill_creator()
        self.editable = AgentSkill.objects.create(
            slug="mine", name="Mine", instructions="i", description="d",
            level="user", created_by=self.user,
        )
        self.system = AgentSkill.objects.create(
            slug="sys", name="Sys", instructions="i", description="d", level="system",
        )

    def _make_thread(self, source_skill_id):
        return ChatThread.objects.create(
            created_by=self.user,
            skill=self.skill_creator,
            metadata={"source_skill_id": str(source_skill_id)},
        )

    def test_editable_source_returned_in_place(self):
        thread = self._make_thread(self.editable.id)
        skill, err = resolve_skill_for_thread_edit(self.user, thread.id, "mine")
        self.assertIsNone(err)
        self.assertEqual(skill.pk, self.editable.pk)

    def test_non_editable_source_forks_and_updates_metadata(self):
        thread = self._make_thread(self.system.id)
        skill, err = resolve_skill_for_thread_edit(self.user, thread.id, "sys")
        self.assertIsNone(err)
        self.assertEqual(skill.level, "user")
        self.assertEqual(skill.created_by_id, self.user.pk)
        self.assertNotEqual(skill.pk, self.system.pk)

        thread.refresh_from_db()
        self.assertEqual(thread.metadata["source_skill_id"], str(skill.id))

        # System skill is unchanged.
        self.system.refresh_from_db()
        self.assertEqual(self.system.name, "Sys")

    def test_second_call_targets_fork_no_duplicate(self):
        thread = self._make_thread(self.system.id)
        first, _ = resolve_skill_for_thread_edit(self.user, thread.id, "sys")
        second, _ = resolve_skill_for_thread_edit(self.user, thread.id, "sys")
        self.assertEqual(first.pk, second.pk)
        # Only one user-tier copy of "sys" exists.
        self.assertEqual(
            AgentSkill.objects.filter(level="user", parent=self.system).count(), 1
        )

    def test_no_source_falls_back_to_slug(self):
        # Thread without source_skill_id uses legacy slug-based lookup.
        thread = ChatThread.objects.create(
            created_by=self.user, skill=self.skill_creator, metadata={},
        )
        skill, err = resolve_skill_for_thread_edit(self.user, thread.id, "mine")
        self.assertIsNone(err)
        self.assertEqual(skill.pk, self.editable.pk)

    def test_no_source_unknown_slug_errors(self):
        thread = ChatThread.objects.create(
            created_by=self.user, skill=self.skill_creator, metadata={},
        )
        skill, err = resolve_skill_for_thread_edit(self.user, thread.id, "unknown")
        self.assertIsNone(skill)
        self.assertIn("not found", err)


class SaveCanvasForkOnWriteTests(TestCase):
    """End-to-end: SaveCanvasToSkillFieldTool routes through fork-on-write."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="save@example.com", password="pw")
        self.skill_creator = _seed_skill_creator()
        self.system = AgentSkill.objects.create(
            slug="sys-save",
            name="Sys Save",
            instructions="orig",
            description="orig",
            level="system",
        )
        self.thread = ChatThread.objects.create(
            created_by=self.user,
            skill=self.skill_creator,
            metadata={"source_skill_id": str(self.system.id)},
        )
        _make_canvas(self.thread, "Sys Save \u2014 instructions", "edited content")

        self.tool = SaveCanvasToSkillFieldTool()
        self.tool.context = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(self.thread.id),
        )

    def test_save_to_non_editable_forks_on_write(self):
        result = json.loads(self.tool._run(
            skill_slug="sys-save",
            field_name="instructions",
            canvas_name="Sys Save \u2014 instructions",
        ))
        self.assertEqual(result["status"], "ok")

        # System skill unchanged.
        self.system.refresh_from_db()
        self.assertEqual(self.system.instructions, "orig")

        # A user-tier fork now exists with the edited content.
        forks = AgentSkill.objects.filter(level="user", created_by=self.user, parent=self.system)
        self.assertEqual(forks.count(), 1)
        self.assertEqual(forks.first().instructions, "edited content")

        # Thread metadata now points at the fork.
        self.thread.refresh_from_db()
        self.assertEqual(self.thread.metadata["source_skill_id"], str(forks.first().id))


class EditSkillToolForkOnWriteTests(TestCase):
    """EditSkillTool also routes through fork-on-write."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="editfork@example.com", password="pw")
        self.skill_creator = _seed_skill_creator()
        self.system = AgentSkill.objects.create(
            slug="sys-edit",
            name="Sys Edit",
            instructions="i",
            description="orig desc",
            level="system",
        )
        self.thread = ChatThread.objects.create(
            created_by=self.user,
            skill=self.skill_creator,
            metadata={"source_skill_id": str(self.system.id)},
        )
        self.tool = EditSkillTool()
        self.tool.context = RunContext.create(
            user_id=self.user.pk,
            conversation_id=str(self.thread.id),
        )

    def test_text_edit_on_non_editable_forks(self):
        result = json.loads(self.tool._run(
            skill_slug="sys-edit",
            text_edits=[{
                "field": "description",
                "old_text": "orig desc",
                "new_text": "new desc",
            }],
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["edits_applied"], 1)

        self.system.refresh_from_db()
        self.assertEqual(self.system.description, "orig desc")

        fork = AgentSkill.objects.get(level="user", created_by=self.user, parent=self.system)
        self.assertEqual(fork.description, "new desc")
