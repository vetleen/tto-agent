"""Tests for agent_skills.tools — skill management tools."""

import json
import uuid
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.tools import (
    CreateSkillTool,
    DeleteSkillTool,
    EditSkillTool,
    InspectToolTool,
    ListSkillToolsTool,
    LoadTemplateToCanvasTool,
    SaveCanvasToSkillFieldTool,
    ShowSkillFieldInCanvasTool,
    SkillResourceAttachTool,
    SkillResourceDeleteTool,
    SkillResourceListTool,
    SkillResourceSaveTool,
    SkillResourceUpdateTool,
    ViewTemplateTool,
)
from llm.types import RunContext

User = get_user_model()


def _make_context(user, thread_id=None):
    return RunContext.create(
        user_id=user.pk,
        conversation_id=thread_id or str(uuid.uuid4()),
    )


class CreateSkillToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="tool@example.com", password="pass")
        self.tool = CreateSkillTool()
        self.tool.context = _make_context(self.user)

    @patch("agent_skills.tools._generate_emoji_for_skill", return_value="")
    def test_create_skill(self, mock_gen):
        result = json.loads(self.tool._run(name="My Skill"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "My Skill")
        self.assertEqual(result["slug"], "my-skill")
        self.assertTrue(AgentSkill.objects.filter(slug="my-skill", created_by=self.user).exists())

    @patch("agent_skills.tools._generate_emoji_for_skill", return_value="🧪")
    def test_create_skill_saves_generated_emoji(self, mock_gen):
        result = json.loads(self.tool._run(name="Lab Notes"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["emoji"], "🧪")
        skill = AgentSkill.objects.get(slug="lab-notes", created_by=self.user)
        self.assertEqual(skill.emoji, "🧪")
        mock_gen.assert_called_once()

    @patch("agent_skills.tools._generate_emoji_for_skill", side_effect=RuntimeError("llm down"))
    def test_create_skill_ignores_emoji_failure(self, mock_gen):
        result = json.loads(self.tool._run(name="Resilient"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["emoji"], "")
        skill = AgentSkill.objects.get(slug="resilient", created_by=self.user)
        self.assertEqual(skill.emoji, "")

    @patch("agent_skills.tools._generate_emoji_for_skill", return_value="")
    def test_create_skill_passes_org_id_to_emoji(self, mock_gen):
        """The user's org is forwarded so the org-scoped skill_emoji model wins."""
        org = Organization.objects.create(name="Acme", slug="acme")
        Membership.objects.create(user=self.user, org=org, role=Membership.Role.ADMIN)
        self.tool._run(name="Org Skill")
        self.assertEqual(mock_gen.call_args.kwargs.get("org_id"), org.id)

    @patch("agent_skills.tools._generate_emoji_for_skill", return_value="")
    def test_create_skill_org_id_none_without_membership(self, mock_gen):
        """Unaffiliated user → org_id None (resolver falls back to system default)."""
        self.tool._run(name="Solo Skill")
        self.assertEqual(mock_gen.call_args.kwargs.get("org_id"), None)

    @patch("agent_skills.tools._generate_emoji_for_skill", return_value="")
    def test_create_skill_caps_long_name(self, mock_gen):
        """A >255-char name is capped before the CharField (DataError on
        Postgres otherwise; SQLite silently accepts the overflow)."""
        result = json.loads(self.tool._run(name="N" * 300))
        self.assertEqual(result["status"], "ok")
        skill = AgentSkill.objects.get(created_by=self.user)
        self.assertEqual(len(skill.name), 255)


class EditSkillToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="edit@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="editable", name="Editable", instructions="Original instructions.",
            description="Original desc.", level="user", created_by=self.user,
        )
        self.tool = EditSkillTool()
        self.tool.context = _make_context(self.user)

    def test_update_name(self):
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"name": "Renamed"}, text_edits=[],
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "Renamed")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "Renamed")

    def test_new_slug_is_slugified(self):
        """A raw new_slug with spaces/uppercase is slugified before saving."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"new_slug": "My New Slug"},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["slug"], "my-new-slug")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.slug, "my-new-slug")

    def test_new_slug_truncated_to_64(self):
        """An over-long new_slug is capped at the SlugField's 64-char limit."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"new_slug": "a" * 200},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.slug, "a" * 64)

    def test_new_slug_empty_falls_back_to_name(self):
        """A new_slug that slugifies to empty falls back to the name-derived
        slug (rather than erroring) and still freezes the slug."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"new_slug": "!!!"},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        # name "Editable" -> "editable" (unchanged here, but name-derived).
        self.assertEqual(self.skill.slug, "editable")
        self.assertTrue(self.skill.slug_customized)

    def test_new_slug_dedupes_instead_of_erroring(self):
        """A new_slug colliding with another of the user's skills gets a
        running-number suffix instead of a hard error."""
        AgentSkill.objects.create(
            slug="taken", name="Taken", instructions="i",
            level="user", created_by=self.user,
        )
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"new_slug": "taken"},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["slug"], "taken-1")
        self.skill.refresh_from_db()
        self.assertTrue(self.skill.slug_customized)

    def test_rename_reslugs_when_not_customized(self):
        """Changing the name re-derives the slug while it is not customized."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"name": "Patent Analyzer"},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["slug"], "patent-analyzer")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.slug, "patent-analyzer")

    def test_rename_keeps_slug_when_customized(self):
        """Once the slug is customized, a rename no longer moves it."""
        self.skill.slug_customized = True
        self.skill.save(update_fields=["slug_customized"])
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"name": "Patent Analyzer"},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.slug, "editable")
        self.assertEqual(self.skill.name, "Patent Analyzer")

    def test_find_replace_description(self):
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={},
            text_edits=[{"field": "description", "old_text": "Original desc.", "new_text": "Updated desc."}],
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["edits_applied"], 1)
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.description, "Updated desc.")

    def test_tool_names_filters_out_standard_tools(self):
        """Standard (chat-section) tools are silently removed from tool_names."""
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={"tool_names": ["skill_resource_view", "chat_task_update", "chat_subagent_create"]},
        ))
        self.assertEqual(result["status"], "ok")
        # chat_task_update and chat_subagent_create are chat-section tools — silently removed
        self.assertEqual(result["tool_names"], ["skill_resource_view"])
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.tool_names, ["skill_resource_view"])

    def test_tool_names_all_standard_results_in_empty_list(self):
        """If only standard tools are passed, tool_names becomes empty."""
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={"tool_names": ["chat_task_update", "chat_subagent_create"]},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool_names"], [])

    def test_tool_names_drops_unknown_tools(self):
        """Unknown tool names are dropped (allow-list), not passed through."""
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={"tool_names": ["skill_resource_view", "totally_made_up_tool"]},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool_names"], ["skill_resource_view"])
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.tool_names, ["skill_resource_view"])

    def test_find_replace_caps_instructions(self):
        """A find-replace that would grow instructions past the cap is clamped."""
        from agent_skills.models import MAX_INSTRUCTIONS_CHARS

        self.skill.instructions = "START"
        self.skill.save(update_fields=["instructions"])
        huge = "z" * (MAX_INSTRUCTIONS_CHARS + 1000)
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={},
            text_edits=[{"field": "instructions", "old_text": "START", "new_text": huge}],
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(len(self.skill.instructions), MAX_INSTRUCTIONS_CHARS)

    def test_find_replace_caps_description(self):
        """A find-replace that would grow description past 1024 is clamped."""
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={},
            text_edits=[{
                "field": "description",
                "old_text": "Original desc.",
                "new_text": "z" * 2000,
            }],
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(len(self.skill.description), 1024)

    def test_update_name_caps_at_255(self):
        """A >255-char name is capped at the CharField limit (DataError on
        Postgres otherwise)."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"name": "N" * 300},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(len(self.skill.name), 255)

    def test_update_name_blank_ignored(self):
        """A blank name is skipped, mirroring the save form's fallback."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"name": "   "},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.name, "Editable")

    def test_is_active_not_editable(self):
        """Regression: edit_skill accepted is_active=False, but every skill
        lookup filters is_active=True — a deactivated skill became invisible
        and unrecoverable outside the Django admin."""
        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"is_active": False},
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertTrue(self.skill.is_active)

    def test_new_slug_migrates_user_prefs(self):
        """Renaming the slug via edit_skill carries the user's slug-keyed
        selection over to the new slug."""
        from accounts.models import UserSettings

        us, _ = UserSettings.objects.get_or_create(user=self.user)
        us.preferences = {
            "skills": {"editable": {"selected_skill_id": str(self.skill.id)}}
        }
        us.save(update_fields=["preferences"])

        result = json.loads(self.tool._run(
            skill_slug="editable", updates={"new_slug": "renamed"},
        ))
        self.assertEqual(result["status"], "ok")
        us.refresh_from_db()
        skills_prefs = us.preferences["skills"]
        self.assertNotIn("editable", skills_prefs)
        self.assertEqual(
            skills_prefs["renamed"]["selected_skill_id"], str(self.skill.id)
        )

    def test_system_skill_not_editable(self):
        AgentSkill.objects.create(
            slug="sys", name="System", instructions="Inst.", level="system",
        )
        result = json.loads(self.tool._run(skill_slug="sys", updates={"name": "Hacked"}))
        self.assertEqual(result["status"], "error")


class DeleteSkillToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="del@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="doomed", name="Doomed", instructions="Inst.",
            level="user", created_by=self.user,
        )
        self.tool = DeleteSkillTool()
        self.tool.context = _make_context(self.user)

    def test_delete_skill(self):
        result = json.loads(self.tool._run(skill_slug="doomed"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deleted"], "doomed")
        # Soft-delete: the row is retained (restorable from admin) but hidden.
        self.skill.refresh_from_db()
        self.assertIsNotNone(self.skill.deleted_at)
        self.assertFalse(self.skill.is_active)
        # ...and no longer resolvable for editing/deleting via the tool gate.
        from agent_skills.services import get_editable_skill_for_user
        self.assertIsNone(get_editable_skill_for_user(self.user, "doomed"))

    def test_delete_system_skill_denied(self):
        AgentSkill.objects.create(
            slug="sys", name="System", instructions="Inst.", level="system",
        )
        result = json.loads(self.tool._run(skill_slug="sys"))
        self.assertEqual(result["status"], "error")

    def test_delete_other_user_skill_denied(self):
        other = User.objects.create_user(email="other@example.com", password="pass")
        AgentSkill.objects.create(
            slug="other", name="Other", instructions="Inst.",
            level="user", created_by=other,
        )
        result = json.loads(self.tool._run(skill_slug="other"))
        self.assertEqual(result["status"], "error")


class SaveCanvasToSkillFieldToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="canvas@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="canvas-skill", name="Canvas Skill", instructions="Old inst.",
            level="user", created_by=self.user,
        )
        # Create a thread and canvas
        from chat.models import ChatCanvas, ChatThread

        self.thread = ChatThread.objects.create(created_by=self.user)
        from django.utils import timezone

        self.canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Draft", content="Canvas content here.",
            is_active=True, last_activated_at=timezone.now(),
        )
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])
        self.tool = SaveCanvasToSkillFieldTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_save_instructions(self):
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name="instructions"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["field"], "instructions")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.instructions, "Canvas content here.")

    def test_save_description(self):
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name="description"))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.description, "Canvas content here.")

    def test_template_field_name_rejected(self):
        """skill_field_save no longer writes resources — a non-column field_name
        is rejected (resources go through skill_resource_save/attach)."""
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name="Patent Claim"))
        self.assertEqual(result["status"], "error")
        self.assertIn("skill_resource_save", result["message"])
        self.assertFalse(SkillTemplate.objects.filter(skill=self.skill, name="Patent Claim").exists())

    def test_not_editable_denied(self):
        AgentSkill.objects.create(
            slug="sys", name="System", instructions="Inst.", level="system",
        )
        result = json.loads(self.tool._run(skill_slug="sys", field_name="instructions"))
        self.assertEqual(result["status"], "error")

    def test_empty_field_name_rejected(self):
        """A blank field_name is rejected and creates no junk template."""
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name=""))
        self.assertEqual(result["status"], "error")
        self.assertFalse(SkillTemplate.objects.filter(skill=self.skill).exists())

    def test_whitespace_field_name_rejected(self):
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name="   "))
        self.assertEqual(result["status"], "error")
        self.assertFalse(SkillTemplate.objects.filter(skill=self.skill).exists())

    def test_save_from_named_canvas(self):
        """canvas_name parameter targets a specific canvas by title."""
        from chat.models import ChatCanvas

        ChatCanvas.objects.create(
            thread=self.thread, title="Instructions Draft", content="Named canvas content.",
        )
        result = json.loads(self.tool._run(
            skill_slug="canvas-skill", field_name="instructions", canvas_name="Instructions Draft",
        ))
        self.assertEqual(result["status"], "ok")
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.instructions, "Named canvas content.")

    def test_save_description_caps_at_1024(self):
        """An oversized canvas saved to description is capped at the field's
        1024 limit — it is injected verbatim into the system prompt of every
        thread using the skill."""
        self.canvas.content = "z" * 2000
        self.canvas.save(update_fields=["content"])
        result = json.loads(self.tool._run(
            skill_slug="canvas-skill", field_name="description",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["chars_saved"], 1024)
        self.skill.refresh_from_db()
        self.assertEqual(len(self.skill.description), 1024)

class ShowSkillFieldInCanvasToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="show@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="show-skill", name="Show Skill",
            instructions="Skill instructions here.",
            description="Skill desc.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=self.skill, name="My Template", content="Template content.",
        )
        from chat.models import ChatThread

        self.thread = ChatThread.objects.create(created_by=self.user)
        self.tool = ShowSkillFieldInCanvasTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_show_instructions(self):
        result = json.loads(self.tool._run(skill_slug="show-skill", field_name="instructions"))
        self.assertEqual(result["status"], "ok")
        self.assertIn("title", result)
        self.assertNotIn("content", result)
        self.assertNotIn("accepted_content", result)

    def test_template_field_name_rejected(self):
        """skill_field_load loads only the text columns; a resource name is
        rejected with a pointer to skill_resource_load."""
        result = json.loads(self.tool._run(skill_slug="show-skill", field_name="My Template"))
        self.assertEqual(result["status"], "error")
        self.assertIn("skill_resource_load", result["message"])

    def test_nonexistent_field_rejected(self):
        result = json.loads(self.tool._run(skill_slug="show-skill", field_name="No Such"))
        self.assertEqual(result["status"], "error")

    def test_nonexistent_skill(self):
        result = json.loads(self.tool._run(skill_slug="no-such-skill", field_name="instructions"))
        self.assertEqual(result["status"], "error")

    def test_returns_canvas_id(self):
        result = json.loads(self.tool._run(skill_slug="show-skill", field_name="instructions"))
        self.assertIn("canvas_id", result)

    def test_sets_active_canvas(self):
        from chat.models import ChatCanvas

        json.loads(self.tool._run(skill_slug="show-skill", field_name="instructions"))
        self.thread.refresh_from_db()
        self.assertIsNotNone(self.thread.active_canvas)
        canvas = ChatCanvas.objects.get(pk=self.thread.active_canvas_id)
        self.assertIn("Show Skill", canvas.title)

    def test_custom_canvas_name(self):
        result = json.loads(self.tool._run(
            skill_slug="show-skill", field_name="instructions", canvas_name="My Custom Tab",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "My Custom Tab")
        from chat.models import ChatCanvas
        self.assertTrue(ChatCanvas.objects.filter(thread=self.thread, title="My Custom Tab").exists())


class ListSkillToolsToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="listtools@example.com", password="pass")
        self.tool = ListSkillToolsTool()
        self.tool.context = _make_context(self.user)

    def test_returns_only_skill_tools(self):
        result = json.loads(self.tool._run())
        self.assertEqual(result["status"], "ok")
        self.assertIn("tools", result)
        tool_names = [t["name"] for t in result["tools"]]
        self.assertIn("skill_create", tool_names)
        self.assertIn("skill_tool_inspect", tool_names)
        self.assertNotIn("chat_task_update", tool_names)
        self.assertNotIn("chat_subagent_create", tool_names)

    def test_each_entry_has_name_and_description(self):
        result = json.loads(self.tool._run())
        for tool_entry in result["tools"]:
            self.assertIn("name", tool_entry)
            self.assertIn("description", tool_entry)


class InspectToolToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="inspect@example.com", password="pass")
        self.tool = InspectToolTool()
        self.tool.context = _make_context(self.user)

    def test_inspect_existing_tool(self):
        result = json.loads(self.tool._run(tool_name="skill_create"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "skill_create")
        self.assertIn("description", result)
        self.assertTrue(len(result["description"]) > 0)

    def test_inspect_nonexistent_tool(self):
        result = json.loads(self.tool._run(tool_name="no_such_tool"))
        self.assertEqual(result["status"], "error")


class ViewTemplateToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="viewtmpl@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="vt-skill", name="VT Skill", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=self.skill, name="Report Format", content="# Title\n## Summary\n## Details",
        )
        from chat.models import ChatThread, ChatThreadSkill

        self.thread = ChatThread.objects.create(created_by=self.user)
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        self.tool = ViewTemplateTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_view_existing_template(self):
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["resource_name"], "Report Format")
        # Content is wrapped in begin/end markers so a later detach stays legible.
        self.assertIn("# Title\n## Summary\n## Details", result["content"])
        self.assertIn('begin resource "Report Format"', result["content"])
        self.assertIn('end resource "Report Format"', result["content"])

    def test_view_template_from_second_attached_skill(self):
        """A template on any attached skill resolves, not just the first."""
        from chat.models import ChatThreadSkill

        other = AgentSkill.objects.create(
            slug="vt-skill-2", name="VT Skill 2", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=other, name="Other Tmpl", content="other body",
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=other)
        result = json.loads(self.tool._run(template_name="Other Tmpl"))
        self.assertEqual(result["status"], "ok")
        self.assertIn("other body", result["content"])

    def test_view_template_collision_picks_first_attached_with_note(self):
        """When two attached skills share a template name, the earliest-attached
        skill wins and a note flags the collision."""
        from chat.models import ChatThreadSkill

        other = AgentSkill.objects.create(
            slug="vt-skill-3", name="VT Skill 3", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=other, name="Report Format", content="LOSER body",
        )
        ChatThreadSkill.objects.create(thread=self.thread, skill=other)
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "ok")
        # self.skill was attached first in setUp, so its template wins.
        self.assertIn("# Title\n## Summary\n## Details", result["content"])
        self.assertIn("note", result)
        self.assertIn("VT Skill 3", result["note"])

    def test_view_nonexistent_template(self):
        result = json.loads(self.tool._run(template_name="No Such"))
        self.assertEqual(result["status"], "error")

    def test_no_skill_attached(self):
        from chat.models import ChatThread

        bare_thread = ChatThread.objects.create(created_by=self.user)
        self.tool.context = _make_context(self.user, thread_id=str(bare_thread.id))
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "error")

    def test_soft_deleted_skill_template_not_resolvable(self):
        """A soft-deleted skill still linked to the thread must not serve its
        resources — its ChatThreadSkill row survives the soft-delete."""
        from agent_skills.services import soft_delete_skill

        soft_delete_skill(self.skill)
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "error")

    def test_oversized_template_truncated(self):
        """Pre-existing oversized rows are truncated before entering the LLM
        context, with an explicit truncation marker."""
        from agent_skills.models import MAX_TEMPLATE_CHARS

        SkillTemplate.objects.create(
            skill=self.skill, name="Huge", content="z" * (MAX_TEMPLATE_CHARS + 1000),
        )
        result = json.loads(self.tool._run(template_name="Huge"))
        self.assertEqual(result["status"], "ok")
        # Content is capped at MAX_TEMPLATE_CHARS (plus the wrap markers).
        self.assertEqual(result["content"].count("z"), MAX_TEMPLATE_CHARS)
        self.assertIs(result["truncated"], True)
        self.assertIn("note", result)

    def test_normal_template_has_no_truncation_marker(self):
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("truncated", result)
        self.assertNotIn("note", result)

    def test_view_image_resource_attaches_inline(self):
        from django.core.files.base import ContentFile

        from agent_skills.models import SkillResource

        res = SkillResource(
            skill=self.skill, name="Diagram", kind="reference",
            file_type="image", original_filename="d.png",
            media_type="image/png", status="ready",
        )
        res.original_file.save("d.png", ContentFile(b"\x89PNG-fake-bytes"), save=False)
        res.save()

        result = json.loads(self.tool._run(template_name="Diagram"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["file_type"], "image")
        self.assertIn("attached below", result["content"])
        self.assertIn('begin resource "Diagram"', result["content"])
        # Queued for the pipeline to drain into an inline image content block.
        self.assertEqual(len(self.tool.context.pending_native_assets), 1)
        self.assertEqual(self.tool.context.pending_native_assets[0]["kind"], "image")

    def test_oversized_asset_not_read_when_budget_exhausted(self):
        """When the run's native-asset budget can't fit the file, the tool bails
        on S3 metadata alone — the bytes are never opened/read into memory."""
        from unittest.mock import patch

        from django.core.files.base import ContentFile

        from agent_skills.models import SkillResource

        res = SkillResource(
            skill=self.skill, name="BigPic", kind="reference",
            file_type="image", original_filename="big.png",
            media_type="image/png", status="ready",
        )
        res.original_file.save("big.png", ContentFile(b"x" * 100), save=False)
        res.save()

        with patch.object(
            RunContext, "native_asset_budget_remaining", return_value=1
        ), patch(
            "django.db.models.fields.files.FieldFile.open"
        ) as m_open:
            result = json.loads(self.tool._run(template_name="BigPic"))

        self.assertEqual(result["status"], "error")
        self.assertIn("budget", result["message"])
        self.assertEqual(self.tool.context.pending_native_assets, [])
        m_open.assert_not_called()  # never pulled the bytes into memory

    def test_quarantined_resource_not_viewable(self):
        from agent_skills.models import SkillResource

        SkillResource.objects.create(
            skill=self.skill, name="Bad", content="secret",
            is_quarantined=True, status="quarantined",
        )
        result = json.loads(self.tool._run(template_name="Bad"))
        self.assertEqual(result["status"], "error")

    def test_view_from_skill_under_edit_via_skill_slug(self):
        """skill_slug reads a resource on a skill that is NOT attached to the
        thread — the authoring case."""
        editing = AgentSkill.objects.create(
            slug="under-edit", name="Under Edit", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=editing, name="Draft Format", content="draft body here",
        )
        # A bare thread with nothing attached — resolution must come from skill_slug.
        from chat.models import ChatThread

        bare = ChatThread.objects.create(created_by=self.user)
        self.tool.context = _make_context(self.user, thread_id=str(bare.id))
        result = json.loads(self.tool._run(
            template_name="Draft Format", skill_slug="under-edit",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertIn("draft body here", result["content"])

    def test_view_via_skill_slug_other_user_denied(self):
        other = User.objects.create_user(email="vt-other@example.com", password="pass")
        theirs = AgentSkill.objects.create(
            slug="theirs", name="Theirs", instructions="Inst.",
            level="user", created_by=other,
        )
        SkillTemplate.objects.create(skill=theirs, name="Secret", content="nope")
        result = json.loads(self.tool._run(
            template_name="Secret", skill_slug="theirs",
        ))
        self.assertEqual(result["status"], "error")


class LoadTemplateToCanvasToolTests(TestCase):
    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="loadtmpl@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="lt-skill", name="LT Skill", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=self.skill, name="Claim Template",
            content="1. A method comprising:\n   a) step one\n   b) step two",
        )
        from chat.models import ChatThread, ChatThreadSkill

        self.thread = ChatThread.objects.create(created_by=self.user)
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        self.tool = LoadTemplateToCanvasTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_loads_template_to_canvas(self):
        result = json.loads(self.tool._run(template_name="Claim Template"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "Claim Template")
        self.assertNotIn("content", result)
        self.assertNotIn("accepted_content", result)

        # Verify canvas was created
        from chat.models import ChatCanvas

        canvas = ChatCanvas.objects.get(thread=self.thread)
        self.assertEqual(canvas.title, "Claim Template")
        self.assertIn("step one", canvas.content)

    def test_nonexistent_template(self):
        result = json.loads(self.tool._run(template_name="No Such"))
        self.assertEqual(result["status"], "error")

    def test_no_skill_attached(self):
        from chat.models import ChatThread

        bare_thread = ChatThread.objects.create(created_by=self.user)
        self.tool.context = _make_context(self.user, thread_id=str(bare_thread.id))
        result = json.loads(self.tool._run(template_name="Claim Template"))
        self.assertEqual(result["status"], "error")

    def test_returns_canvas_id(self):
        result = json.loads(self.tool._run(template_name="Claim Template"))
        self.assertIn("canvas_id", result)

    def test_sets_active_canvas(self):
        json.loads(self.tool._run(template_name="Claim Template"))
        self.thread.refresh_from_db()
        self.assertIsNotNone(self.thread.active_canvas)

    def test_custom_canvas_name(self):
        result = json.loads(self.tool._run(template_name="Claim Template", canvas_name="My Tab"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "My Tab")
        from chat.models import ChatCanvas
        self.assertTrue(ChatCanvas.objects.filter(thread=self.thread, title="My Tab").exists())

    def test_loads_reference_kind_resource(self):
        # Users mislabel kinds, so load must work for a reference resource too —
        # gating is on having text content, not on kind == template.
        from agent_skills.models import SkillResource

        SkillResource.objects.create(
            skill=self.skill, name="Ref Note",
            kind=SkillResource.Kind.REFERENCE,
            content="Reference body to load.",
        )
        result = json.loads(self.tool._run(template_name="Ref Note"))
        self.assertEqual(result["status"], "ok")
        from chat.models import ChatCanvas
        canvas = ChatCanvas.objects.get(thread=self.thread, title="Ref Note")
        self.assertIn("Reference body", canvas.content)

    def test_load_from_skill_under_edit_via_skill_slug(self):
        """skill_slug loads a resource from a skill not attached to the thread."""
        editing = AgentSkill.objects.create(
            slug="lt-under-edit", name="LT Under Edit", instructions="Inst.",
            level="user", created_by=self.user,
        )
        SkillTemplate.objects.create(
            skill=editing, name="Edit Draft", content="editable draft body",
        )
        from chat.models import ChatCanvas, ChatThread

        bare = ChatThread.objects.create(created_by=self.user)
        self.tool.context = _make_context(self.user, thread_id=str(bare.id))
        result = json.loads(self.tool._run(
            template_name="Edit Draft", skill_slug="lt-under-edit",
        ))
        self.assertEqual(result["status"], "ok")
        canvas = ChatCanvas.objects.get(thread=bare, title="Edit Draft")
        self.assertIn("editable draft body", canvas.content)


class LoadResourceToDeckTests(TestCase):
    """skill_resource_load(target="deck"): a slide resource is added to a deck, a
    deck resource becomes a NEW deck, anything else is refused — never overwrites."""

    def setUp(self):
        from chat.models import ChatThread, ChatThreadSkill
        from chat.slides import layouts

        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="loaddeck@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="deck-tmpl", name="Deck Tmpl", instructions="Inst.",
            level="user", created_by=self.user,
        )
        self.slide = layouts.get_layout("bullets")
        # A slide copied out of a real deck carries its old id.
        stale = dict(self.slide, id="s1")
        SkillTemplate.objects.create(skill=self.skill, name="One Slide", content=json.dumps(stale))
        self.deck_json = {
            "version": 1, "size": {"w": 960, "h": 540},
            "slides": [layouts.get_layout("title"), layouts.get_layout("closing")],
        }
        SkillTemplate.objects.create(skill=self.skill, name="Full Deck", content=json.dumps(self.deck_json))
        SkillTemplate.objects.create(skill=self.skill, name="Junk", content='{"foo": 1}')
        SkillTemplate.objects.create(skill=self.skill, name="Not JSON", content="1. A method comprising")
        SkillTemplate.objects.create(skill=self.skill, name="Array", content=json.dumps([self.slide]))

        self.thread = ChatThread.objects.create(created_by=self.user)
        ChatThreadSkill.objects.create(thread=self.thread, skill=self.skill)
        self.tool = LoadTemplateToCanvasTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def _make_deck(self, title="QA Deck", n=2, activate=True):
        from chat.slides import layouts, schema, service

        content = {
            "version": 1, "size": {"w": 960, "h": 540},
            "slides": [layouts.get_layout("bullets") for _ in range(n)],
        }
        schema.mint_ids(content)
        deck, _, _ = service.write_deck(str(self.thread.id), title=title, content=content)
        if activate:
            service.activate_deck(str(self.thread.id), deck)
        return deck

    def _load(self, name, **kw):
        return json.loads(self.tool._run(template_name=name, target="deck", **kw))

    def test_slide_appended_to_active_deck(self):
        deck = self._make_deck()
        before = [s["id"] for s in deck.content["slides"]]
        cp_count = deck.checkpoints.count()

        result = self._load("One Slide")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["loaded_as"], "slide")
        self.assertEqual(result["deck_id"], str(deck.pk))
        self.assertNotIn("canvas_id", result)
        self.assertEqual(result["slide_count"], 3)
        self.assertEqual(result["position"], 2)
        deck.refresh_from_db()
        ids = [s["id"] for s in deck.content["slides"]]
        self.assertEqual(ids[:2], before)  # existing slides untouched
        # The stale "s1" id clashed, so the new slide got a fresh one.
        self.assertNotIn(ids[2], before)
        self.assertEqual(result["slide_id"], ids[2])
        self.assertEqual(result["changed_slide_ids"], [ids[2]])
        self.assertEqual(deck.checkpoints.count(), cp_count + 1)

        from chat.models import ChatCanvas
        self.assertFalse(ChatCanvas.objects.filter(thread=self.thread).exists())

    def test_slide_to_named_deck_at_position(self):
        other = self._make_deck("Other", activate=False)
        self._make_deck("Active")

        result = self._load("One Slide", deck_name="Other", position=0)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deck_id"], str(other.pk))
        other.refresh_from_db()
        self.assertEqual(other.content["slides"][0]["id"], result["slide_id"])
        self.assertTrue(other.is_active)

    def test_slide_unknown_deck_name(self):
        self._make_deck()
        result = self._load("One Slide", deck_name="Nope")
        self.assertEqual(result["status"], "error")
        self.assertIn("available_decks", result)

    def test_slide_without_deck_creates_one(self):
        from chat.models import SlideSet

        result = self._load("One Slide")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["loaded_as"], "slide")
        deck = SlideSet.objects.get(thread=self.thread)
        self.assertEqual(deck.title, "One Slide")
        self.assertTrue(deck.is_active)
        self.assertEqual(len(deck.content["slides"]), 1)

    def test_slide_past_slide_cap_rejected(self):
        from chat.slides import schema

        deck = self._make_deck(n=schema.MAX_SLIDES_PER_DECK)
        result = self._load("One Slide")
        self.assertEqual(result["status"], "error")
        deck.refresh_from_db()
        self.assertEqual(len(deck.content["slides"]), schema.MAX_SLIDES_PER_DECK)

    def test_deck_resource_creates_new_deck_never_overwrites(self):
        from chat.models import SlideSet

        existing = self._make_deck("Full Deck")
        existing_content = existing.content

        first = self._load("Full Deck")
        second = self._load("Full Deck")

        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["loaded_as"], "deck")
        self.assertEqual(first["title"], "Full Deck (2)")
        self.assertEqual(first["slide_count"], 2)
        self.assertIn("deck_json", first)
        self.assertEqual(second["title"], "Full Deck (3)")
        existing.refresh_from_db()
        self.assertEqual(existing.content, existing_content)
        new = SlideSet.objects.get(pk=second["deck_id"])
        self.assertTrue(new.is_active)
        self.assertTrue(all(s.get("id") for s in new.content["slides"]))

    def test_deck_resource_uses_canvas_name_as_title(self):
        result = self._load("Full Deck", canvas_name="Pitch")
        self.assertEqual(result["title"], "Pitch")

    def test_deck_resource_keeps_own_theme(self):
        from chat.models import SlideSet

        themed = dict(self.deck_json, theme={"colors": {"accent1": "#B87333"}})
        SkillTemplate.objects.create(skill=self.skill, name="Themed", content=json.dumps(themed))
        result = self._load("Themed")
        self.assertEqual(result["status"], "ok")
        deck = SlideSet.objects.get(pk=result["deck_id"])
        self.assertEqual(deck.content["theme"]["colors"]["accent1"], "#B87333")

    def test_deck_resource_at_deck_limit(self):
        from chat.slides import schema

        for i in range(schema.MAX_SLIDE_SETS_PER_THREAD):
            self._make_deck(f"D{i}")
        result = self._load("Full Deck")
        self.assertEqual(result["status"], "error")
        self.assertIn("Maximum", result["message"])

    def test_unverifiable_resources_refused(self):
        from chat.models import SlideSet

        deck = self._make_deck()
        content = deck.content
        for name in ("Junk", "Not JSON", "Array"):
            with self.subTest(name=name):
                result = self._load(name)
                self.assertEqual(result["status"], "error")
                self.assertIn("couldn't be verified as a valid slide or deck", result["message"])
                self.assertIn("slide_canvas_edit", result["message"])
                self.assertTrue(result["issues"])
        deck.refresh_from_db()
        self.assertEqual(deck.content, content)
        self.assertEqual(SlideSet.objects.filter(thread=self.thread).count(), 1)

    def test_default_target_still_loads_canvas(self):
        from chat.models import ChatCanvas, SlideSet

        result = json.loads(self.tool._run(template_name="One Slide"))
        self.assertEqual(result["status"], "ok")
        self.assertIn("canvas_id", result)
        self.assertTrue(ChatCanvas.objects.filter(thread=self.thread, title="One Slide").exists())
        self.assertFalse(SlideSet.objects.filter(thread=self.thread).exists())

    def test_end_labels(self):
        self.assertEqual(self.tool.end_label_for_result({"status": "ok", "loaded_as": "slide"}),
                         "Added a slide from template")
        self.assertEqual(self.tool.end_label_for_result({"status": "ok", "loaded_as": "deck"}),
                         "Created a deck from template")
        self.assertIsNone(self.tool.end_label_for_result({"status": "ok", "canvas_id": "x"}))
        self.assertEqual(self.tool.end_label_for_result({"status": "error"}), "Couldn't load template")


class _ResourceToolTestBase(TestCase):
    """Shared fixture: a user-owned skill + an owned thread (so
    resolve_skill_for_thread_edit resolves the skill by slug for writes)."""

    def setUp(self):
        AgentSkill.objects.all().delete()
        self.user = User.objects.create_user(email="res@example.com", password="pass")
        self.skill = AgentSkill.objects.create(
            slug="res-skill", name="Res Skill", instructions="Inst.",
            level="user", created_by=self.user,
        )
        from chat.models import ChatThread

        self.thread = ChatThread.objects.create(created_by=self.user)


class SkillResourceListToolTests(_ResourceToolTestBase):
    def setUp(self):
        super().setUp()
        self.tool = SkillResourceListTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_lists_resources_with_metadata(self):
        from agent_skills.models import SkillResource

        SkillResource.objects.create(
            skill=self.skill, name="Fmt", kind=SkillResource.Kind.TEMPLATE,
            content="body", status="ready", token_count=3,
        )
        SkillResource.objects.create(
            skill=self.skill, name="Bad", kind=SkillResource.Kind.REFERENCE,
            content="x", is_quarantined=True, status="quarantined",
            quarantine_reason="nope",
        )
        result = json.loads(self.tool._run(skill_slug="res-skill"))
        self.assertEqual(result["status"], "ok")
        by_name = {r["name"]: r for r in result["resources"]}
        self.assertEqual(by_name["Fmt"]["kind"], "template")
        # Quarantined rows ARE listed so the author can see + remove them.
        self.assertTrue(by_name["Bad"]["is_quarantined"])
        self.assertEqual(by_name["Bad"]["quarantine_reason"], "nope")

    def test_nonexistent_skill(self):
        result = json.loads(self.tool._run(skill_slug="no-such"))
        self.assertEqual(result["status"], "error")


class SkillResourceSaveToolTests(_ResourceToolTestBase):
    def setUp(self):
        super().setUp()
        from django.utils import timezone

        from chat.models import ChatCanvas

        self.canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Draft", content="Resource body from canvas.",
            is_active=True, last_activated_at=timezone.now(),
        )
        self.thread.active_canvas = self.canvas
        self.thread.save(update_fields=["active_canvas"])
        self.tool = SkillResourceSaveTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_creates_text_resource_with_kind(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Report Fmt", kind="template",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["created"])
        from agent_skills.models import SkillResource

        res = SkillResource.objects.get(skill=self.skill, name="Report Fmt")
        self.assertEqual(res.kind, "template")
        self.assertEqual(res.content, "Resource body from canvas.")

    def test_invalid_kind_defaults_to_reference(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Ref A", kind="bogus",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["kind"], "reference")

    def test_updates_existing_text_resource(self):
        from agent_skills.models import SkillResource

        SkillResource.objects.create(
            skill=self.skill, name="Report Fmt", kind="reference", content="old",
        )
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Report Fmt", kind="template",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["created"])
        res = SkillResource.objects.get(skill=self.skill, name="Report Fmt")
        self.assertEqual(res.content, "Resource body from canvas.")
        self.assertEqual(res.kind, "template")

    def test_cannot_overwrite_file_resource(self):
        from django.core.files.base import ContentFile

        from agent_skills.models import SkillResource

        res = SkillResource(
            skill=self.skill, name="Doc", kind="reference",
            file_type="pdf", original_filename="d.pdf", status="ready",
        )
        res.original_file.save("d.pdf", ContentFile(b"%PDF-fake"), save=False)
        res.save()
        result = json.loads(self.tool._run(skill_slug="res-skill", name="Doc"))
        self.assertEqual(result["status"], "error")
        self.assertIn("file", result["message"])

    def test_count_cap_enforced(self):
        from agent_skills.models import SkillResource
        from agent_skills.resources import RESOURCE_COUNT_CAP

        for i in range(RESOURCE_COUNT_CAP):
            SkillResource.objects.create(
                skill=self.skill, name=f"r{i}", kind="reference", content="x",
            )
        result = json.loads(self.tool._run(skill_slug="res-skill", name="one-too-many"))
        self.assertEqual(result["status"], "error")


class SkillResourceUpdateToolTests(_ResourceToolTestBase):
    def setUp(self):
        super().setUp()
        from agent_skills.models import SkillResource

        self.res = SkillResource.objects.create(
            skill=self.skill, name="Old Name", kind="reference", content="body",
        )
        self.tool = SkillResourceUpdateTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_rename(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Old Name", new_name="New Name",
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "New Name")
        self.res.refresh_from_db()
        self.assertEqual(self.res.name, "New Name")

    def test_retype_kind(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Old Name", kind="template",
        ))
        self.assertEqual(result["status"], "ok")
        self.res.refresh_from_db()
        self.assertEqual(self.res.kind, "template")

    def test_nothing_to_update_rejected(self):
        result = json.loads(self.tool._run(skill_slug="res-skill", name="Old Name"))
        self.assertEqual(result["status"], "error")

    def test_rename_collision_rejected(self):
        from agent_skills.models import SkillResource

        SkillResource.objects.create(
            skill=self.skill, name="Taken", kind="reference", content="x",
        )
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Old Name", new_name="Taken",
        ))
        self.assertEqual(result["status"], "error")

    def test_missing_resource_rejected(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", name="Ghost", new_name="X",
        ))
        self.assertEqual(result["status"], "error")


class SkillResourceDeleteToolTests(_ResourceToolTestBase):
    def setUp(self):
        super().setUp()
        self.tool = SkillResourceDeleteTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_deletes_by_name(self):
        from agent_skills.models import SkillResource

        SkillResource.objects.create(skill=self.skill, name="a", content="A")
        SkillResource.objects.create(skill=self.skill, name="b", content="B")
        SkillResource.objects.create(skill=self.skill, name="keep", content="K")
        result = json.loads(self.tool._run(skill_slug="res-skill", names=["a", "b"]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deleted_count"], 2)
        self.assertFalse(SkillResource.objects.filter(skill=self.skill, name="a").exists())
        self.assertTrue(SkillResource.objects.filter(skill=self.skill, name="keep").exists())

    def test_empty_names_rejected(self):
        result = json.loads(self.tool._run(skill_slug="res-skill", names=[]))
        self.assertEqual(result["status"], "error")

    def test_nonexistent_names_delete_zero(self):
        result = json.loads(self.tool._run(skill_slug="res-skill", names=["ghost"]))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["deleted_count"], 0)


class SkillResourceAttachToolTests(_ResourceToolTestBase):
    def setUp(self):
        super().setUp()
        self.tool = SkillResourceAttachTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def _thread_image_asset(self, owner=None, data=b"\x89PNG-fake"):
        from django.core.files.base import ContentFile

        from chat.models import Asset, ChatThread

        thr = ChatThread.objects.create(created_by=owner or self.user)
        asset = Asset(thread=thr, kind=Asset.KIND_IMAGE, content_type="image/png")
        asset.blob.save("a.png", ContentFile(data), save=True)
        return asset

    @patch("agent_skills.tasks.process_skill_resource_upload_task.delay")
    def test_attach_thread_image_returns_processing(self, m_delay):
        asset = self._thread_image_asset()
        result = json.loads(self.tool._run(
            skill_slug="res-skill", source=f"[[image:{asset.id}|pic]]",
            kind="reference", name="Diagram",
        ))
        self.assertEqual(result["status"], "processing")
        from agent_skills.models import SkillResource

        res = SkillResource.objects.get(skill=self.skill, name="Diagram")
        self.assertEqual(res.file_type, "image")
        self.assertEqual(res.status, "processing")
        m_delay.assert_called_once()

    @patch("agent_skills.tasks.process_skill_resource_upload_task.delay")
    def test_attach_accepts_bare_uuid(self, m_delay):
        asset = self._thread_image_asset()
        result = json.loads(self.tool._run(
            skill_slug="res-skill", source=str(asset.id),
        ))
        self.assertEqual(result["status"], "processing")
        m_delay.assert_called_once()

    @patch("agent_skills.tasks.process_skill_resource_upload_task.delay")
    def test_attach_other_users_asset_denied(self, m_delay):
        other = User.objects.create_user(email="att-other@example.com", password="pass")
        asset = self._thread_image_asset(owner=other)
        result = json.loads(self.tool._run(
            skill_slug="res-skill", source=str(asset.id),
        ))
        self.assertEqual(result["status"], "error")
        m_delay.assert_not_called()

    def test_attach_bad_source_rejected(self):
        result = json.loads(self.tool._run(skill_slug="res-skill", source="not-a-uuid"))
        self.assertEqual(result["status"], "error")

    def test_attach_missing_asset_rejected(self):
        result = json.loads(self.tool._run(
            skill_slug="res-skill", source=str(uuid.uuid4()),
        ))
        self.assertEqual(result["status"], "error")
