"""Tests for agent_skills.tools — skill management tools."""

import json
import uuid

from django.contrib.auth import get_user_model
from django.test import TestCase

from accounts.models import Membership, Organization
from agent_skills.models import AgentSkill, SkillTemplate
from agent_skills.tools import (
    CreateSkillTool,
    DeleteSkillTool,
    EditSkillTool,
    InspectToolTool,
    ListAllToolsTool,
    LoadTemplateToCanvasTool,
    SaveCanvasToSkillFieldTool,
    ShowSkillFieldInCanvasTool,
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

    def test_create_skill(self):
        result = json.loads(self.tool._run(name="My Skill"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "My Skill")
        self.assertEqual(result["slug"], "my-skill")
        self.assertTrue(AgentSkill.objects.filter(slug="my-skill", created_by=self.user).exists())


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
            updates={"tool_names": ["view_template", "write_canvas", "search_documents"]},
        ))
        self.assertEqual(result["status"], "ok")
        # write_canvas and search_documents are chat-section tools — silently removed
        self.assertEqual(result["tool_names"], ["view_template"])
        self.skill.refresh_from_db()
        self.assertEqual(self.skill.tool_names, ["view_template"])

    def test_tool_names_all_standard_results_in_empty_list(self):
        """If only standard tools are passed, tool_names becomes empty."""
        result = json.loads(self.tool._run(
            skill_slug="editable",
            updates={"tool_names": ["write_canvas", "edit_canvas"]},
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool_names"], [])

    def test_delete_templates(self):
        SkillTemplate.objects.create(skill=self.skill, name="tmpl-a", content="A")
        SkillTemplate.objects.create(skill=self.skill, name="tmpl-b", content="B")
        SkillTemplate.objects.create(skill=self.skill, name="tmpl-keep", content="Keep")
        result = json.loads(self.tool._run(
            skill_slug="editable", delete_templates=["tmpl-a", "tmpl-b"],
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["templates_deleted"], 2)
        self.assertFalse(SkillTemplate.objects.filter(skill=self.skill, name="tmpl-a").exists())
        self.assertFalse(SkillTemplate.objects.filter(skill=self.skill, name="tmpl-b").exists())
        self.assertTrue(SkillTemplate.objects.filter(skill=self.skill, name="tmpl-keep").exists())

    def test_delete_templates_nonexistent_ignored(self):
        result = json.loads(self.tool._run(
            skill_slug="editable", delete_templates=["no-such-template"],
        ))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["templates_deleted"], 0)

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
        self.assertFalse(AgentSkill.objects.filter(slug="doomed").exists())

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
        self.canvas = ChatCanvas.objects.create(
            thread=self.thread, title="Draft", content="Canvas content here.",
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

    def test_save_to_template(self):
        result = json.loads(self.tool._run(skill_slug="canvas-skill", field_name="Patent Claim"))
        self.assertEqual(result["status"], "ok")
        tmpl = SkillTemplate.objects.get(skill=self.skill, name="Patent Claim")
        self.assertEqual(tmpl.content, "Canvas content here.")

    def test_not_editable_denied(self):
        AgentSkill.objects.create(
            slug="sys", name="System", instructions="Inst.", level="system",
        )
        result = json.loads(self.tool._run(skill_slug="sys", field_name="instructions"))
        self.assertEqual(result["status"], "error")

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
        self.assertEqual(result["content"], "Skill instructions here.")
        self.assertIn("title", result)
        self.assertIn("accepted_content", result)

    def test_show_template(self):
        result = json.loads(self.tool._run(skill_slug="show-skill", field_name="My Template"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["content"], "Template content.")

    def test_nonexistent_template(self):
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


class ListAllToolsToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="listtools@example.com", password="pass")
        self.tool = ListAllToolsTool()
        self.tool.context = _make_context(self.user)

    def test_returns_both_standard_and_skill_tools(self):
        result = json.loads(self.tool._run())
        self.assertEqual(result["status"], "ok")
        # Both groups present
        self.assertIn("standard_tools", result)
        self.assertIn("skill_tools", result)
        skill_names = [t["name"] for t in result["skill_tools"]]
        standard_names = [t["name"] for t in result["standard_tools"]]
        # Skill-section tools in skill_tools
        self.assertIn("create_skill", skill_names)
        self.assertIn("inspect_tool", skill_names)
        # Chat-section tools in standard_tools
        self.assertIn("write_canvas", standard_names)
        self.assertIn("edit_canvas", standard_names)
        # No overlap
        self.assertFalse(set(skill_names) & set(standard_names))

    def test_each_entry_has_name_and_description(self):
        result = json.loads(self.tool._run())
        for tool_entry in result["skill_tools"] + result["standard_tools"]:
            self.assertIn("name", tool_entry)
            self.assertIn("description", tool_entry)


class InspectToolToolTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="inspect@example.com", password="pass")
        self.tool = InspectToolTool()
        self.tool.context = _make_context(self.user)

    def test_inspect_existing_tool(self):
        result = json.loads(self.tool._run(tool_name="create_skill"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["name"], "create_skill")
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
        from chat.models import ChatThread

        self.thread = ChatThread.objects.create(created_by=self.user, skill=self.skill)
        self.tool = ViewTemplateTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_view_existing_template(self):
        result = json.loads(self.tool._run(template_name="Report Format"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["template_name"], "Report Format")
        self.assertEqual(result["content"], "# Title\n## Summary\n## Details")

    def test_view_nonexistent_template(self):
        result = json.loads(self.tool._run(template_name="No Such"))
        self.assertEqual(result["status"], "error")

    def test_no_skill_attached(self):
        from chat.models import ChatThread

        bare_thread = ChatThread.objects.create(created_by=self.user)
        self.tool.context = _make_context(self.user, thread_id=str(bare_thread.id))
        result = json.loads(self.tool._run(template_name="Report Format"))
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
        from chat.models import ChatThread

        self.thread = ChatThread.objects.create(created_by=self.user, skill=self.skill)
        self.tool = LoadTemplateToCanvasTool()
        self.tool.context = _make_context(self.user, thread_id=str(self.thread.id))

    def test_loads_template_to_canvas(self):
        result = json.loads(self.tool._run(template_name="Claim Template"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["title"], "Claim Template")
        self.assertIn("step one", result["content"])
        self.assertIn("accepted_content", result)

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
