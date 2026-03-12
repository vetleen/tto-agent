"""Skill management tools for agent use during chat."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, get_tool_registry


# -- Input schemas --


class CreateSkillInput(BaseModel):
    name: str = Field(description="Name for the new skill.")


class SaveCanvasToSkillFieldInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill to save to.")
    field_name: str = Field(
        description=(
            "Field to save canvas content to: 'instructions', 'description', "
            "or a template name."
        )
    )


class ShowSkillFieldInCanvasInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill to read from.")
    field_name: str = Field(
        description=(
            "Field to show: 'instructions', 'description', or a template name."
        )
    )


class TextEdit(BaseModel):
    field: str = Field(description="Field name to edit (e.g. 'description').")
    old_text: str = Field(description="Exact text to find.")
    new_text: str = Field(description="Replacement text.")


class EditSkillInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill to edit.")
    updates: dict = Field(
        default_factory=dict,
        description="Optional keys: name, new_slug, tool_names, is_active.",
    )
    text_edits: list[TextEdit] = Field(
        default_factory=list,
        description="Find-replace edits for text fields like description.",
    )


class DeleteSkillInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill to delete.")


class AddSkillTemplateInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill.")
    template_name: str = Field(description="Name for the new template.")
    content: str = Field(default="", description="Initial template content.")


class TemplateEdit(BaseModel):
    old_text: str = Field(description="Exact text to find.")
    new_text: str = Field(description="Replacement text.")


class EditSkillTemplateInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill.")
    template_name: str = Field(description="Name of the template to edit.")
    new_name: Optional[str] = Field(default=None, description="Rename template.")
    edits: list[TemplateEdit] = Field(
        default_factory=list,
        description="Find-replace edits on template content.",
    )


class DeleteSkillTemplateInput(BaseModel):
    skill_slug: str = Field(description="Slug of the skill.")
    template_name: str = Field(description="Name of the template to delete.")


class ViewTemplateInput(BaseModel):
    template_name: str = Field(description="Name of the template to view.")


class LoadTemplateToCanvasInput(BaseModel):
    template_name: str = Field(description="Name of the template to load into the canvas.")


class InspectToolInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to inspect.")


# -- Tools --


class CreateSkillTool(ContextAwareTool):
    """Create a new user-level skill."""

    name: str = "create_skill"
    description: str = (
        "Create a new user-level skill. Returns the slug and ID of the created skill."
    )
    args_schema: type[BaseModel] = CreateSkillInput
    section: str = "skills"

    def _run(self, name: str) -> str:
        from agent_skills.services import create_user_skill

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = create_user_skill(user, name)
        return json.dumps({
            "status": "ok",
            "slug": skill.slug,
            "name": skill.name,
            "id": str(skill.id),
        })


class SaveCanvasToSkillFieldTool(ContextAwareTool):
    """Save the current canvas content to a skill field or template."""

    name: str = "save_canvas_to_skill_field"
    description: str = (
        "Save the current canvas content into a skill's instructions, description, "
        "or a named template. The canvas content is saved verbatim."
    )
    args_schema: type[BaseModel] = SaveCanvasToSkillFieldInput
    section: str = "skills"

    def _run(self, skill_slug: str, field_name: str) -> str:
        from agent_skills.models import SkillTemplate
        from agent_skills.services import get_editable_skill_for_user
        from chat.models import ChatCanvas

        user_id = self.context.user_id if self.context else None
        thread_id = self.context.conversation_id if self.context else None
        if not user_id or not thread_id:
            return json.dumps({"status": "error", "message": "No context available."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        try:
            canvas = ChatCanvas.objects.get(thread_id=thread_id)
        except ChatCanvas.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": "No canvas exists for this thread.",
            })

        content = canvas.content

        if field_name in ("instructions", "description"):
            setattr(skill, field_name, content)
            skill.save(update_fields=[field_name, "updated_at"])
        else:
            SkillTemplate.objects.update_or_create(
                skill=skill, name=field_name,
                defaults={"content": content},
            )

        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "field": field_name,
            "chars_saved": len(content),
        })


class ShowSkillFieldInCanvasTool(ContextAwareTool):
    """Load a skill field or template into the canvas for viewing/editing."""

    name: str = "show_skill_field_in_canvas"
    description: str = (
        "Load a skill's instructions, description, or a named template into "
        "the canvas. This allows the user to view and edit the content."
    )
    args_schema: type[BaseModel] = ShowSkillFieldInCanvasInput
    section: str = "skills"

    def _run(self, skill_slug: str, field_name: str) -> str:
        from agent_skills.services import get_available_skills
        from chat.models import ChatCanvas
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint

        user_id = self.context.user_id if self.context else None
        thread_id = self.context.conversation_id if self.context else None
        if not user_id or not thread_id:
            return json.dumps({"status": "error", "message": "No context available."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        # Read access: any accessible skill via shadowing
        skills = get_available_skills(user)
        skill = None
        for s in skills:
            if s.slug == skill_slug:
                skill = s
                break
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found.",
            })

        if field_name in ("instructions", "description"):
            content = getattr(skill, field_name) or ""
        else:
            try:
                tmpl = skill.templates.get(name=field_name)
                content = tmpl.content
            except Exception:
                return json.dumps({
                    "status": "error",
                    "message": f"Template '{field_name}' not found on skill '{skill_slug}'.",
                })

        title = f"{skill.name} \u2014 {field_name}"
        content = content[:CANVAS_MAX_CHARS]

        canvas, created = ChatCanvas.objects.update_or_create(
            thread_id=thread_id,
            defaults={"title": title, "content": content},
        )
        cp = create_canvas_checkpoint(canvas, source="import", description=f"Loaded {field_name}")
        if created:
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])

        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else ""

        return json.dumps({
            "status": "ok",
            "title": title,
            "content": content,
            "accepted_content": accepted_content,
        })


class EditSkillTool(ContextAwareTool):
    """Edit a skill's metadata or text fields."""

    name: str = "edit_skill"
    description: str = (
        "Edit a skill's name, slug, tool_names, is_active, or apply "
        "find-replace edits to text fields like description."
    )
    args_schema: type[BaseModel] = EditSkillInput
    section: str = "skills"

    def _run(
        self,
        skill_slug: str,
        updates: dict | None = None,
        text_edits: list[dict] | list[TextEdit] | None = None,
    ) -> str:
        from agent_skills.services import get_editable_skill_for_user

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        updates = updates or {}
        update_fields = ["updated_at"]

        if "name" in updates:
            skill.name = updates["name"]
            update_fields.append("name")
        if "new_slug" in updates:
            skill.slug = updates["new_slug"]
            update_fields.append("slug")
        if "tool_names" in updates:
            # Silently filter out standard (chat-section) tools — they're
            # always available and don't need to be attached to a skill.
            # Unknown tool names are kept (they may belong to another app).
            registry = get_tool_registry()
            filtered = []
            for t in updates["tool_names"]:
                tool_obj = registry.get_tool(t)
                if tool_obj is None or getattr(tool_obj, "section", "chat") != "chat":
                    filtered.append(t)
            skill.tool_names = filtered
            update_fields.append("tool_names")
        if "is_active" in updates:
            skill.is_active = bool(updates["is_active"])
            update_fields.append("is_active")

        # Apply text edits (find-replace on text fields)
        failed = []
        applied = 0
        for edit in text_edits or []:
            if isinstance(edit, dict):
                field = edit.get("field", "")
                old_text = edit.get("old_text", "")
                new_text = edit.get("new_text", "")
            else:
                field = edit.field
                old_text = edit.old_text
                new_text = edit.new_text

            if field not in ("description", "instructions"):
                failed.append({"field": field, "error": "Invalid field for text edit."})
                continue

            current = getattr(skill, field) or ""
            count = current.count(old_text)
            if count == 1:
                setattr(skill, field, current.replace(old_text, new_text, 1))
                if field not in update_fields:
                    update_fields.append(field)
                applied += 1
            elif count > 1:
                failed.append({
                    "field": field,
                    "old_text": old_text[:80],
                    "error": f"Found {count} matches — include more text to make it unique.",
                })
            else:
                failed.append({
                    "field": field,
                    "old_text": old_text[:80],
                    "error": "Text not found.",
                })

        skill.save(update_fields=update_fields)

        return json.dumps({
            "status": "ok",
            "slug": skill.slug,
            "name": skill.name,
            "id": str(skill.id),
            "is_active": skill.is_active,
            "tool_names": skill.tool_names,
            "edits_applied": applied,
            "edits_failed": failed,
        })


class DeleteSkillTool(ContextAwareTool):
    """Delete a user or org skill."""

    name: str = "delete_skill"
    description: str = "Delete a skill that the user owns. System skills cannot be deleted."
    args_schema: type[BaseModel] = DeleteSkillInput
    section: str = "skills"

    def _run(self, skill_slug: str) -> str:
        from agent_skills.services import get_editable_skill_for_user

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        skill.delete()
        return json.dumps({"status": "ok", "deleted": skill_slug})


class AddSkillTemplateTool(ContextAwareTool):
    """Add a template to a skill."""

    name: str = "add_skill_template"
    description: str = "Add a new named template to a skill."
    args_schema: type[BaseModel] = AddSkillTemplateInput
    section: str = "skills"

    def _run(self, skill_slug: str, template_name: str, content: str = "") -> str:
        from agent_skills.models import SkillTemplate
        from agent_skills.services import get_editable_skill_for_user

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        from django.db import IntegrityError

        try:
            tmpl = SkillTemplate.objects.create(
                skill=skill, name=template_name, content=content,
            )
        except IntegrityError:
            return json.dumps({
                "status": "error",
                "message": f"Template '{template_name}' already exists on skill '{skill_slug}'.",
            })
        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "template_name": tmpl.name,
            "id": str(tmpl.id),
            "chars": len(tmpl.content),
        })


class EditSkillTemplateTool(ContextAwareTool):
    """Edit a skill template's name or content via find-replace."""

    name: str = "edit_skill_template"
    description: str = (
        "Edit a skill template. Optionally rename it and/or apply "
        "find-replace edits to its content."
    )
    args_schema: type[BaseModel] = EditSkillTemplateInput
    section: str = "skills"

    def _run(
        self,
        skill_slug: str,
        template_name: str,
        new_name: str | None = None,
        edits: list[dict] | list[TemplateEdit] | None = None,
    ) -> str:
        from agent_skills.models import SkillTemplate
        from agent_skills.services import get_editable_skill_for_user

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        try:
            tmpl = SkillTemplate.objects.get(skill=skill, name=template_name)
        except SkillTemplate.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": f"Template '{template_name}' not found.",
            })

        update_fields = ["updated_at"]

        if new_name:
            tmpl.name = new_name
            update_fields.append("name")

        content = tmpl.content
        applied = 0
        failed = []
        for edit in edits or []:
            if isinstance(edit, dict):
                old_text = edit.get("old_text", "")
                new_text = edit.get("new_text", "")
            else:
                old_text = edit.old_text
                new_text = edit.new_text

            count = content.count(old_text)
            if count == 1:
                content = content.replace(old_text, new_text, 1)
                applied += 1
            elif count > 1:
                failed.append({
                    "old_text": old_text[:80],
                    "error": f"Found {count} matches.",
                })
            else:
                failed.append({
                    "old_text": old_text[:80],
                    "error": "Text not found.",
                })

        if applied > 0:
            tmpl.content = content
            update_fields.append("content")

        tmpl.save(update_fields=update_fields)

        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "template_name": tmpl.name,
            "id": str(tmpl.id),
            "edits_applied": applied,
            "edits_failed": failed,
        })


class DeleteSkillTemplateTool(ContextAwareTool):
    """Delete a template from a skill."""

    name: str = "delete_skill_template"
    description: str = "Delete a named template from a skill."
    args_schema: type[BaseModel] = DeleteSkillTemplateInput
    section: str = "skills"

    def _run(self, skill_slug: str, template_name: str) -> str:
        from agent_skills.models import SkillTemplate
        from agent_skills.services import get_editable_skill_for_user

        user_id = self.context.user_id if self.context else None
        if not user_id:
            return json.dumps({"status": "error", "message": "No user context."})

        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        skill = get_editable_skill_for_user(user, skill_slug)
        if not skill:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{skill_slug}' not found or not editable.",
            })

        try:
            tmpl = SkillTemplate.objects.get(skill=skill, name=template_name)
        except SkillTemplate.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": f"Template '{template_name}' not found.",
            })

        tmpl.delete()
        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "deleted_template": template_name,
        })


class ViewTemplateTool(ContextAwareTool):
    """View the content of a template from the active skill."""

    name: str = "view_template"
    description: str = (
        "View the full content of a named template from the current skill. "
        "Returns the template text so you can reference it when generating output."
    )
    args_schema: type[BaseModel] = ViewTemplateInput
    section: str = "skills"

    def _run(self, template_name: str) -> str:
        from agent_skills.models import SkillTemplate
        from chat.models import ChatThread

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        try:
            thread = ChatThread.objects.select_related("skill").get(pk=thread_id)
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        if not thread.skill_id:
            return json.dumps({"status": "error", "message": "No skill attached to this thread."})

        try:
            tmpl = SkillTemplate.objects.get(skill_id=thread.skill_id, name=template_name)
        except SkillTemplate.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": f"Template '{template_name}' not found on the active skill.",
            })

        return json.dumps({
            "status": "ok",
            "template_name": tmpl.name,
            "content": tmpl.content,
        })


class LoadTemplateToCanvasTool(ContextAwareTool):
    """Load a template from the active skill into the canvas."""

    name: str = "load_template_to_canvas"
    description: str = (
        "Load a named template from the current skill into the canvas. "
        "Use this to give the user a starting point they can edit. "
        "This replaces the current canvas content."
    )
    args_schema: type[BaseModel] = LoadTemplateToCanvasInput
    section: str = "skills"

    def _run(self, template_name: str) -> str:
        from agent_skills.models import SkillTemplate
        from chat.models import ChatCanvas, ChatThread
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        try:
            thread = ChatThread.objects.select_related("skill").get(pk=thread_id)
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        if not thread.skill_id:
            return json.dumps({"status": "error", "message": "No skill attached to this thread."})

        try:
            tmpl = SkillTemplate.objects.get(skill_id=thread.skill_id, name=template_name)
        except SkillTemplate.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": f"Template '{template_name}' not found on the active skill.",
            })

        content = tmpl.content[:CANVAS_MAX_CHARS]
        title = tmpl.name

        canvas, created = ChatCanvas.objects.update_or_create(
            thread_id=thread_id,
            defaults={"title": title, "content": content},
        )
        cp = create_canvas_checkpoint(canvas, source="import", description=f"Loaded template: {template_name}")
        if created:
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])

        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else ""

        return json.dumps({
            "status": "ok",
            "title": title,
            "content": content,
            "accepted_content": accepted_content,
        })


class ListAllToolsTool(ContextAwareTool):
    """List all tools grouped by availability."""

    name: str = "list_all_tools"
    description: str = (
        "List all available tools, grouped into two categories: "
        "standard tools (always available — no need to attach) and "
        "skill-specific tools (must be explicitly attached via tool_names). "
        "Use this to discover which tools exist and decide which ones "
        "a skill needs."
    )
    args_schema: type[BaseModel] = BaseModel
    section: str = "skills"

    def _run(self) -> str:
        registry = get_tool_registry()
        all_tools = registry.list_tools()
        standard_tools = []
        skill_tools = []
        for name, tool in sorted(all_tools.items()):
            first_sentence = (tool.description or "").split(". ")[0]
            entry = {"name": name, "description": first_sentence}
            if getattr(tool, "section", "chat") == "skills":
                skill_tools.append(entry)
            else:
                standard_tools.append(entry)
        return json.dumps({
            "status": "ok",
            "standard_tools": standard_tools,
            "standard_tools_note": "Always available. Do not need to be attached to a skill.",
            "skill_tools": skill_tools,
            "skill_tools_note": "Only available when explicitly listed in a skill's tool_names.",
        })


class InspectToolTool(ContextAwareTool):
    """Inspect a tool to see its description and determine if it's appropriate for a skill."""

    name: str = "inspect_tool"
    description: str = (
        "Get the description of a specific tool by name. Use this to "
        "understand what a tool does before adding it to a skill's tool_names."
    )
    args_schema: type[BaseModel] = InspectToolInput
    section: str = "skills"

    def _run(self, tool_name: str) -> str:
        registry = get_tool_registry()
        tool = registry.get_tool(tool_name)
        if not tool:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found.",
            })
        return json.dumps({
            "status": "ok",
            "name": tool.name,
            "description": tool.description,
            "section": getattr(tool, "section", "chat"),
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(CreateSkillTool())
_registry.register_tool(SaveCanvasToSkillFieldTool())
_registry.register_tool(ShowSkillFieldInCanvasTool())
_registry.register_tool(EditSkillTool())
_registry.register_tool(DeleteSkillTool())
_registry.register_tool(AddSkillTemplateTool())
_registry.register_tool(EditSkillTemplateTool())
_registry.register_tool(DeleteSkillTemplateTool())
_registry.register_tool(ViewTemplateTool())
_registry.register_tool(LoadTemplateToCanvasTool())
_registry.register_tool(ListAllToolsTool())
_registry.register_tool(InspectToolTool())
