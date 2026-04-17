"""Skill management tools for agent use during chat."""

from __future__ import annotations

import json
from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry


def resolve_skill_for_thread_edit(user, thread_id, slug: str):
    """Resolve which skill an in-thread edit tool should mutate.

    The "edit skill in chat" flow stores ``source_skill_id`` on
    ``ChatThread.metadata``. When present, edits target *that* skill (by id,
    not slug, so the LLM doesn't have to track slug renames). If the user
    cannot edit the source skill it is forked to a user-tier copy on first
    write, and the fork's id is written back to ``thread.metadata`` so
    subsequent edits in the same thread land on the fork directly.

    When ``source_skill_id`` is not set we fall back to the legacy
    slug-based lookup, preserving behavior for any other thread that has
    Skill Creator attached without going through the edit-in-chat flow.

    Returns ``(skill, error_message)``: exactly one of the two is non-None.
    """
    from agent_skills.models import AgentSkill
    from agent_skills.services import (
        can_edit_skill,
        fork_skill,
        get_editable_skill_for_user,
    )
    from chat.models import ChatThread

    thread = ChatThread.objects.filter(pk=thread_id, created_by=user).first()
    source_id = (thread.metadata or {}).get("source_skill_id") if thread else None
    if not source_id:
        skill = get_editable_skill_for_user(user, slug)
        if not skill:
            return None, f"Skill '{slug}' not found or not editable."
        return skill, None

    try:
        source = AgentSkill.objects.get(pk=source_id)
    except AgentSkill.DoesNotExist:
        return None, "Source skill no longer exists."

    if can_edit_skill(user, source):
        return source, None

    # Fork on first write, then rewrite metadata so future edits target the fork.
    fork = fork_skill(user, source, copy_templates=True)
    meta = thread.metadata or {}
    meta["source_skill_id"] = str(fork.id)
    thread.metadata = meta
    thread.save(update_fields=["metadata"])
    return fork, None


def load_skill_field_into_canvas(thread_id, skill, field_name: str, *, canvas_name: str = "") -> "ChatCanvas":
    """Create or refresh a canvas holding a skill field's content for editing.

    Shared by ``ShowSkillFieldInCanvasTool`` (in-chat tool call) and the
    ``edit_skill_in_chat`` view (server-side pre-population). Returns the
    saved ChatCanvas. Sets it as the thread's active canvas. Idempotent on
    canvas title (uses the unique_canvas_title_per_thread constraint).
    """
    from django.db import IntegrityError

    from agent_skills.models import SkillTemplate
    from chat.models import ChatCanvas
    from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint, set_active_canvas

    if field_name in ("instructions", "description"):
        content = getattr(skill, field_name) or ""
    else:
        try:
            tmpl = skill.templates.get(name=field_name)
            content = tmpl.content
        except SkillTemplate.DoesNotExist as exc:
            raise ValueError(
                f"Template '{field_name}' not found on skill '{skill.slug}'."
            ) from exc

    title = canvas_name or f"{skill.name} \u2014 {field_name}"
    content = content[:CANVAS_MAX_CHARS]

    try:
        canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
            thread_id=thread_id, title=title,
        )
        canvas.content = content
        canvas.save(update_fields=["content", "updated_at"])
        created = False
    except ChatCanvas.DoesNotExist:
        try:
            canvas = ChatCanvas.objects.create(
                thread_id=thread_id, title=title, content=content,
            )
            created = True
        except IntegrityError:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=title,
            )
            canvas.content = content
            canvas.save(update_fields=["content", "updated_at"])
            created = False

    cp = create_canvas_checkpoint(canvas, source="import", description=f"Loaded {field_name}")
    if created:
        canvas.accepted_checkpoint = cp
        canvas.save(update_fields=["accepted_checkpoint"])

    set_active_canvas(thread_id, canvas)
    return canvas


# -- Input schemas --


class CreateSkillInput(ReasonBaseModel):
    name: str = Field(description="Name for the new skill.")


class SaveCanvasToSkillFieldInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to save to.")
    field_name: str = Field(
        description=(
            "Field to save canvas content to: 'instructions', 'description', "
            "or a template name."
        )
    )
    canvas_name: str = Field(
        default="",
        description="Title of the canvas to save from. If omitted, uses the active canvas.",
    )


class ShowSkillFieldInCanvasInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to read from.")
    field_name: str = Field(
        description=(
            "Field to show: 'instructions', 'description', or a template name."
        )
    )
    canvas_name: str = Field(
        default="",
        description="Title for the canvas tab. If omitted, uses '{skill_name} — {field_name}'.",
    )


class TextEdit(BaseModel):
    field: str = Field(description="Field name to edit (e.g. 'description').")
    old_text: str = Field(description="Exact text to find.")
    new_text: str = Field(description="Replacement text.")


class EditSkillInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to edit.")
    updates: dict = Field(
        default_factory=dict,
        description="Optional keys: name, new_slug, tool_names, is_active.",
    )
    text_edits: list[TextEdit] = Field(
        default_factory=list,
        description="Find-replace edits for text fields like description.",
    )
    delete_templates: list[str] = Field(
        default_factory=list,
        description="Template names to delete.",
    )


class DeleteSkillInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to delete.")


class ViewTemplateInput(ReasonBaseModel):
    template_name: str = Field(description="Name of the template to view.")


class LoadTemplateToCanvasInput(ReasonBaseModel):
    template_name: str = Field(description="Name of the template to load into the canvas.")
    canvas_name: str = Field(
        default="",
        description="Title for the canvas tab. If omitted, uses the template name.",
    )


class InspectToolInput(ReasonBaseModel):
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

    def _run(self, name: str, **kwargs) -> str:
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

    def _run(self, skill_slug: str, field_name: str, canvas_name: str = "", **kwargs) -> str:
        from agent_skills.models import SkillTemplate
        from chat.services import resolve_canvas

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

        skill, err = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if err:
            return json.dumps({"status": "error", "message": err})

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            return json.dumps({
                "status": "error",
                "message": err,
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

    def _run(self, skill_slug: str, field_name: str, canvas_name: str = "", **kwargs) -> str:
        from agent_skills.services import get_available_skills

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

        try:
            canvas = load_skill_field_into_canvas(
                thread_id, skill, field_name, canvas_name=canvas_name,
            )
        except ValueError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        return json.dumps({
            "status": "ok",
            "title": canvas.title,
            "canvas_id": str(canvas.pk),
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
        delete_templates: list[str] | None = None,
        **kwargs,
    ) -> str:
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

        skill, err = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if err:
            return json.dumps({"status": "error", "message": err})

        updates = updates or {}
        update_fields = ["updated_at"]

        if "name" in updates:
            skill.name = updates["name"]
            update_fields.append("name")
        if "new_slug" in updates:
            from agent_skills.models import AgentSkill

            new_slug = updates["new_slug"]
            conflict = AgentSkill.objects.filter(
                slug=new_slug, level=skill.level, **{
                    "organization": skill.organization} if skill.level == "org"
                    else {"created_by": skill.created_by} if skill.level == "user"
                    else {}
            ).exclude(pk=skill.pk).exists()
            if conflict:
                return json.dumps({
                    "status": "error",
                    "message": f"Slug '{new_slug}' is already taken.",
                })
            skill.slug = new_slug
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

        if len(update_fields) > 1 or applied > 0:
            skill.save(update_fields=update_fields)

        # Delete templates by name
        templates_deleted = 0
        if delete_templates:
            from agent_skills.models import SkillTemplate

            deleted_count, _ = SkillTemplate.objects.filter(
                skill=skill, name__in=delete_templates,
            ).delete()
            templates_deleted = deleted_count

        return json.dumps({
            "status": "ok",
            "slug": skill.slug,
            "name": skill.name,
            "id": str(skill.id),
            "is_active": skill.is_active,
            "tool_names": skill.tool_names,
            "edits_applied": applied,
            "edits_failed": failed,
            "templates_deleted": templates_deleted,
        })


class DeleteSkillTool(ContextAwareTool):
    """Delete a user or org skill."""

    name: str = "delete_skill"
    description: str = "Delete a skill that the user owns. System skills cannot be deleted."
    args_schema: type[BaseModel] = DeleteSkillInput
    section: str = "skills"

    def _run(self, skill_slug: str, **kwargs) -> str:
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


class ViewTemplateTool(ContextAwareTool):
    """View the content of a template from the active skill."""

    name: str = "view_template"
    description: str = (
        "View the full content of a named template from the current skill. "
        "Returns the template text so you can reference it when generating output."
    )
    args_schema: type[BaseModel] = ViewTemplateInput
    section: str = "skills"

    def _run(self, template_name: str, **kwargs) -> str:
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

    def _run(self, template_name: str, canvas_name: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from agent_skills.models import SkillTemplate
        from chat.models import ChatCanvas, ChatThread
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint, set_active_canvas

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
        title = canvas_name or tmpl.name

        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=title,
            )
            canvas.content = content
            canvas.save(update_fields=["content", "updated_at"])
            created = False
        except ChatCanvas.DoesNotExist:
            try:
                canvas = ChatCanvas.objects.create(
                    thread_id=thread_id, title=title, content=content,
                )
                created = True
            except IntegrityError:
                canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    thread_id=thread_id, title=title,
                )
                canvas.content = content
                canvas.save(update_fields=["content", "updated_at"])
                created = False

        cp = create_canvas_checkpoint(canvas, source="import", description=f"Loaded template: {template_name}")
        if created:
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])

        set_active_canvas(thread_id, canvas)

        return json.dumps({
            "status": "ok",
            "title": title,
            "canvas_id": str(canvas.pk),
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
    args_schema: type[BaseModel] = ReasonBaseModel
    section: str = "skills"

    def _run(self, **kwargs) -> str:
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

    def _run(self, tool_name: str, **kwargs) -> str:
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


# NOTE: Section is "chat" (not "skills" like the tools above) because this
# tool manages which skill is attached to the thread — it is exposed to the
# base chat agent, not gated behind an already-attached skill.
class AttachSkillsInput(ReasonBaseModel):
    skill_slugs: list[str] = Field(
        default_factory=list,
        description=(
            "Slugs of skills to attach. Pass an empty list to detach the "
            "current skill. Currently only one skill may be attached per "
            "thread — pass at most one slug."
        ),
    )


class AttachSkillsTool(ContextAwareTool):
    name: str = "attach_skills"
    description: str = (
        "Attach a skill from the available skills list to this chat thread, "
        "replacing any skill currently attached. Pass an empty list to "
        "detach. Skill-declared tools become available on your next turn, "
        "not the turn you attach."
    )
    args_schema: type[BaseModel] = AttachSkillsInput
    section: str = "chat"

    def _run(self, skill_slugs: list[str] | None = None, **kwargs) -> str:
        from django.contrib.auth import get_user_model

        from agent_skills.services import get_available_skills
        from chat.models import ChatThread

        user_id = self.context.user_id if self.context else None
        thread_id = self.context.conversation_id if self.context else None
        if not user_id or not thread_id:
            return json.dumps({"status": "error", "message": "No context available."})

        slugs = [s.strip() for s in (skill_slugs or []) if s and s.strip()]
        if len(slugs) > 1:
            return json.dumps({
                "status": "error",
                "message": "Only one skill can be attached per thread. Pass at most one slug.",
            })

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        try:
            thread = ChatThread.objects.select_related("skill").get(
                pk=thread_id, created_by=user,
            )
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        previous_skill_id = str(thread.skill_id) if thread.skill_id else None
        previous_skill_name = thread.skill.name if thread.skill else None

        if not slugs:
            if thread.skill_id is None:
                return json.dumps({
                    "status": "ok",
                    "detached": False,
                    "attached_skill_id": None,
                    "attached_skill_name": None,
                    "message": "No skill was attached; nothing to detach.",
                })
            ChatThread.objects.filter(pk=thread_id).update(skill=None)
            return json.dumps({
                "status": "ok",
                "detached": True,
                "previous_skill_id": previous_skill_id,
                "previous_skill_name": previous_skill_name,
                "attached_skill_id": None,
                "attached_skill_name": None,
            })

        slug = slugs[0]
        available = get_available_skills(user)
        chosen = next((s for s in available if s.slug == slug), None)
        if chosen is None:
            return json.dumps({
                "status": "error",
                "message": f"Skill '{slug}' is not available to this user.",
                "available_slugs": [s.slug for s in available],
            })

        if thread.skill_id == chosen.id:
            return json.dumps({
                "status": "ok",
                "detached": False,
                "no_change": True,
                "attached_skill_id": str(chosen.id),
                "attached_skill_name": chosen.name,
            })

        ChatThread.objects.filter(pk=thread_id).update(skill_id=chosen.id)

        return json.dumps({
            "status": "ok",
            "detached": bool(previous_skill_id),
            "previous_skill_id": previous_skill_id,
            "previous_skill_name": previous_skill_name,
            "attached_skill_id": str(chosen.id),
            "attached_skill_name": chosen.name,
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(CreateSkillTool())
_registry.register_tool(SaveCanvasToSkillFieldTool())
_registry.register_tool(ShowSkillFieldInCanvasTool())
_registry.register_tool(EditSkillTool())
_registry.register_tool(DeleteSkillTool())
_registry.register_tool(ViewTemplateTool())
_registry.register_tool(LoadTemplateToCanvasTool())
_registry.register_tool(ListAllToolsTool())
_registry.register_tool(InspectToolTool())
_registry.register_tool(AttachSkillsTool())
