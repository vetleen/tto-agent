"""Skill management tools for agent use during chat."""

from __future__ import annotations

import json
import logging
from pydantic import BaseModel, Field

from agent_skills.models import MAX_THREAD_SKILLS
from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


class _EmojiResult(BaseModel):
    """Structured-output schema for the skill-emoji auto-pick call."""

    emoji: str = Field(description="A single emoji representing the skill.")


def _generate_emoji_for_skill(name: str, user_id, conversation_id, org_id: int | None = None) -> str:
    """Best-effort: ask the cheap LLM for one emoji for a skill name.

    Returns empty string on any failure. Caller decides whether to persist.
    """
    from core.preferences import resolve_org_feature_model

    model = resolve_org_feature_model(org_id, "skill_emoji")
    if not model:
        return ""

    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    request = ChatRequest(
        messages=[
            Message(role="system", content=(
                "You pick a single emoji to represent a skill by its name. "
                "Return exactly one emoji character, no other text."
            )),
            Message(role="user", content=f"Skill name: {name}"),
        ],
        model=model,
        stream=False,
        tools=[],
        context=RunContext.create(user_id=user_id, conversation_id=conversation_id),
    )
    service = get_llm_service()
    parsed, _ = service.run_structured(request, _EmojiResult)
    return (parsed.emoji or "").strip()[:16]


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
    from agent_skills.services import (
        can_edit_skill,
        fork_skill,
        get_editable_skill_for_user,
        get_skill_for_user,
    )
    from chat.models import ChatThread

    thread = ChatThread.objects.filter(pk=thread_id, created_by=user).first()
    source_id = (thread.metadata or {}).get("source_skill_id") if thread else None
    if not source_id:
        skill = get_editable_skill_for_user(user, slug)
        if not skill:
            return None, f"Skill '{slug}' not found or not editable."
        return skill, None

    # Resolve the source through the access gate — never fork a skill the user
    # can't actually see. source_skill_id is server-set today, but routing the
    # lookup through get_skill_for_user keeps a stray or forged value from
    # cloning an arbitrary skill by UUID.
    source = get_skill_for_user(user, source_id)
    if source is None:
        return None, "Source skill no longer exists or is not accessible."

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
        description="Optional keys: name, new_slug, tool_names.",
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

    name: str = "skill_create"
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

        # Cap at the CharField limit, mirroring the create form — an over-long
        # name raises a DataError on Postgres (invisible on SQLite).
        skill = create_user_skill(user, name[:255])

        # Best-effort emoji auto-pick. Any failure leaves emoji empty; the
        # user can still set one manually via the skill detail form.
        try:
            from accounts.models import get_user_org

            conversation_id = self.context.conversation_id if self.context else None
            # skill_emoji is an org-scoped feature: pass the org so the model
            # resolver honors the org's chosen cheap model / allowed_models
            # instead of falling back to the system default.
            org = get_user_org(user)
            emoji = _generate_emoji_for_skill(
                name, user_id, conversation_id, org_id=org.id if org else None,
            )
            if emoji:
                skill.emoji = emoji
                skill.save(update_fields=["emoji", "updated_at"])
        except Exception:
            logger.exception("Failed to auto-generate emoji for skill %s", skill.id)

        return json.dumps({
            "status": "ok",
            "slug": skill.slug,
            "name": skill.name,
            "emoji": skill.emoji,
            "id": str(skill.id),
        })


class SaveCanvasToSkillFieldTool(ContextAwareTool):
    """Save the current canvas content to a skill field or template."""

    name: str = "skill_field_save"
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

        # Guard against a blank target so an empty/whitespace name can't create
        # a junk SkillTemplate(name="") via the template branch below.
        field_name = (field_name or "").strip()
        if not field_name:
            return json.dumps({"status": "error", "message": "field_name is required."})

        skill, err = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if err:
            return json.dumps({"status": "error", "message": err})

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            return json.dumps({
                "status": "error",
                "message": err,
            })

        # Cap each target at its field's limit: instructions at the model cap,
        # description at the CharField-style 1024 the edit form enforces (it is
        # injected verbatim into the system prompt of every thread using the
        # skill), templates at the shared template cap.
        content = canvas.content
        if field_name == "instructions":
            from agent_skills.models import MAX_INSTRUCTIONS_CHARS

            content = content[:MAX_INSTRUCTIONS_CHARS]
        elif field_name == "description":
            content = content[:1024]
        else:
            from agent_skills.models import MAX_TEMPLATE_CHARS

            content = content[:MAX_TEMPLATE_CHARS]

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

    name: str = "skill_field_load"
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

    name: str = "skill_edit"
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
            # Cap at the CharField limit (DataError on Postgres otherwise) and
            # skip blank names, mirroring the save form's fallback.
            new_name = str(updates["name"]).strip()[:255]
            if new_name:
                skill.name = new_name
                update_fields.append("name")
        if "new_slug" in updates:
            from django.utils.text import slugify

            from agent_skills.models import AgentSkill

            # Slugify + cap at the SlugField's 64-char limit, mirroring the
            # save form (views._apply_skill_form). Without this, a raw value
            # with spaces/uppercase would persist a malformed slug, and a
            # >64-char value would raise a DataError on Postgres (SlugField
            # validators don't run on .save()).
            new_slug = slugify(str(updates["new_slug"]))[:64]
            if not new_slug:
                return json.dumps({
                    "status": "error",
                    "message": "Invalid slug.",
                })
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
            old_slug = skill.slug
            skill.slug = new_slug
            update_fields.append("slug")
        if "tool_names" in updates:
            # Allow-list to skills-section tools only — standard chat/doc tools
            # are always available and don't belong on a skill, and unknown
            # names are dropped rather than passed through to the LLM.
            from agent_skills.services import filter_to_skill_tools

            skill.tool_names = filter_to_skill_tools(updates["tool_names"])
            update_fields.append("tool_names")
        # NOTE: is_active is deliberately NOT editable here. Every skill lookup
        # (list, detail, this tool's own resolution) filters is_active=True, so
        # deactivating would make the skill invisible and unrecoverable outside
        # the Django admin. Deletion has its own explicit tool; per-user
        # disabling is the UI toggle's job.

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

        # A find-replace can grow text fields past their limits; clamp them.
        if "instructions" in update_fields:
            from agent_skills.models import MAX_INSTRUCTIONS_CHARS

            skill.instructions = (skill.instructions or "")[:MAX_INSTRUCTIONS_CHARS]
        if "description" in update_fields:
            skill.description = (skill.description or "")[:1024]

        if len(update_fields) > 1 or applied > 0:
            skill.save(update_fields=update_fields)
            # Keep the user's slug-keyed enable/disable selection pointing at
            # this skill across a rename.
            if "slug" in update_fields:
                from agent_skills.services import migrate_skill_slug_prefs

                migrate_skill_slug_prefs(skill, old_slug, skill.slug)

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

    name: str = "skill_delete"
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


def _resolve_thread_template(thread_id, template_name):
    """Resolve a template by name across every skill attached to a thread.

    Returns ``(template, note)``. ``template`` is None when nothing matches,
    in which case ``note`` is a user-facing error message. When several
    attached skills each define a template with this name, the one from the
    earliest-attached skill wins (deterministic, via the ChatThreadSkill
    ``attached_at, id`` ordering) and ``note`` describes the collision. The
    note is a tool-result field for the model's awareness — it is not prompt
    text and carries no conflict-resolution instructions.
    """
    from agent_skills.models import SkillTemplate
    from chat.models import ChatThreadSkill

    skill_ids = list(
        ChatThreadSkill.objects.filter(thread_id=thread_id).values_list(
            "skill_id", flat=True
        )
    )
    if not skill_ids:
        return None, "No skills attached to this thread."

    matches = list(
        SkillTemplate.objects.filter(
            skill_id__in=skill_ids, name=template_name
        ).select_related("skill")
    )
    if not matches:
        return None, f"Template '{template_name}' not found on any attached skill."
    if len(matches) == 1:
        return matches[0], None

    order = {sid: i for i, sid in enumerate(skill_ids)}
    matches.sort(key=lambda t: order.get(t.skill_id, len(order)))
    winner = matches[0]
    others = ", ".join(t.skill.name for t in matches[1:])
    note = (
        f"Multiple attached skills define a template named '{template_name}'. "
        f"Using the one from '{winner.skill.name}' (attached first); also on: {others}."
    )
    return winner, note


class ViewTemplateTool(ContextAwareTool):
    """View the content of a template from an attached skill."""

    name: str = "skill_template_view"
    description: str = (
        "View the full content of a named template from an attached skill. "
        "Returns the template text so you can reference it when generating output."
    )
    args_schema: type[BaseModel] = ViewTemplateInput
    section: str = "skills"

    def _run(self, template_name: str, **kwargs) -> str:
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        tmpl, note = _resolve_thread_template(thread_id, template_name)
        if tmpl is None:
            return json.dumps({"status": "error", "message": note})

        # Bound what goes into the LLM context. Write paths now cap content at
        # MAX_TEMPLATE_CHARS, so this only fires for pre-existing oversized rows.
        from agent_skills.models import MAX_TEMPLATE_CHARS

        content = tmpl.content
        truncated = len(content) > MAX_TEMPLATE_CHARS
        if truncated:
            content = content[:MAX_TEMPLATE_CHARS]

        result = {
            "status": "ok",
            "template_name": tmpl.name,
            "content": content,
        }
        notes = [note] if note else []
        if truncated:
            result["truncated"] = True
            notes.append(
                f"Template content exceeded {MAX_TEMPLATE_CHARS} characters "
                "and was truncated."
            )
        if notes:
            result["note"] = " ".join(notes)
        return json.dumps(result)


class LoadTemplateToCanvasTool(ContextAwareTool):
    """Load a template from an attached skill into the canvas."""

    name: str = "skill_template_load"
    description: str = (
        "Load a named template from an attached skill into the canvas. "
        "Use this to give the user a starting point they can edit. "
        "This replaces the current canvas content."
    )
    args_schema: type[BaseModel] = LoadTemplateToCanvasInput
    section: str = "skills"

    def _run(self, template_name: str, canvas_name: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from chat.models import ChatCanvas
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint, set_active_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        tmpl, note = _resolve_thread_template(thread_id, template_name)
        if tmpl is None:
            return json.dumps({"status": "error", "message": note})

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

        result = {
            "status": "ok",
            "title": title,
            "canvas_id": str(canvas.pk),
        }
        if note:
            result["note"] = note
        return json.dumps(result)


class ListSkillToolsTool(ContextAwareTool):
    """List skill-specific tools that can be attached to a skill."""

    name: str = "skill_tool_list"
    description: str = (
        "List all skill-specific tools — the tools that must be explicitly "
        "attached to a skill via tool_names. Use this to discover which "
        "tools a skill can use."
    )
    args_schema: type[BaseModel] = ReasonBaseModel
    section: str = "skills"

    def _run(self, **kwargs) -> str:
        registry = get_tool_registry()
        all_tools = registry.list_tools()
        skill_tools = []
        for name, tool in sorted(all_tools.items()):
            if getattr(tool, "section", "chat") == "skills":
                skill_tools.append({
                    "name": name,
                    "description": tool.description or "",
                })
        return json.dumps({
            "status": "ok",
            "tools": skill_tools,
            "note": "Only available when explicitly listed in a skill's tool_names.",
        })


class InspectToolTool(ContextAwareTool):
    """Inspect a tool to see its description and determine if it's appropriate for a skill."""

    name: str = "skill_tool_inspect"
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
            "The complete set of skill slugs that should be attached to this "
            f"thread (up to {MAX_THREAD_SKILLS}). This REPLACES whatever is "
            "currently attached, so pass every slug you want attached — not "
            "just new ones. Pass an empty list to detach all skills."
        ),
    )


class AttachSkillsTool(ContextAwareTool):
    name: str = "chat_skill_attach"
    description: str = (
        "Set the skills attached to this chat thread to the given set of "
        f"slugs (up to {MAX_THREAD_SKILLS}), replacing whatever is currently "
        "attached. Pass an empty list to detach all. Skill-declared tools "
        "become available on your next turn, not the turn you attach."
    )
    args_schema: type[BaseModel] = AttachSkillsInput
    section: str = "chat"

    def _run(self, skill_slugs: list[str] | None = None, **kwargs) -> str:
        from django.contrib.auth import get_user_model
        from django.db import transaction

        from agent_skills.services import get_available_skills
        from chat.models import ChatThread, ChatThreadSkill

        user_id = self.context.user_id if self.context else None
        thread_id = self.context.conversation_id if self.context else None
        if not user_id or not thread_id:
            return json.dumps({"status": "error", "message": "No context available."})

        # Dedupe slugs, preserving the order the caller asked for.
        slugs: list[str] = []
        for raw in (skill_slugs or []):
            slug = raw.strip()
            if slug and slug not in slugs:
                slugs.append(slug)
        if len(slugs) > MAX_THREAD_SKILLS:
            return json.dumps({
                "status": "error",
                "message": (
                    f"At most {MAX_THREAD_SKILLS} skills can be attached per "
                    f"thread; you passed {len(slugs)}."
                ),
            })

        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        try:
            thread = ChatThread.objects.get(pk=thread_id, created_by=user)
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        # Resolve each slug against the access-gated available set, preserving order.
        available = get_available_skills(user)
        by_slug = {s.slug: s for s in available}
        chosen = []
        for slug in slugs:
            skill = by_slug.get(slug)
            if skill is None:
                return json.dumps({
                    "status": "error",
                    "message": f"Skill '{slug}' is not available to this user.",
                    "available_slugs": [s.slug for s in available],
                })
            chosen.append(skill)

        previous_ids = [
            str(i) for i in ChatThreadSkill.objects.filter(
                thread=thread
            ).values_list("skill_id", flat=True)
        ]
        desired_ids = [str(s.id) for s in chosen]
        no_change = previous_ids == desired_ids

        if not no_change:
            # Declarative full replace: the new set IS the desired state, in
            # the caller's order (the ChatThreadSkill id tie-break preserves
            # insertion order for the prompt / tool-union / template lookups).
            with transaction.atomic():
                ChatThreadSkill.objects.filter(thread=thread).delete()
                ChatThreadSkill.objects.bulk_create(
                    [ChatThreadSkill(thread=thread, skill=s) for s in chosen]
                )

        prev_set = set(previous_ids)
        desired_set = set(desired_ids)
        return json.dumps({
            "status": "ok",
            "no_change": no_change,
            "skills": [
                {"id": str(s.id), "name": s.name, "emoji": s.emoji}
                for s in chosen
            ],
            "added": [i for i in desired_ids if i not in prev_set],
            "removed": [i for i in previous_ids if i not in desired_set],
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
_registry.register_tool(ListSkillToolsTool())
_registry.register_tool(InspectToolTool())
_registry.register_tool(AttachSkillsTool())
