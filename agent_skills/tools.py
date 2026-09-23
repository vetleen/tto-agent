"""Skill management tools for agent use during chat."""

from __future__ import annotations

import copy
import json
import logging
from typing import Literal

from pydantic import BaseModel, Field

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
    # The calling tool writes to the fork right away and queues the safety scan
    # itself (request_skill_rescan), so the fork doesn't queue a second one.
    fork = fork_skill(user, source, copy_templates=True, rescan=False)
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
            thread_id=thread_id, title=title, deleted_at__isnull=True,
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
                thread_id=thread_id, title=title, deleted_at__isnull=True,
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
        description="Field to save canvas content to: 'instructions' or 'description'."
    )
    canvas_name: str = Field(
        default="",
        description="Title of the canvas to save from. If omitted, uses the active canvas.",
    )


class ShowSkillFieldInCanvasInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to read from.")
    field_name: str = Field(
        description="Field to show: 'instructions' or 'description'."
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


class DeleteSkillInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to delete.")


class ViewTemplateInput(ReasonBaseModel):
    template_name: str = Field(description="Name of the resource to view.")
    pages: str | None = Field(
        default=None,
        description=(
            "Optional 1-based page selection for PDF resources, e.g. '3-5,12'. "
            "Only those pages are attached (as a smaller PDF), saving context. "
            "Omit to view the whole PDF."
        ),
    )
    skill_slug: str = Field(
        default="",
        description=(
            "Optional: slug of the skill you are authoring, to read a resource "
            "on THAT skill. Omit to read from a skill attached to this thread."
        ),
    )


class LoadTemplateToCanvasInput(ReasonBaseModel):
    template_name: str = Field(description="Name of the resource to load into the canvas.")
    canvas_name: str = Field(
        default="",
        description=(
            "Title for the canvas tab (or, with target='deck' and a full-deck "
            "resource, for the new deck). If omitted, uses the resource name."
        ),
    )
    skill_slug: str = Field(
        default="",
        description=(
            "Optional: slug of the skill you are authoring, to load a resource "
            "from THAT skill. Omit to load from a skill attached to this thread."
        ),
    )
    target: Literal["canvas", "deck"] = Field(
        default="canvas",
        description=(
            "'canvas' (default) loads the resource's text into a canvas. 'deck' "
            "loads slide-deck JSON: a single slide is added to a deck, a full "
            "deck becomes a NEW deck (existing decks are never overwritten)."
        ),
    )
    deck_name: str = Field(
        default="",
        description=(
            "target='deck' only: title of the deck to add a single slide to. "
            "Empty = the active deck (a new deck is created if there is none)."
        ),
    )
    position: int = Field(
        default=-1,
        description="target='deck' only: 0-based index to insert a single slide at; -1 appends.",
    )


class SkillResourceListInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill whose resources to list.")


class SkillResourceSaveInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to save the resource on.")
    name: str = Field(description="Resource name (unique within the skill).")
    kind: str = Field(
        default="reference",
        description=(
            "'template' for a fill-in skeleton the agent loads and completes, "
            "'reference' for read-only material."
        ),
    )
    canvas_name: str = Field(
        default="",
        description="Title of the canvas to save from. If omitted, uses the active canvas.",
    )


class SkillResourceAttachInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill to attach the file to.")
    source: str = Field(
        description=(
            "A file/image reference you already have: a [[file:<uuid>]] or "
            "[[image:<uuid>]] token, or the bare uuid."
        )
    )
    kind: str = Field(
        default="reference",
        description="'reference' (read-only material) or 'template' (a fill-in skeleton).",
    )
    name: str = Field(
        default="",
        description="Optional resource name. If omitted, the source file's name is used.",
    )


class SkillResourceUpdateInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill owning the resource.")
    name: str = Field(description="Current name of the resource to update.")
    new_name: str = Field(default="", description="New name (omit to keep the current name).")
    kind: str = Field(
        default="",
        description="New kind: 'reference' or 'template' (omit to keep the current kind).",
    )


class SkillResourceDeleteInput(ReasonBaseModel):
    skill_slug: str = Field(description="Slug of the skill owning the resource(s).")
    names: list[str] = Field(
        default_factory=list, description="Resource names to delete."
    )


class InspectToolInput(ReasonBaseModel):
    tool_name: str = Field(description="Name of the tool to inspect.")


# -- Tools --


class CreateSkillTool(ContextAwareTool):
    """Create a new user-level skill."""

    name: str = "skill_create"
    audience: str = "main"
    start_label: str = "Creating skill..."
    end_label: str = "Created skill"
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
    audience: str = "main"
    start_label: str = "Saving to skill..."
    end_label: str = "Saved to skill"
    description: str = (
        "Save the current canvas content into a skill's instructions or "
        "description (saved verbatim). Resources/templates are managed with the "
        "skill_resource_* tools, not this one."
    )
    args_schema: type[BaseModel] = SaveCanvasToSkillFieldInput
    section: str = "skills"

    def _run(self, skill_slug: str, field_name: str, canvas_name: str = "", **kwargs) -> str:
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

        # Only the two authored text columns are writable here. A template/
        # resource name is rejected with a pointer to the resource tools.
        field_name = (field_name or "").strip()
        if field_name not in ("instructions", "description"):
            return json.dumps({
                "status": "error",
                "message": (
                    "skill_field_save writes only 'instructions' or 'description'. "
                    "To save a resource, use skill_resource_save (text) or "
                    "skill_resource_attach (files)."
                ),
            })

        skill, err = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if err:
            return json.dumps({"status": "error", "message": err})

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            return json.dumps({
                "status": "error",
                "message": err,
            })

        # Cap each column at its limit: instructions at the model cap; description
        # at the CharField-style 1024 the edit form enforces (it is injected
        # verbatim into the system prompt of every thread using the skill).
        content = canvas.content
        if field_name == "instructions":
            from agent_skills.models import MAX_INSTRUCTIONS_CHARS

            content = content[:MAX_INSTRUCTIONS_CHARS]
        else:  # description
            content = content[:1024]

        setattr(skill, field_name, content)
        skill.save(update_fields=[field_name, "updated_at"])

        from agent_skills.resources import request_skill_rescan

        scan = request_skill_rescan(skill, user)

        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "field": field_name,
            "chars_saved": len(content),
            "scan_state": scan["scan_state"],
        })


class ShowSkillFieldInCanvasTool(ContextAwareTool):
    """Load a skill field or template into the canvas for viewing/editing."""

    name: str = "skill_field_load"
    audience: str = "main"
    start_label: str = "Loading skill field..."
    end_label: str = "Loaded skill field to canvas"
    description: str = (
        "Load a skill's instructions or description into the canvas so the user "
        "can view and edit it. To load a resource/template, use skill_resource_load."
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

        field_name = (field_name or "").strip()
        if field_name not in ("instructions", "description"):
            return json.dumps({
                "status": "error",
                "message": (
                    "skill_field_load loads only 'instructions' or 'description'. "
                    "To load a resource into the canvas, use skill_resource_load."
                ),
            })

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
    audience: str = "main"
    start_label: str = "Editing skill..."
    end_label: str = "Updated skill"
    description: str = (
        "Edit a skill's name, slug, or tool_names, or apply find-replace edits "
        "to its description/instructions. Resources are managed with the "
        "skill_resource_* tools."
    )
    args_schema: type[BaseModel] = EditSkillInput
    section: str = "skills"

    def _run(
        self,
        skill_slug: str,
        updates: dict | None = None,
        text_edits: list[dict] | list[TextEdit] | None = None,
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
        old_slug = skill.slug

        if "name" in updates:
            # Cap at the CharField limit (DataError on Postgres otherwise) and
            # skip blank names, mirroring the save form's fallback.
            new_name = str(updates["name"]).strip()[:255]
            if new_name:
                skill.name = new_name
                update_fields.append("name")
                # The slug auto-follows the name unless the user froze it or is
                # also setting one explicitly in this same call.
                if not skill.slug_customized and "new_slug" not in updates:
                    from django.utils.text import slugify

                    from agent_skills.services import (
                        _live_slug_taken,
                        _next_free_slug,
                    )

                    base = slugify(new_name)[:64] or "skill"
                    skill.slug = _next_free_slug(base, _live_slug_taken(skill))
                    if "slug" not in update_fields:
                        update_fields.append("slug")
        if "new_slug" in updates:
            from django.utils.text import slugify

            from agent_skills.services import _live_slug_taken, _next_free_slug

            # Slugify + cap at the SlugField's 64-char limit, mirroring the save
            # form (views._apply_skill_form). Without this, a raw value with
            # spaces/uppercase would persist a malformed slug, and a >64-char
            # value would raise a DataError on Postgres (SlugField validators
            # don't run on .save()). Empty after slugify falls back to the
            # name-derived slug. Collisions resolve with a running-number suffix
            # instead of erroring, and setting an explicit slug freezes it from
            # future name-driven auto-reslugging.
            base = (
                slugify(str(updates["new_slug"]))[:64]
                or slugify(skill.name)[:64]
                or "skill"
            )
            skill.slug = _next_free_slug(base, _live_slug_taken(skill))
            skill.slug_customized = True
            if "slug" not in update_fields:
                update_fields.append("slug")
            update_fields.append("slug_customized")
        if "tool_names" in updates:
            # Allow-list to skills-section tools only — standard chat/doc tools
            # are always available and don't belong on a skill, and unknown
            # names are dropped rather than passed through to the LLM.
            from agent_skills.services import filter_to_skill_tools

            skill.tool_names = filter_to_skill_tools(
                updates["tool_names"], skill_audience=skill.audience
            )
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

        scan_state = skill.scan_state
        if len(update_fields) > 1 or applied > 0:
            skill.save(update_fields=update_fields)
            # Keep the user's slug-keyed enable/disable selection pointing at
            # this skill across a rename.
            if "slug" in update_fields:
                from agent_skills.services import migrate_skill_slug_prefs

                migrate_skill_slug_prefs(skill, old_slug, skill.slug)
            # A changed description/instructions re-queues the safety scan; a
            # rename / tool-list change alone is a no-op here.
            from agent_skills.resources import request_skill_rescan

            scan_state = request_skill_rescan(skill, user)["scan_state"]

        return json.dumps({
            "status": "ok",
            "slug": skill.slug,
            "name": skill.name,
            "id": str(skill.id),
            "is_active": skill.is_active,
            "tool_names": skill.tool_names,
            "edits_applied": applied,
            "edits_failed": failed,
            "scan_state": scan_state,
        })


class DeleteSkillTool(ContextAwareTool):
    """Delete a user or org skill."""

    name: str = "skill_delete"
    audience: str = "main"
    start_label: str = "Deleting skill..."
    end_label: str = "Deleted skill"
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

        # Soft-delete: retain the row (restorable from the Django admin) and hide
        # it from every list/resolve path. See services.soft_delete_skill.
        from agent_skills.services import soft_delete_skill

        soft_delete_skill(skill)
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
    from agent_skills.models import SkillResource
    from chat.models import ChatThreadSkill

    # Exclude soft-deleted (or deactivated) skills: their attachment rows survive
    # a soft-delete, but their resources must not stay resolvable in a live thread.
    skill_ids = list(
        ChatThreadSkill.objects.filter(
            thread_id=thread_id, skill__deleted_at__isnull=True, skill__is_active=True
        ).values_list("skill_id", flat=True)
    )
    if not skill_ids:
        return None, "No skills attached to this thread."

    # Quarantined resources are never resolvable — they cannot be read or loaded.
    matches = list(
        SkillResource.objects.filter(
            skill_id__in=skill_ids, name=template_name, is_quarantined=False
        ).select_related("skill")
    )
    if not matches:
        return None, f"Resource '{template_name}' not found on any attached skill."
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


def _coerce_kind(raw, default: str | None = None):
    """Coerce a caller-supplied kind to 'reference'/'template'.

    Returns ``default`` (None for "leave unchanged") when the value is blank or
    unrecognized. Mirrors views._kind_from_post.
    """
    from agent_skills.models import SkillResource

    val = (raw or "").strip().lower()
    return val if val in SkillResource.Kind.values else default


def _load_context_user(context):
    """Resolve the acting User from a RunContext. Returns ``(user, error_json)``
    where exactly one is non-None (error_json is a ready-to-return JSON string)."""
    from django.contrib.auth import get_user_model

    user_id = context.user_id if context else None
    if not user_id:
        return None, json.dumps({"status": "error", "message": "No user context."})
    User = get_user_model()
    try:
        return User.objects.get(pk=user_id), None
    except User.DoesNotExist:
        return None, json.dumps({"status": "error", "message": "User not found."})


def _read_skill_by_slug(user, slug: str):
    """The user-accessible skill for ``slug`` (read access via shadowing), or None."""
    from agent_skills.services import get_available_skills

    for s in get_available_skills(user):
        if s.slug == slug:
            return s
    return None


def _resolve_authoring_resource(user, skill_slug: str, name: str, *, allow_quarantined=False):
    """Resolve a resource by name on a specific skill the user can read.

    Returns ``(resource, error)`` — exactly one is non-None. Quarantined
    resources are hidden from view/load (they must not be readable) but visible
    to ``skill_resource_list`` so the author can see why and remove them.
    """
    from agent_skills.models import SkillResource

    skill = _read_skill_by_slug(user, skill_slug)
    if skill is None:
        return None, f"Skill '{skill_slug}' not found."
    qs = skill.templates.filter(name=name)
    if not allow_quarantined:
        qs = qs.filter(is_quarantined=False)
    resource = qs.first()
    if resource is None:
        return None, f"Resource '{name}' not found on skill '{skill_slug}'."
    return resource, None


class ViewTemplateTool(ContextAwareTool):
    """View the content of a resource from an attached skill (read whole, by name).

    Text resources come back wrapped in begin/end markers so the read stays
    legible even if the skill is later detached. PDF/image resources are handed
    to the model inline (as an attached file) rather than as text.
    """

    name: str = "skill_resource_view"
    audience: str = "main"
    start_label: str = "Loading resource..."
    end_label: str = "Viewed resource"
    description: str = (
        "View the full content of a named resource. Text resources are returned "
        "as text; PDF and image resources are attached inline for you to read "
        "directly. Reads from a skill attached to this thread by default; pass "
        "skill_slug to read a resource on the skill you are authoring. Use this "
        "to consult bundled reference material or a template before generating."
    )
    args_schema: type[BaseModel] = ViewTemplateInput
    section: str = "skills"

    def _run(self, template_name: str, pages: str | None = None,
             skill_slug: str = "", **kwargs) -> str:
        from agent_skills.models import MAX_RESOURCE_CHARS, SkillResource

        if skill_slug:
            user, err = _load_context_user(self.context)
            if err:
                return err
            resource, note = _resolve_authoring_resource(user, skill_slug, template_name)
        else:
            thread_id = self.context.conversation_id if self.context else None
            if not thread_id:
                return json.dumps({"status": "error", "message": "No thread context."})
            resource, note = _resolve_thread_template(thread_id, template_name)
        if resource is None:
            return json.dumps({"status": "error", "message": note})

        skill_name = resource.skill.name
        begin = (
            f'--- begin resource "{resource.name}" ({resource.file_type}) '
            f'from skill "{skill_name}" ---'
        )
        end = f'--- end resource "{resource.name}" ---'

        # PDF / image: hand the model the file itself, inline.
        if resource.file_type in (
            SkillResource.FileType.PDF, SkillResource.FileType.IMAGE
        ) and self.context is not None:
            if self._add_native_asset(resource, pages):
                stub = (
                    f"{begin}\nThe {resource.file_type} file is attached below "
                    f"for you to view directly.\n{end}"
                )
                result = {
                    "status": "ok", "resource_name": resource.name,
                    "file_type": resource.file_type, "content": stub,
                }
                if note:
                    result["note"] = note
                return json.dumps(result)
            if resource.file_type == SkillResource.FileType.IMAGE:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"The image resource '{resource.name}' could not be "
                        "attached (attachment budget exhausted this turn)."
                    ),
                })
            # PDF: fall through to the extracted-text fallback below.

        content = (resource.content or "")[:MAX_RESOURCE_CHARS]
        if not content.strip():
            return json.dumps({
                "status": "error",
                "message": f"Resource '{resource.name}' has no readable text content.",
            })
        truncated = len(resource.content or "") > MAX_RESOURCE_CHARS
        result = {
            "status": "ok", "resource_name": resource.name,
            "file_type": resource.file_type,
            "content": f"{begin}\n{content}\n{end}",
        }
        notes = [note] if note else []
        if truncated:
            result["truncated"] = True
            notes.append(
                f"Resource content exceeded {MAX_RESOURCE_CHARS} characters "
                "and was truncated."
            )
        if notes:
            result["note"] = " ".join(notes)
        return json.dumps(result)

    def _add_native_asset(self, resource, pages: str | None = None) -> bool:
        """Queue a PDF/image resource's bytes for inline injection by the
        pipeline. Reads the vision-optimized derivative (``optimized_file``) when
        present, else the pristine ``original_file``. For PDFs, an optional
        ``pages`` selection is sliced into a smaller native sub-PDF and the whole
        thing is losslessly compressed. Returns False when there are no bytes or
        the per-turn native-asset budget is exhausted."""
        import base64
        import logging

        from agent_skills.models import SkillResource

        if not resource.original_file:
            return False
        # Model reads the optimized copy; original_file stays pristine for download.
        src = resource.optimized_file if resource.optimized_file else resource.original_file
        is_pdf = resource.file_type == SkillResource.FileType.PDF
        # Bail BEFORE reading any bytes if the file can't fit the run's remaining
        # native-asset budget (``.size`` is storage metadata — an S3 HEAD, no
        # download). Skipped when a page subset is requested: slicing changes the
        # real size, so we must read to know it.
        if not (is_pdf and pages):
            try:
                raw_size = src.size or 0
            except Exception:
                raw_size = 0
            if raw_size:
                est_b64_chars = ((raw_size + 2) // 3) * 4
                if est_b64_chars > self.context.native_asset_budget_remaining("skill"):
                    return False
        try:
            with src.open("rb") as fh:
                data = fh.read()
        except Exception:
            logging.getLogger(__name__).exception(
                "skill_resource_view: could not read resource %s bytes", resource.pk
            )
            return False

        if is_pdf:
            from chat.pdf_attach import (
                compress_pdf_lossless,
                extract_pdf_pages,
                parse_page_ranges,
                pdf_page_count,
            )
            from django.conf import settings as dj_settings

            if pages:
                data = extract_pdf_pages(data, parse_page_ranges(pages, pdf_page_count(data)))
            data = compress_pdf_lossless(data)
            # Over the per-PDF page cap the native attach would blow the request;
            # returning False makes the caller fall back to the extracted text.
            page_cap = getattr(dj_settings, "NATIVE_REQUEST_MAX_PDF_PAGES", 100)
            n_pages = pdf_page_count(data)
            if 0 < page_cap < n_pages:
                return False
            b64 = base64.b64encode(data).decode("ascii")
            item = {
                "kind": "pdf", "b64": b64,
                "filename": resource.original_filename or resource.name,
                "description": (
                    f"PDF resource '{resource.name}' from skill "
                    f"'{resource.skill.name}'"
                    + (f" (pages {pages})" if pages else "")
                ),
                "extracted_text": resource.content or "",
            }
        else:
            b64 = base64.b64encode(data).decode("ascii")
            item = {
                "kind": "image", "b64": b64,
                "media_type": resource.media_type or "image/png",
                "description": (
                    f"Image resource '{resource.name}' from skill "
                    f"'{resource.skill.name}'"
                ),
            }
        return bool(self.context.try_add_native_asset(item, pathway="skill"))


class LoadTemplateToCanvasTool(ContextAwareTool):
    """Load a template from an attached skill into the canvas."""

    name: str = "skill_resource_load"
    audience: str = "main"
    start_label: str = "Loading template..."
    end_label: str = "Loaded template to canvas"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") == "error":
            return "Couldn't load template"
        loaded_as = result.get("loaded_as")
        if loaded_as == "slide":
            return "Added a slide from template"
        if loaded_as == "deck":
            return "Created a deck from template"
        return None

    description: str = (
        "Load a named text resource into the canvas as an editable starting "
        "point (replaces the current canvas content). Loads from a skill "
        "attached to this thread by default; pass skill_slug to load a resource "
        "from the skill you are authoring. For slide JSON resources pass "
        "target='deck': a single slide is added to the deck (active deck, or "
        "deck_name), a full deck is created as a new deck."
    )
    args_schema: type[BaseModel] = LoadTemplateToCanvasInput
    section: str = "skills"

    def _run(self, template_name: str, canvas_name: str = "",
             skill_slug: str = "", target: str = "canvas", deck_name: str = "",
             position: int = -1, **kwargs) -> str:
        from django.db import IntegrityError

        from chat.models import ChatCanvas
        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint, set_active_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        if skill_slug:
            user, err = _load_context_user(self.context)
            if err:
                return err
            tmpl, note = _resolve_authoring_resource(user, skill_slug, template_name)
        else:
            tmpl, note = _resolve_thread_template(thread_id, template_name)
        if tmpl is None:
            return json.dumps({"status": "error", "message": note})

        if not (tmpl.content or "").strip():
            return json.dumps({
                "status": "error",
                "message": (
                    f"Resource '{tmpl.name}' has no text content to load into the "
                    "canvas (it may be an image or PDF — use skill_resource_view "
                    "to view those)."
                ),
            })

        if target == "deck":
            return self._load_to_deck(
                thread_id, tmpl, note,
                deck_name=deck_name, position=position, title=canvas_name,
            )

        content = tmpl.content[:CANVAS_MAX_CHARS]
        title = canvas_name or tmpl.name

        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=title, deleted_at__isnull=True,
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
                    thread_id=thread_id, title=title, deleted_at__isnull=True,
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

    def _load_to_deck(self, thread_id, tmpl, note, *, deck_name: str,
                      position: int, title: str) -> str:
        """Add a slide resource to a deck, or create a new deck from a deck resource.

        Never overwrites an existing deck: a single slide is inserted (under the
        deck's row lock), a full deck always lands under a fresh, de-duplicated title.
        """
        from chat.slides import schema, service

        try:
            payload = json.loads(tmpl.content)
        except ValueError as exc:
            return _deck_unverified(tmpl.name, [{"path": "(root)", "message": f"Not valid JSON: {exc}"}])

        kind, issues = schema.classify_slide_payload(payload)
        if kind is None:
            return _deck_unverified(tmpl.name, issues)

        if kind == "slide":
            # Drop the slide-level id: a resource slide was likely copied out of a
            # deck, so its id would clash with the target deck's. Element ids are
            # only unique within their slide, so they carry over as-is.
            slide = copy.deepcopy(payload)
            slide.pop("id", None)
            deck, err = service.resolve_deck(thread_id, deck_name or None)
            if err:
                return json.dumps({"status": "error", **err})
            if deck is None:
                return self._create_deck(
                    thread_id, {"version": 1, "size": {"w": 960, "h": 540}, "slides": [slide]},
                    title=title or tmpl.name, note=note, loaded_as="slide",
                )
            deck, content, idx, issues = service.insert_slide(
                deck.pk, slide, position=position,
                description=f"Added a slide from template: {tmpl.name}",
            )
            if deck is None:
                return json.dumps({"status": "error", "message": "The deck was deleted."})
            if issues:
                return json.dumps({
                    "status": "error",
                    "message": "Adding the slide made the deck invalid.",
                    "issues": issues[:10],
                })
            service.activate_deck(thread_id, deck)

            inserted = content["slides"][idx]
            slide_ids = [s.get("id") for s in content["slides"]]
            result = {
                "status": "ok",
                "loaded_as": "slide",
                "deck_id": str(deck.pk),
                "title": deck.title,
                "slide_id": inserted.get("id"),
                "position": idx,
                "slide_json": schema.canonical_deck_text(inserted),
                "slide_ids": slide_ids,
                "changed_slide_ids": [inserted.get("id")],
                "slide_count": len(slide_ids),
            }
            if note:
                result["note"] = note
            return json.dumps(result)

        return self._create_deck(
            thread_id, copy.deepcopy(payload),
            title=title or tmpl.name, note=note, loaded_as="deck",
        )

    def _create_deck(self, thread_id, deck: dict, *, title: str, note, loaded_as: str) -> str:
        from chat.slide_tools import _tool_org, _tool_user
        from chat.slides import schema, service
        from chat.slides import theme as theme_mod

        schema.mint_ids(deck)
        # Keep the resource's own theme; otherwise the new-deck default (user ->
        # org -> Forest), exactly as slides_create_deck seeds it.
        if not deck.get("theme"):
            default = theme_mod.default_theme_for(_tool_user(self), _tool_org(self))
            if default.get("id") != "forest":
                deck["theme"] = theme_mod.theme_to_deck_override(default)

        # A fresh title pins write_deck to its create path — it never overwrites.
        try:
            obj, _created, _old = service.write_deck(
                thread_id, title=service.unique_deck_title(thread_id, title), content=deck,
            )
        except service.DeckLimitError as exc:
            return json.dumps({"status": "error", "message": str(exc)})
        service.activate_deck(thread_id, obj)

        slide_ids = [s.get("id") for s in deck.get("slides", [])]
        result = {
            "status": "ok",
            "loaded_as": loaded_as,
            "deck_id": str(obj.pk),
            "title": obj.title,
            "slide_ids": slide_ids,
            "changed_slide_ids": slide_ids,
            "slide_count": len(slide_ids),
            "deck_json": schema.canonical_deck_text(deck),
        }
        if note:
            result["note"] = note
        return json.dumps(result)


def _deck_unverified(resource_name: str, issues: list[dict]) -> str:
    return json.dumps({
        "status": "error",
        "message": (
            f"The resource '{resource_name}' couldn't be verified as a valid slide "
            "or deck, so the tool couldn't load it. You can still write the "
            "resource to the deck manually with the edit tool (slide_canvas_edit)."
        ),
        "issues": issues[:20],
    })


class SkillResourceListTool(ContextAwareTool):
    """List the resources bundled with a skill the user can read."""

    name: str = "skill_resource_list"
    audience: str = "main"
    start_label: str = "Listing resources..."
    end_label: str = "Listed resources"
    description: str = (
        "List the resources bundled with a skill you're authoring — each with "
        "its name, kind (reference/template), file type, processing status, and "
        "whether it was quarantined. Pass the slug of the skill under edit."
    )
    args_schema: type[BaseModel] = SkillResourceListInput
    section: str = "skills"

    def _run(self, skill_slug: str, **kwargs) -> str:
        user, err = _load_context_user(self.context)
        if err:
            return err
        skill = _read_skill_by_slug(user, skill_slug)
        if skill is None:
            return json.dumps({
                "status": "error", "message": f"Skill '{skill_slug}' not found.",
            })
        resources = [
            {
                "name": r.name,
                "kind": r.kind,
                "file_type": r.file_type,
                "status": r.status,
                "is_quarantined": r.is_quarantined,
                "quarantine_reason": r.quarantine_reason,
                "token_count": r.token_count,
            }
            for r in skill.templates.order_by("name")
        ]
        return json.dumps({
            "status": "ok", "skill_slug": skill.slug, "resources": resources,
        })


class SkillResourceSaveTool(ContextAwareTool):
    """Create or update a text resource by saving canvas content into it."""

    name: str = "skill_resource_save"
    audience: str = "main"
    start_label: str = "Saving resource..."
    end_label: str = "Saved resource"
    description: str = (
        "Create or update a TEXT resource on a skill by saving a canvas tab into "
        "it. Set kind='template' for a fill-in skeleton the agent loads and "
        "completes, or 'reference' for read-only material. For PDF/image/Office "
        "files, use skill_resource_attach instead."
    )
    args_schema: type[BaseModel] = SkillResourceSaveInput
    section: str = "skills"

    def _run(self, skill_slug: str, name: str, kind: str = "reference",
             canvas_name: str = "", **kwargs) -> str:
        from agent_skills.models import SkillResource
        from agent_skills.resources import (
            RESOURCE_COUNT_CAP,
            create_text_resource,
            request_skill_rescan,
            update_resource,
        )
        from chat.services import resolve_canvas

        user, err = _load_context_user(self.context)
        if err:
            return err
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        name = (name or "").strip()
        if not name:
            return json.dumps({"status": "error", "message": "name is required."})
        kind = _coerce_kind(kind, SkillResource.Kind.REFERENCE)

        skill, rerr = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if rerr:
            return json.dumps({"status": "error", "message": rerr})

        canvas, cerr = resolve_canvas(thread_id, canvas_name or None)
        if cerr:
            return json.dumps({"status": "error", "message": cerr})

        existing = skill.templates.filter(name=name).first()
        if existing is not None:
            # Only typed text resources are content-editable; a file resource's
            # content comes from its upload.
            if existing.file_type != SkillResource.FileType.TEXT or existing.original_filename:
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Resource '{name}' is a file, not text. Replace it by "
                        "attaching a new file (skill_resource_attach) and deleting "
                        "the old, or rename/retype it with skill_resource_update."
                    ),
                })
            update_resource(existing, content=canvas.content, kind=kind)
            existing.refresh_from_db()
            resource, created = existing, False
        else:
            if skill.templates.count() >= RESOURCE_COUNT_CAP:
                return json.dumps({
                    "status": "error",
                    "message": f"Resource limit ({RESOURCE_COUNT_CAP}) reached.",
                })
            resource = create_text_resource(
                skill, name=name, content=canvas.content, user=user, kind=kind
            )
            created = True

        scan = request_skill_rescan(skill, user)

        return json.dumps({
            "status": "ok",
            "skill_slug": skill.slug,
            "name": resource.name,
            "kind": resource.kind,
            "created": created,
            "chars_saved": len(resource.content or ""),
            "scan_state": scan["scan_state"],
        })


class SkillResourceAttachTool(ContextAwareTool):
    """Attach a file (from an asset token/uuid) to a skill as a resource."""

    name: str = "skill_resource_attach"
    audience: str = "main"
    start_label: str = "Attaching file..."
    end_label: str = "Attached file — scanning"
    description: str = (
        "Attach a file (PDF, image, or Office document) to a skill as a "
        "resource, using a file/image you already have — a data-room document's "
        "[[file:...]]/[[image:...]] token or a generated image (pass the token or "
        "its uuid). The file is stored and then scanned in the background, so it "
        "starts 'processing'; check skill_resource_list on a later turn to "
        "confirm it became 'ready' (or was 'quarantined')."
    )
    args_schema: type[BaseModel] = SkillResourceAttachInput
    section: str = "skills"

    def _run(self, skill_slug: str, source: str, kind: str = "reference",
             name: str = "", **kwargs) -> str:
        import re as _re

        from django.conf import settings as dj_settings

        from agent_skills.models import SkillResource
        from agent_skills.resources import (
            RESOURCE_COUNT_CAP,
            UnsupportedResourceType,
            create_pending_upload,
            detect_file_type,
        )
        from agent_skills.tasks import process_skill_resource_upload_task
        from chat.assets import (
            file_asset_source,
            image_asset_source,
            user_can_access_asset,
        )
        from chat.models import Asset
        from core.file_types import extension_for_mime

        user, err = _load_context_user(self.context)
        if err:
            return err
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        # Accept a [[file:uuid]]/[[image:uuid]] token or a bare uuid.
        m = _re.search(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            source or "",
        )
        if not m:
            return json.dumps({
                "status": "error",
                "message": "source must be a [[file:uuid]]/[[image:uuid]] token or a uuid.",
            })
        asset_id = m.group(0)

        asset = Asset.objects.filter(id=asset_id).first()
        if asset is None:
            return json.dumps({
                "status": "error", "message": "That file reference no longer exists.",
            })
        if not user_can_access_asset(user, asset):
            return json.dumps({
                "status": "error", "message": "You don't have access to that file.",
            })

        # Resolve bytes + filename + content type by asset kind.
        if asset.kind == Asset.KIND_FILE:
            src, filename, ct = file_asset_source(asset)
        else:
            src, ct = image_asset_source(asset)
            filename = ""
        if src is None:
            return json.dumps({
                "status": "error", "message": "That file's bytes could not be read.",
            })
        if not filename:
            ext = extension_for_mime(ct or "") or "bin"
            base = (name or "image").strip() or "image"
            filename = f"{base}.{ext}"

        try:
            file_type = detect_file_type(filename)
        except UnsupportedResourceType:
            return json.dumps({
                "status": "error",
                "message": f"Unsupported file type for '{filename}'.",
            })

        # Per-type size cap (mirrors the upload view).
        general_max = getattr(dj_settings, "SKILL_RESOURCE_MAX_SIZE_BYTES", 15_000_000)
        image_max = getattr(dj_settings, "SKILL_RESOURCE_IMAGE_MAX_SIZE_BYTES", 25_000_000)
        pdf_max = getattr(dj_settings, "SKILL_RESOURCE_PDF_MAX_SIZE_BYTES", 15_000_000)
        max_size = {
            SkillResource.FileType.IMAGE: image_max,
            SkillResource.FileType.PDF: pdf_max,
        }.get(file_type, general_max)
        size = asset.size_bytes or 0
        if not size:
            try:
                size = src.size or 0
            except Exception:
                size = 0
        if size and size > max_size:
            return json.dumps({
                "status": "error",
                "message": f"File is too large (max {max_size // 1_000_000} MB).",
            })

        skill, rerr = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if rerr:
            return json.dumps({"status": "error", "message": rerr})
        if skill.templates.count() >= RESOURCE_COUNT_CAP:
            return json.dumps({
                "status": "error",
                "message": f"Resource limit ({RESOURCE_COUNT_CAP}) reached.",
            })

        kind = _coerce_kind(kind, SkillResource.Kind.REFERENCE)

        try:
            with src.open("rb") as fh:
                data = fh.read()
        except Exception:
            logger.exception("skill_resource_attach: could not read asset %s bytes", asset_id)
            return json.dumps({
                "status": "error", "message": "That file's bytes could not be read.",
            })

        try:
            resource = create_pending_upload(
                skill, data=data, filename=filename, user=user,
                kind=kind, name=(name.strip() or None),
            )
        except UnsupportedResourceType:
            return json.dumps({
                "status": "error",
                "message": f"Unsupported file type for '{filename}'.",
            })

        # The skill goes PENDING before the worker can finish: with the upload in
        # flight nothing is queued here — process_upload runs the skill's
        # approval gate once the file lands.
        from agent_skills.resources import request_skill_rescan

        scan = request_skill_rescan(skill, user)

        # Extraction + guardrail/PII scan run on the worker (heavy PDF/Office work
        # must not block the turn). The agent polls skill_resource_list.
        process_skill_resource_upload_task.delay(str(resource.id), user.id)

        return json.dumps({
            "status": "processing",
            "skill_slug": skill.slug,
            "resource": {
                "name": resource.name,
                "kind": resource.kind,
                "file_type": resource.file_type,
                "status": resource.status,
            },
            "scan_state": scan["scan_state"],
            "note": (
                "The file is being scanned in the background. Check "
                "skill_resource_list on a later turn for 'ready' or 'quarantined'."
            ),
        })


class SkillResourceUpdateTool(ContextAwareTool):
    """Rename a resource and/or change its kind (no content change)."""

    name: str = "skill_resource_update"
    audience: str = "main"
    start_label: str = "Updating resource..."
    end_label: str = "Updated resource"
    description: str = (
        "Rename a resource and/or change its kind (reference/template) without "
        "touching its content. To change a text resource's content, use "
        "skill_resource_save; to swap a file, attach the new one and delete the old."
    )
    args_schema: type[BaseModel] = SkillResourceUpdateInput
    section: str = "skills"

    def _run(self, skill_slug: str, name: str, new_name: str = "",
             kind: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from agent_skills.resources import request_skill_rescan, update_resource

        user, err = _load_context_user(self.context)
        if err:
            return err
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        new_kind = _coerce_kind(kind, None)  # None → leave unchanged
        new_name = (new_name or "").strip()
        if not new_name and new_kind is None:
            return json.dumps({
                "status": "error",
                "message": "Nothing to update: provide new_name and/or kind.",
            })

        skill, rerr = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if rerr:
            return json.dumps({"status": "error", "message": rerr})

        resource = skill.templates.filter(name=name).first()
        if resource is None:
            return json.dumps({
                "status": "error",
                "message": f"Resource '{name}' not found on skill '{skill.slug}'.",
            })

        try:
            update_resource(resource, name=(new_name or None), kind=new_kind)
        except IntegrityError:
            return json.dumps({
                "status": "error",
                "message": f"A resource named '{new_name}' already exists on this skill.",
            })
        resource.refresh_from_db()
        scan = request_skill_rescan(skill, user)
        return json.dumps({
            "status": "ok", "skill_slug": skill.slug,
            "name": resource.name, "kind": resource.kind,
            "scan_state": scan["scan_state"],
        })


class SkillResourceDeleteTool(ContextAwareTool):
    """Delete one or more resources from a skill by name."""

    name: str = "skill_resource_delete"
    audience: str = "main"
    start_label: str = "Deleting resources..."
    end_label: str = "Deleted resources"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return None
        n = result.get("deleted_count", 0)
        return f"Deleted {n} resource" + ("" if n == 1 else "s")

    description: str = (
        "Delete one or more resources from a skill by name. Pass the skill slug "
        "and the list of resource names."
    )
    args_schema: type[BaseModel] = SkillResourceDeleteInput
    section: str = "skills"

    def _run(self, skill_slug: str, names: list[str] | None = None, **kwargs) -> str:
        from agent_skills.models import SkillResource
        from agent_skills.resources import recompute_standing_tokens, request_skill_rescan

        user, err = _load_context_user(self.context)
        if err:
            return err
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context."})

        names = [str(n).strip() for n in (names or []) if str(n).strip()]
        if not names:
            return json.dumps({"status": "error", "message": "names is required."})

        skill, rerr = resolve_skill_for_thread_edit(user, thread_id, skill_slug)
        if rerr:
            return json.dumps({"status": "error", "message": rerr})

        deleted_count, _ = SkillResource.objects.filter(
            skill=skill, name__in=names
        ).delete()
        recompute_standing_tokens(skill)
        scan = request_skill_rescan(skill, user)
        return json.dumps({
            "status": "ok", "skill_slug": skill.slug, "deleted_count": deleted_count,
            "scan_state": scan["scan_state"],
        })


class ListSkillToolsTool(ContextAwareTool):
    """List skill-specific tools that can be attached to a skill."""

    name: str = "skill_tool_list"
    audience: str = "main"
    start_label: str = "Listing skill tools..."
    end_label: str = "Listed skill tools"
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
    audience: str = "main"
    start_label: str = "Inspecting tool..."
    end_label: str = "Inspected tool"
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


def _skill_entry(skill, attached_by: str) -> dict:
    """One attached skill as the attach/detach tools (and the consumer's
    ``skills.set`` mirror) report it. ``slug`` is what the model needs to
    reference the skill again; ``attached_by`` tells it whether it may detach."""
    return {
        "id": str(skill.id),
        "slug": skill.slug,
        "name": skill.name,
        "emoji": skill.emoji,
        "attached_by": attached_by,
    }


def _live(row) -> bool:
    """Whether an attachment row's skill is still usable (rows survive a
    soft-delete / deactivation; see _resolve_thread_template)."""
    return bool(row.skill.is_active) and row.skill.deleted_at is None


# NOTE: Section is "chat" (not "skills" like the tools above) because these
# two tools manage which skills are attached to the thread — they are exposed
# to the base chat agent, not gated behind an already-attached skill.
class AttachSkillsInput(ReasonBaseModel):
    skill_slugs: list[str] = Field(
        default_factory=list,
        description=(
            "Slugs of the skills to ADD to this thread. Attaching is additive: "
            "everything already attached stays attached, so pass only the new "
            "slugs. An empty list does nothing. To remove a skill you attached, "
            "use chat_skill_detach."
        ),
    )


def _await_skill_approval(skills, user) -> str | None:
    """The approval gate for skills about to be attached to a thread.

    Waits briefly for scans that are still running — a skill the user (or the
    agent) saved moments ago is PENDING for a few seconds while the worker
    scans it — then returns one refusal message naming every skill that isn't
    attachable and what the user must do, or None when all of them passed.
    Runs OUTSIDE the thread row lock (the wait must not stall the UI's
    skills.set or the round's other tools).
    """
    import time

    from django.conf import settings

    from agent_skills.models import AgentSkill
    from agent_skills.resources import bulk_skill_approval

    skills = list(skills)
    if not skills:
        return None

    wait = float(getattr(settings, "SKILL_ATTACH_PENDING_WAIT_SECONDS", 10))
    poll = max(float(getattr(settings, "SKILL_ATTACH_PENDING_POLL_SECONDS", 1)), 0.05)
    pending_state = AgentSkill.ScanState.PENDING
    deadline = time.monotonic() + wait
    while True:
        info = bulk_skill_approval(user, skills)
        still_pending = [
            s for s in skills
            if not info[str(s.id)]["approved"]
            and info[str(s.id)]["scan_state"] == pending_state
        ]
        remaining = deadline - time.monotonic()
        if not still_pending or remaining <= 0:
            break
        time.sleep(min(poll, remaining))
        fresh = {
            str(s.pk): s
            for s in AgentSkill.objects.filter(pk__in=[s.pk for s in skills])
        }
        skills = [fresh.get(str(s.pk), s) for s in skills]

    pending, blocked, needed = [], [], []
    for s in skills:
        verdict = info[str(s.id)]
        if verdict["approved"]:
            continue
        if verdict["scan_state"] == pending_state:
            pending.append(s.slug)
        elif verdict["scan_state"] == AgentSkill.ScanState.BLOCKED:
            detail = (verdict.get("detail") or "").strip()
            blocked.append(f"{s.slug} ({detail[:200]})" if detail else s.slug)
        else:
            needed.append(s.slug)
    if not (pending or blocked or needed):
        return None

    parts = []
    if pending:
        parts.append(
            "Still being safety-scanned — try again in a moment: " + ", ".join(pending) + "."
        )
    if blocked:
        parts.append(
            "Blocked by the safety scan; the user must fix them on the Skills page "
            "before they can be attached: " + ", ".join(blocked) + "."
        )
    if needed:
        parts.append(
            "Not safety-scanned yet; ask the user to enable them on the Skills page "
            "(that runs the scan): " + ", ".join(needed) + "."
        )
    return "These skills can't be attached yet. " + " ".join(parts)


class AttachSkillsTool(ContextAwareTool):
    name: str = "chat_skill_attach"
    audience: str = "main"
    start_label: str = "Attaching skill..."
    end_label: str = "Attached skill"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            # Never let a refusal fall back to the success label.
            return "Couldn't attach skill"
        added = result.get("added") or []
        if not added:
            return "Skill already attached" if result.get("skills") else "No skills attached"
        if len(added) == 1:
            names = {s.get("slug"): s.get("name", "") for s in result.get("skills") or []}
            return f"Attached skill: {names.get(added[0], '')}".rstrip()
        return f"Attached {len(added)} skills"

    description: str = (
        "Attach one or more skills to this chat thread by slug. Additive: skills "
        "already attached stay attached, so pass only the ones to add. A skill's "
        "tools and instructions become active immediately — from your next step in "
        "this same turn — so you can attach a skill and then use its tools right "
        "away. You may attach any number of skills as long as their combined size "
        "fits the thread's skill budget. To remove a skill you attached earlier, "
        "use chat_skill_detach; skills the user attached can't be detached by you."
    )
    args_schema: type[BaseModel] = AttachSkillsInput
    section: str = "chat"

    def _run(self, skill_slugs: list[str] | None = None, **kwargs) -> str:
        from django.contrib.auth import get_user_model
        from django.db import transaction

        from agent_skills.services import get_available_skills
        from chat.models import ChatThread, ChatThreadSkill
        from chat.thread_skills import add_thread_skills, lock_thread

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

        from agent_skills.resources import attach_token_budget, skills_within_budget

        # Approval gate — BEFORE the row lock, because it may wait for a scan
        # that is still running (a wait inside the lock would stall the UI's
        # skills.set and this round's other tools). It applies only to NEWLY-
        # attached skills: refusing one the user already attached would just
        # strand the agent (its manifest already told it to use the resource
        # tools). This lock-free read is a pre-check; the diff is redone below.
        attached_ids = {
            str(sid)
            for sid in ChatThreadSkill.objects.filter(thread=thread)
            .values_list("skill_id", flat=True)
        }
        refusal = _await_skill_approval(
            [s for s in chosen if str(s.id) not in attached_ids], user
        )
        if refusal:
            return json.dumps({"status": "error", "message": refusal})

        # Read → diff → write under the thread's row lock (chat.thread_skills).
        # The pipeline runs a round's tool calls in parallel, and this tool can
        # also race the UI's skills.set: a concurrent writer waits here, then
        # diffs against what the other writer committed instead of racing into
        # the (thread, skill) unique constraint (WILFRED-8P). The early returns
        # commit nothing.
        added: list = []
        with transaction.atomic():
            lock_thread(thread.pk)

            # What's already on the thread (attach order). ALL rows count for
            # the diff — re-listing an attached skill is a no-op, never an
            # approval event — but only rows whose skill is still live feed the
            # budget and the reported set (rows survive a soft-delete).
            rows = list(
                ChatThreadSkill.objects.filter(thread=thread).select_related("skill")
            )
            previous_ids = {str(r.skill_id) for r in rows}
            live_rows = [r for r in rows if _live(r)]
            newly = [s for s in chosen if str(s.id) not in previous_ids]

            if newly:
                # Token budget over the RESULTING set (attached + new). Greedy in
                # attach order, so what's already attached is kept and only new
                # skills can be refused. Aim-relative: a small max_context_tokens
                # caps skills so they can't crowd out history (same budget the UI
                # applies). None → the fixed ceiling.
                aim = getattr(self.context, "max_context_tokens", None) if self.context else None
                _, dropped = skills_within_budget(
                    [r.skill for r in live_rows] + newly, max_context_tokens=aim,
                )
                newly_ids = {str(s.id) for s in newly}
                dropped_new = [s for s in dropped if str(s.id) in newly_ids]
                if dropped_new:
                    return json.dumps({
                        "status": "error",
                        "message": (
                            "Attaching these would exceed this thread's skill size "
                            f"budget (~{attach_token_budget(aim)} tokens): "
                            + ", ".join(s.slug for s in dropped_new)
                            + ". Detach a skill you attached earlier with "
                            "chat_skill_detach, or ask the user to remove one of theirs."
                        ),
                    })

                # Additive: insert only the new rows, marked as agent-attached.
                # Existing rows keep their attach order and origin.
                added_ids = set(
                    add_thread_skills(thread, [s.id for s in newly], attached_by="agent")
                )
                added = [s for s in newly if str(s.id) in added_ids]

        no_change = not added

        # The same-turn, in-memory effects below run only after the write has
        # committed, so a failed write can't leave the turn's live tool set or
        # injected instructions out of step with what was persisted.

        # Unlock the listed skills' tools for the REST OF THIS TURN. The chat
        # pipeline drains ctx.added_tool_names each tool-loop iteration and unions
        # them into the live tool set (SimpleChatPipeline._expand_tools_from_context),
        # so a skill the agent attaches takes effect on its next step, not next turn.
        # Declarative (runs even on no_change); the pipeline dedupes against tools
        # already active. Names come from ctx.skill_tool_map (org-filtered upstream by
        # the consumer) so this never bypasses org tool-toggles.
        ctx = self.context
        if ctx is not None:
            tool_map = getattr(ctx, "skill_tool_map", None) or {}
            for s in chosen:
                for t in tool_map.get(s.slug, []):
                    if t not in ctx.added_tool_names:
                        ctx.added_tool_names.append(t)

            # Inject NEWLY-attached skills' instructions into this same turn so the
            # agent follows them immediately. Skills already attached are already
            # rendered into this turn's system prompt (# Relevant skills), so
            # re-injecting would duplicate. The pipeline drains this into an
            # ephemeral (non-persisted) user message.
            from chat.prompts import _render_one_skill  # local: avoid app import cycle
            for s in added:
                ctx.pending_skill_instructions.append(
                    _render_one_skill(s, attached_by="agent")
                )

        return json.dumps({
            "status": "ok",
            "no_change": no_change,
            "skills": (
                [_skill_entry(r.skill, r.attached_by) for r in live_rows]
                + [_skill_entry(s, "agent") for s in added]
            ),
            "added": [s.slug for s in added],
        })


class DetachSkillsInput(ReasonBaseModel):
    skill_slugs: list[str] = Field(
        default_factory=list,
        description=(
            "Slugs of the skills to detach from this thread (each attached skill's "
            "slug is listed under its heading in '# Relevant skills'). Only skills "
            "you attached yourself can be detached; skills the user attached are "
            "protected, and the call then fails without detaching anything."
        ),
    )


class DetachSkillsTool(ContextAwareTool):
    name: str = "chat_skill_detach"
    audience: str = "main"
    start_label: str = "Detaching skill..."
    end_label: str = "Detached skill"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return "Couldn't detach skill"
        removed = result.get("removed") or []
        if not removed:
            return "No skills detached"
        if len(removed) == 1:
            name = (result.get("removed_names") or [""])[0]
            return f"Detached skill: {name}".rstrip()
        return f"Detached {len(removed)} skills"

    description: str = (
        "Detach skills YOU attached earlier (with chat_skill_attach) from this chat "
        "thread when they're no longer needed. Skills attached by the user can't be "
        "detached by you — ask the user to remove them instead. Detaching takes "
        "effect from the next turn: the skill's tools and instructions stay active "
        "for the rest of the current turn."
    )
    args_schema: type[BaseModel] = DetachSkillsInput
    section: str = "chat"

    def _run(self, skill_slugs: list[str] | None = None, **kwargs) -> str:
        from django.contrib.auth import get_user_model
        from django.db import transaction

        from chat.models import ChatThread, ChatThreadSkill
        from chat.thread_skills import lock_thread, remove_thread_skills

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
        User = get_user_model()
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        try:
            thread = ChatThread.objects.get(pk=thread_id, created_by=user)
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        # Same lock discipline as chat_skill_attach: the protection check and
        # the delete see the same committed rows, even against a concurrent
        # attach or a UI skills.set on this thread. Early returns commit nothing.
        with transaction.atomic():
            lock_thread(thread.pk)
            rows = list(
                ChatThreadSkill.objects.filter(thread=thread).select_related("skill")
            )
            by_slug = {r.skill.slug: r for r in rows}
            unknown = [s for s in slugs if s not in by_slug]
            if unknown:
                return json.dumps({
                    "status": "error",
                    "message": f"Skill '{unknown[0]}' is not attached to this thread.",
                    "attached_slugs": list(by_slug),
                })
            targets = [by_slug[s] for s in slugs]
            # Skills the user attached are theirs to remove, never the agent's:
            # the whole request is refused so a mixed call detaches nothing.
            protected = [
                r for r in targets
                if r.attached_by != ChatThreadSkill.AttachedBy.AGENT
            ]
            if protected:
                return json.dumps({
                    "status": "error",
                    "message": (
                        "These skills were attached by the user and can't be detached "
                        "by you: "
                        + ", ".join(r.skill.slug for r in protected)
                        + ". Ask the user to remove them if they're not needed."
                    ),
                })
            removed_ids = (
                set(remove_thread_skills(thread, [r.skill_id for r in targets]))
                if targets else set()
            )

        # No same-turn effect: the pipeline's tool set is additive-only, so the
        # detached skill's tools/instructions stay live until the next turn.
        removed_rows = [r for r in targets if str(r.skill_id) in removed_ids]
        remaining = [
            r for r in rows if str(r.skill_id) not in removed_ids and _live(r)
        ]
        return json.dumps({
            "status": "ok",
            "no_change": not removed_rows,
            "removed": [r.skill.slug for r in removed_rows],
            "removed_names": [r.skill.name for r in removed_rows],
            "skills": [_skill_entry(r.skill, r.attached_by) for r in remaining],
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
_registry.register_tool(SkillResourceListTool())
_registry.register_tool(SkillResourceSaveTool())
_registry.register_tool(SkillResourceAttachTool())
_registry.register_tool(SkillResourceUpdateTool())
_registry.register_tool(SkillResourceDeleteTool())
_registry.register_tool(ListSkillToolsTool())
_registry.register_tool(InspectToolTool())
_registry.register_tool(AttachSkillsTool())
_registry.register_tool(DetachSkillsTool())
