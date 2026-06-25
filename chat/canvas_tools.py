"""Canvas tools: write and edit a per-thread canvas document."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

# Markdown image syntax the model sometimes emits. It never renders in the
# canvas (DOMPurify drops <img>) and a bare URL/filename has no asset behind it,
# so we strip it and tell the model to embed an [[image:uuid|label]] token
# instead. Our own tokens use different syntax and are left untouched.
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def _strip_markdown_images(content: str) -> tuple[str, int]:
    """Replace ``![alt](url)`` with its alt text; return ``(clean, num_removed)``."""
    removed = 0

    def repl(m: re.Match) -> str:
        nonlocal removed
        removed += 1
        return (m.group(1) or "").strip()

    return _MD_IMAGE_RE.sub(repl, content), removed


_MD_IMAGE_WARNING = (
    "Removed %d markdown image(s) (![...](...)) — that syntax does not render in "
    "the canvas. To embed an image, paste the [[image:<uuid>|label]] token a "
    "document tool (document_search / document_list / document_read / document_view_image) "
    "gave you, verbatim, into the content."
)


class ActiveCanvasInput(ReasonBaseModel):
    canvas_names: list[str] = Field(
        description=(
            "List of canvas titles (1-3) to make active. All other canvases "
            "will be deactivated. Active canvases have their full content "
            "included in your context."
        ),
    )


class WriteCanvasInput(ReasonBaseModel):
    title: str = Field(description="Title for the document.")
    content: str = Field(description="Full markdown content of the document.")
    canvas_name: str = Field(
        default="",
        description=(
            "Target an existing canvas by its title. Leave empty to create a "
            "new canvas with the given title, or overwrite the active canvas "
            "if titles match."
        ),
    )


class EditItem(BaseModel):
    old_text: str = Field(description="Exact text to find and replace.")
    new_text: str = Field(description="Replacement text.")
    reason: str = Field(default="", description="Brief reason for this edit.")


class EditCanvasInput(ReasonBaseModel):
    edits: list[EditItem] = Field(
        description="List of targeted find-replace edits to apply sequentially."
    )
    canvas_name: str = Field(
        default="",
        description="Title of the canvas to edit. If omitted, edits the active canvas.",
    )

    @field_validator("edits", mode="before")
    @classmethod
    def _coerce_json_string_edits(cls, value):
        """Tolerate models that pass ``edits`` as a JSON-encoded string.

        Some tool-calling models serialize the array argument as a string
        (e.g. ``'[{"old_text": ...}]'``) instead of a native list, which made
        canvas_edit fail Pydantic validation outright (WILFRED-40). If the value
        is a string, parse it as JSON; on success use the result, otherwise fall
        through to the normal list-type validation error.
        """
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return value
        return value


class ActiveCanvasTool(ContextAwareTool):
    """Set which canvases are active (visible in your context)."""

    name: str = "canvas_activate"
    description: str = (
        "Set which canvases are active. Active canvases have their full "
        "content included in your prompt context (up to 3). When called, "
        "ALL existing active canvases are deactivated, then ONLY the listed "
        "ones are activated. Use this when you need to bring specific "
        "canvases into your working context — for example, to compare two "
        "documents or to reference a source while editing another."
    )
    args_schema: type[BaseModel] = ActiveCanvasInput

    def _run(self, canvas_names: list[str], **kwargs) -> str:
        from chat.services import MAX_ACTIVE_CANVASES, set_active_canvases

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        if len(canvas_names) > MAX_ACTIVE_CANVASES:
            return json.dumps({
                "status": "error",
                "message": f"You can activate at most {MAX_ACTIVE_CANVASES} canvases at a time.",
            })

        activated, errors = set_active_canvases(thread_id, canvas_names)

        result = {
            "status": "ok",
            "activated": [{"title": c.title, "canvas_id": str(c.pk)} for c in activated],
        }
        if errors:
            result["errors"] = errors
        return json.dumps(result)


class WriteCanvasTool(ContextAwareTool):
    """Create or completely rewrite the canvas document."""

    name: str = "canvas_write"
    description: str = (
        "Create or completely rewrite the canvas document for this conversation. "
        "Use this when starting a document from scratch or doing a full rewrite. "
        "For targeted changes to an existing document, prefer the canvas_edit tool instead."
    )
    args_schema: type[BaseModel] = WriteCanvasInput

    def _run(self, title: str, content: str, canvas_name: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from chat.models import ChatCanvas
        from chat.services import (
            CANVAS_MAX_CHARS,
            MAX_CANVASES_PER_THREAD,
            activate_canvas,
            create_canvas_checkpoint,
            snapshot_user_edits,
        )

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        content = content[:CANVAS_MAX_CHARS]
        content, stripped_images = _strip_markdown_images(content)
        # Truncate to the column limit before the lookup so it matches stored titles
        title = title[:255]

        # Resolve target canvas
        lookup_title = (canvas_name or title)[:255]
        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=lookup_title, deleted_at__isnull=True,
            )
            # Preserve any uncommitted user edits before overwriting
            snapshot_user_edits(canvas)
            # Update existing canvas
            canvas.title = title
            canvas.content = content
            canvas.save(update_fields=["title", "content", "updated_at"])
            created = False
        except ChatCanvas.DoesNotExist:
            # Check canvas cap
            count = ChatCanvas.objects.filter(
                thread_id=thread_id, deleted_at__isnull=True,
            ).count()
            if count >= MAX_CANVASES_PER_THREAD:
                return json.dumps({
                    "status": "error",
                    "message": f"Maximum of {MAX_CANVASES_PER_THREAD} canvases per thread reached.",
                })
            try:
                canvas = ChatCanvas.objects.create(
                    thread_id=thread_id, title=title, content=content,
                )
                created = True
            except IntegrityError:
                # Race condition — title was created concurrently
                canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    thread_id=thread_id, title=title, deleted_at__isnull=True,
                )
                canvas.content = content
                canvas.save(update_fields=["content", "updated_at"])
                created = False

        # An existing canvas with no diff baseline yet (user-created, never
        # accepted) needs one established at its pre-edit state, so this AI edit
        # surfaces as a reviewable diff rather than silently overwriting.
        pre_ai_cp = None
        if not created and canvas.accepted_checkpoint_id is None:
            pre_ai_cp = canvas.checkpoints.order_by("-order").first()

        source = "original" if created else "ai_edit"
        cp = create_canvas_checkpoint(canvas, source=source, description="Full rewrite")
        if created:
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
        elif pre_ai_cp is not None and canvas.accepted_checkpoint_id is None:
            canvas.accepted_checkpoint = pre_ai_cp
            canvas.save(update_fields=["accepted_checkpoint"])

        activate_canvas(thread_id, canvas)

        result = {
            "status": "ok", "title": canvas.title, "canvas_id": str(canvas.pk),
        }
        if stripped_images:
            result["warning"] = _MD_IMAGE_WARNING % stripped_images
        return json.dumps(result)


class EditCanvasTool(ContextAwareTool):
    """Apply targeted find-replace edits to the existing canvas document."""

    name: str = "canvas_edit"
    description: str = (
        "Make targeted find-replace edits to the existing canvas document. "
        "Prefer this over canvas_write when the document already exists and you "
        "only need to change specific parts. Each edit specifies old_text to find "
        "and new_text to replace it with. The old_text must match exactly once in "
        "the document — if it appears multiple times, include more surrounding "
        "context to make it unique. If no canvas exists yet, use canvas_write first."
    )
    args_schema: type[BaseModel] = EditCanvasInput

    def _run(self, edits: list[dict] | list[EditItem], canvas_name: str = "", **kwargs) -> str:
        from chat.services import resolve_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            return json.dumps({
                "status": "error",
                "message": err if canvas_name else "No canvas exists for this thread. Use canvas_write to create one first.",
            })

        # Preserve any uncommitted user edits before applying AI edits
        from chat.services import snapshot_user_edits
        snapshot_user_edits(canvas)

        # Capture the pre-edit baseline for a canvas that has none yet (user-created,
        # never accepted), so the AI edit below surfaces as a reviewable diff. Adopted
        # only if an edit is actually applied (see the `applied > 0` block).
        pre_ai_cp = None
        if canvas.accepted_checkpoint_id is None:
            pre_ai_cp = canvas.checkpoints.order_by("-order").first()

        content = canvas.content
        applied = 0
        failed = []

        for edit in edits:
            if isinstance(edit, dict):
                old_text = edit.get("old_text", "")
                new_text = edit.get("new_text", "")
                reason = edit.get("reason", "")
            else:
                old_text = edit.old_text
                new_text = edit.new_text
                reason = edit.reason

            count = content.count(old_text)
            if count == 1:
                content = content.replace(old_text, new_text, 1)
                applied += 1
            elif count > 1:
                failed.append({"old_text": old_text[:80], "error": "Found %d matches — include more surrounding text to make it unique." % count})
            else:
                failed.append({"old_text": old_text[:80], "error": "Text not found in document."})

        from chat.services import CANVAS_MAX_CHARS, activate_canvas, create_canvas_checkpoint

        content, stripped_images = _strip_markdown_images(content)

        truncated = len(content) > CANVAS_MAX_CHARS
        if truncated:
            content = content[:CANVAS_MAX_CHARS]

        canvas.content = content
        canvas.save(update_fields=["content", "updated_at"])

        if applied > 0:
            activate_canvas(thread_id, canvas)
            create_canvas_checkpoint(
                canvas, source="ai_edit",
                description="Edited %d section(s)" % applied,
            )
            if pre_ai_cp is not None and canvas.accepted_checkpoint_id is None:
                canvas.accepted_checkpoint = pre_ai_cp
                canvas.save(update_fields=["accepted_checkpoint"])

        result = {
            "status": "ok",
            "applied": applied,
            "failed": failed,
            "title": canvas.title,
            "canvas_id": str(canvas.pk),
        }
        if truncated:
            result["note"] = "Content truncated to %d character limit." % CANVAS_MAX_CHARS
        if stripped_images:
            result["warning"] = _MD_IMAGE_WARNING % stripped_images
        return json.dumps(result)


class DeleteCanvasInput(ReasonBaseModel):
    canvas_name: str = Field(description="Exact title of the canvas to delete.")


class DeleteCanvasTool(ContextAwareTool):
    """Soft-delete a canvas from the conversation (preserved + user-undoable)."""

    name: str = "canvas_delete"
    description: str = (
        "Delete a canvas from this conversation when it's no longer needed — e.g. "
        "to discard a scratch/draft canvas, or to free room when the per-thread "
        "canvas limit is reached. The canvas's content and version history are "
        "preserved and the user can undo the deletion, so this is safe. Provide the "
        "exact title of the canvas to delete."
    )
    args_schema: type[BaseModel] = DeleteCanvasInput

    def _run(self, canvas_name: str, **kwargs) -> str:
        from chat.models import ChatCanvas
        from chat.services import soft_delete_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        try:
            canvas = ChatCanvas.objects.get(
                thread_id=thread_id, title=canvas_name, deleted_at__isnull=True,
            )
        except ChatCanvas.DoesNotExist:
            available = list(
                ChatCanvas.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
                .order_by("created_at")
                .values_list("title", flat=True)
            )
            return json.dumps({
                "status": "error",
                "message": f"No canvas named '{canvas_name}' in this thread.",
                "available_canvases": available,
            })

        title = canvas.title
        soft_delete_canvas(thread_id, canvas)
        return json.dumps({
            "status": "ok", "canvas_id": str(canvas.pk), "canvas_title": title,
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(ActiveCanvasTool())
_registry.register_tool(WriteCanvasTool())
_registry.register_tool(EditCanvasTool())
_registry.register_tool(DeleteCanvasTool())
