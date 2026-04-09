"""Canvas tools: write and edit a per-thread canvas document."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry


class ReadCanvasInput(ReasonBaseModel):
    canvas_name: str = Field(
        description="Title of the canvas to read."
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


class ReadCanvasTool(ContextAwareTool):
    """Read the full content of a canvas in this thread by title."""

    name: str = "read_canvas"
    description: str = (
        "Read the full content of a canvas in this thread by title. Use this "
        "to retrieve the text of a canvas that is not currently active — for "
        "example a transcript, source document, or earlier draft sitting in "
        "another tab. The content is returned in the tool result and stays "
        "available in the conversation context for follow-up turns. Only the "
        "active canvas is included in your prompt automatically; use this "
        "tool whenever you need to look at any other canvas."
    )
    args_schema: type[BaseModel] = ReadCanvasInput

    def _run(self, canvas_name: str, **kwargs) -> str:
        from chat.services import resolve_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        canvas, err = resolve_canvas(thread_id, canvas_name)
        if err:
            return json.dumps({"status": "error", "message": err})

        return json.dumps({
            "status": "ok",
            "title": canvas.title,
            "content": canvas.content,
            "canvas_id": str(canvas.pk),
        })


class WriteCanvasTool(ContextAwareTool):
    """Create or completely rewrite the canvas document."""

    name: str = "write_canvas"
    description: str = (
        "Create or completely rewrite the canvas document for this conversation. "
        "Use this when starting a document from scratch or doing a full rewrite. "
        "For targeted changes to an existing document, prefer the edit_canvas tool instead."
    )
    args_schema: type[BaseModel] = WriteCanvasInput

    def _run(self, title: str, content: str, canvas_name: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from chat.models import ChatCanvas
        from chat.services import (
            CANVAS_MAX_CHARS,
            MAX_CANVASES_PER_THREAD,
            create_canvas_checkpoint,
            set_active_canvas,
        )

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        content = content[:CANVAS_MAX_CHARS]

        # Resolve target canvas
        lookup_title = canvas_name or title
        try:
            canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                thread_id=thread_id, title=lookup_title,
            )
            # Update existing canvas
            canvas.title = title
            canvas.content = content
            canvas.save(update_fields=["title", "content", "updated_at"])
            created = False
        except ChatCanvas.DoesNotExist:
            # Check canvas cap
            count = ChatCanvas.objects.filter(thread_id=thread_id).count()
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
                    thread_id=thread_id, title=title,
                )
                canvas.content = content
                canvas.save(update_fields=["content", "updated_at"])
                created = False

        source = "original" if created else "ai_edit"
        cp = create_canvas_checkpoint(canvas, source=source, description="Full rewrite")
        if created:
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])

        set_active_canvas(thread_id, canvas)

        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else ""
        return json.dumps({
            "status": "ok", "title": title, "content": content,
            "accepted_content": accepted_content,
            "canvas_id": str(canvas.pk),
        })


class EditCanvasTool(ContextAwareTool):
    """Apply targeted find-replace edits to the existing canvas document."""

    name: str = "edit_canvas"
    description: str = (
        "Make targeted find-replace edits to the existing canvas document. "
        "Prefer this over write_canvas when the document already exists and you "
        "only need to change specific parts. Each edit specifies old_text to find "
        "and new_text to replace it with. The old_text must match exactly once in "
        "the document — if it appears multiple times, include more surrounding "
        "context to make it unique. If no canvas exists yet, use write_canvas first."
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
                "message": err if canvas_name else "No canvas exists for this thread. Use write_canvas to create one first.",
            })

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

        from chat.services import CANVAS_MAX_CHARS, create_canvas_checkpoint

        truncated = len(content) > CANVAS_MAX_CHARS
        if truncated:
            content = content[:CANVAS_MAX_CHARS]

        canvas.content = content
        canvas.save(update_fields=["content", "updated_at"])

        if applied > 0:
            create_canvas_checkpoint(
                canvas, source="ai_edit",
                description="Edited %d section(s)" % applied,
            )

        accepted_content = canvas.accepted_checkpoint.content if canvas.accepted_checkpoint else ""

        result = {
            "status": "ok",
            "applied": applied,
            "failed": failed,
            "title": canvas.title,
            "content": content,
            "accepted_content": accepted_content,
            "canvas_id": str(canvas.pk),
        }
        if truncated:
            result["note"] = "Content truncated to %d character limit." % CANVAS_MAX_CHARS
        return json.dumps(result)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(ReadCanvasTool())
_registry.register_tool(WriteCanvasTool())
_registry.register_tool(EditCanvasTool())
