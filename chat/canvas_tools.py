"""Canvas tools: write and edit a per-thread canvas document."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, get_tool_registry


class WriteCanvasInput(BaseModel):
    title: str = Field(description="Title for the document.")
    content: str = Field(description="Full markdown content of the document.")


class EditItem(BaseModel):
    old_text: str = Field(description="Exact text to find and replace.")
    new_text: str = Field(description="Replacement text.")
    reason: str = Field(default="", description="Brief reason for this edit.")


class EditCanvasInput(BaseModel):
    edits: list[EditItem] = Field(
        description="List of targeted find-replace edits to apply sequentially."
    )


class WriteCanvasTool(ContextAwareTool):
    """Create or completely rewrite the canvas document."""

    name: str = "write_canvas"
    description: str = (
        "Create or completely rewrite the canvas document for this conversation. "
        "Use this when starting a document from scratch or doing a full rewrite. "
        "For targeted changes to an existing document, prefer the edit_canvas tool instead."
    )
    args_schema: type[BaseModel] = WriteCanvasInput

    def _run(self, title: str, content: str) -> str:
        from chat.models import ChatCanvas
        from chat.services import CANVAS_MAX_CHARS

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        content = content[:CANVAS_MAX_CHARS]
        ChatCanvas.objects.update_or_create(
            thread_id=thread_id,
            defaults={"title": title, "content": content},
        )
        return json.dumps({"status": "ok", "title": title, "content": content})


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

    def _run(self, edits: list[dict] | list[EditItem]) -> str:
        from chat.models import ChatCanvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        try:
            canvas = ChatCanvas.objects.get(thread_id=thread_id)
        except ChatCanvas.DoesNotExist:
            return json.dumps({
                "status": "error",
                "message": "No canvas exists for this thread. Use write_canvas to create one first.",
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

        from chat.services import CANVAS_MAX_CHARS

        truncated = len(content) > CANVAS_MAX_CHARS
        if truncated:
            content = content[:CANVAS_MAX_CHARS]

        canvas.content = content
        canvas.save(update_fields=["content", "updated_at"])

        result = {
            "status": "ok",
            "applied": applied,
            "failed": failed,
            "title": canvas.title,
            "content": content,
        }
        if truncated:
            result["note"] = "Content truncated to %d character limit." % CANVAS_MAX_CHARS
        return json.dumps(result)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(WriteCanvasTool())
_registry.register_tool(EditCanvasTool())
