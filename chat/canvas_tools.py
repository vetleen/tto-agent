"""Canvas tools: write and edit a per-thread canvas document."""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from chat.services import CANVAS_MAX_CHARS, canvas_size_fields
from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry
from llm.tools.interfaces import ArgTrim, drop_args, edits_description

logger = logging.getLogger(__name__)

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
    "document tool (document_search / document_list / document_read / document_view_native) "
    "gave you, verbatim, into the content."
)


_CANVAS_READ_WHERE = "Call canvas_read to see the canvas as it is now."


class WriteCanvasInput(ReasonBaseModel):
    title: str = Field(description="Title for the document.")
    content: str = Field(description="Full markdown content of the document.")
    canvas_name: str = Field(
        default="",
        description=(
            "Title of an existing canvas to overwrite. Leave empty to create a new "
            "canvas titled `title`, or to overwrite the existing canvas whose title "
            "equals `title`."
        ),
    )


class EditItem(BaseModel):
    old_text: str = Field(description="Exact text to find and replace.")
    new_text: str = Field(description="Replacement text.")
    reason: str = Field(default="", description="Brief reason for this edit.")


_NOTHING_TO_DO = "Nothing to do: pass edits (old_text → new_text) and/or append."

_APPEND_DESCRIPTION = (
    "Text to add at the end of the canvas, after a blank line (no old_text needed; "
    "applied after any edits). Use it to write a long document in parts."
)


def _edit_args_dropped(args: dict) -> dict:
    """The big, re-readable arguments of an edit call, for ``drop_args``."""
    dropped = {}
    if "edits" in args:
        dropped["edits"] = edits_description(args["edits"])
    if isinstance(args.get("append"), str) and args["append"]:
        dropped["append"] = f"the {len(args['append']):,}-char appended text"
    return dropped


def _edit_checkpoint_description(applied: int, append: str) -> str:
    parts = ["Edited %d section(s)" % applied] if applied else []
    if append:
        parts.append("appended text")
    return ", ".join(parts).capitalize()


def append_to_canvas(content: str, append: str) -> str:
    """``content`` with ``append`` added at the end, after a blank line."""
    if not append:
        return content
    head = (content or "").rstrip("\n")
    return f"{head}\n\n{append}" if head.strip() else append


class EditCanvasInput(ReasonBaseModel):
    edits: list[EditItem] = Field(
        default_factory=list,
        description="List of targeted find-replace edits to apply sequentially.",
    )
    append: str = Field(default="", description=_APPEND_DESCRIPTION)
    canvas_name: str = Field(
        default="",
        description=(
            "Title of the canvas to edit. Required when the thread has more than one "
            "canvas (the error lists the titles); may be omitted when there is exactly one."
        ),
    )

    @model_validator(mode="after")
    def _edits_or_append(self):
        # A copied trimmed call (edits/append dropped) must fail validation,
        # never reach the canvas.
        if not self.edits and not self.append:
            raise ValueError(_NOTHING_TO_DO)
        return self

    @field_validator("edits", mode="before")
    @classmethod
    def _coerce_and_clean_edits(cls, value):
        """Tolerate the common ways models malform the ``edits`` array.

        1. Some tool-calling models serialize the whole array as a JSON string
           (e.g. ``'[{"old_text": ...}]'``) instead of a native list, which made
           canvas_edit fail Pydantic validation outright (WILFRED-40). Parse a
           string as JSON; on failure fall through to the normal list-type error.
        2. Some models include a non-actionable item missing the find/replace
           fields — e.g. ``{"reason": "..."}`` with no old_text/new_text
           (WILFRED-6A). Such an item can't be applied as a find-replace, so drop
           it and let the valid edits proceed instead of failing the whole call.
        """
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return value
        if isinstance(value, list):
            return [
                item
                for item in value
                if not isinstance(item, dict)
                or (item.get("old_text") and "new_text" in item)
            ]
        return value


class WriteCanvasTool(ContextAwareTool):
    """Create or completely rewrite the canvas document."""

    name: str = "canvas_write"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Writing document..."
    end_label: str = "Rewrote the canvas text from scratch"
    description: str = (
        "Create or completely rewrite the canvas document for this conversation. "
        "Use this when starting a document from scratch or doing a full rewrite. "
        "For targeted changes to an existing document, prefer the canvas_edit tool instead. "
        f"Content over {CANVAS_MAX_CHARS:,} characters is rejected. A long document cannot be "
        "written in one call: write the first part with canvas_write, then add each next "
        "part with canvas_edit(append=...), each call well under your output limit."
    )
    args_schema: type[BaseModel] = WriteCanvasInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        content = args.get("content")
        if not isinstance(content, str):
            return None
        return drop_args(args, {"content": f"the {len(content):,}-char document"}, _CANVAS_READ_WHERE)

    def _run(self, title: str, content: str, canvas_name: str = "", **kwargs) -> str:
        from django.db import IntegrityError

        from chat.models import ChatCanvas
        from chat.services import (
            MAX_CANVASES_PER_THREAD,
            clip_to_canvas,
            create_canvas_checkpoint,
            focus_canvas,
            oversized_content_rejection,
            record_agent_canvas_view,
            snapshot_user_edits,
            truncation_result_fields,
        )

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # A body over the cap is a runaway, not a document: refuse it before
        # touching the canvas (the clip below stays as a defensive backstop).
        rejection = oversized_content_rejection(content)
        if rejection:
            return json.dumps(rejection)

        # A full rewrite replaces whatever the canvas held, so it also replaces the
        # truncation marker / source link: set when this write is cut, else cleared.
        content, original_chars = clip_to_canvas(content)
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
            canvas.truncated_from_chars = original_chars
            canvas.source_document = None
            canvas.save(update_fields=[
                "title", "content", "truncated_from_chars", "source_document", "updated_at",
            ])
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
                    truncated_from_chars=original_chars,
                )
                created = True
            except IntegrityError:
                # Race condition — title was created concurrently
                canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    thread_id=thread_id, title=title, deleted_at__isnull=True,
                )
                canvas.content = content
                canvas.truncated_from_chars = original_chars
                canvas.source_document = None
                canvas.save(update_fields=[
                    "content", "truncated_from_chars", "source_document", "updated_at",
                ])
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

        record_agent_canvas_view(canvas)  # the agent authored this text
        focus_canvas(thread_id, canvas)

        result = {
            "status": "ok", "title": canvas.title, "canvas_id": str(canvas.pk),
            **canvas_size_fields(canvas.content),
        }
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        if stripped_images:
            result["warning"] = _MD_IMAGE_WARNING % stripped_images
        return json.dumps(result)


class EditCanvasTool(ContextAwareTool):
    """Apply targeted find-replace edits to the existing canvas document."""

    name: str = "canvas_edit"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Editing document..."
    end_label: str = "Edited the canvas text"
    description: str = (
        "Make targeted find-replace edits to the existing canvas document. "
        "Prefer this over canvas_write when the document already exists and you "
        "only need to change specific parts. Each edit specifies old_text to find "
        "and new_text to replace it with. The old_text must match exactly once in "
        "the document — if it appears multiple times, include more surrounding "
        "context to make it unique. If no canvas exists yet, use canvas_write first. "
        "Read the canvas first (canvas_read) so old_text matches the text as it is now. "
        "To add text at the end, pass `append` (no read or old_text needed). "
        f"A replacement or appended text over {CANVAS_MAX_CHARS:,} characters is rejected."
    )
    args_schema: type[BaseModel] = EditCanvasInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        return drop_args(args, _edit_args_dropped(args), _CANVAS_READ_WHERE)

    def _run(
        self, edits: list[dict] | list[EditItem] | None = None, canvas_name: str = "",
        append: str = "", **kwargs,
    ) -> str:
        from chat.services import (
            large_replacement_hint,
            oversized_edit_rejection,
            resolve_canvas,
        )

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        pairs = [
            (
                edit.get("old_text", "") if isinstance(edit, dict) else edit.old_text,
                edit.get("new_text", "") if isinstance(edit, dict) else edit.new_text,
            )
            for edit in edits or []
        ]
        if not pairs and not append:
            return json.dumps({
                "status": "error",
                "message": _NOTHING_TO_DO,
            })
        # A replacement or appended text over the cap is a runaway, not an edit:
        # refuse it before resolving or matching anything.
        rejection = oversized_edit_rejection([*pairs, ("", append)] if append else pairs)
        if rejection:
            return json.dumps(rejection)

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            if not canvas_name and err.startswith("No canvas exists"):
                err = "No canvas exists for this thread. Use canvas_write to create one first."
            return json.dumps({"status": "error", "message": err})

        # Resolve every edit against the original snapshot (not the mutated
        # buffer) so one edit can't match text another edit just inserted.
        from chat.edit_utils import apply_unique_text_edits

        if pairs:
            new_content, applied, failed = apply_unique_text_edits(canvas.content, pairs)
        else:
            new_content, applied, failed = canvas.content, 0, []
        hint = large_replacement_hint(failed)
        new_content = append_to_canvas(new_content, append)

        # Nothing matched (and nothing appended) — report an error and leave the
        # canvas untouched (no save, no checkpoint), mirroring document_edit.
        # Returning "ok" here would let the model believe the edit succeeded.
        if applied == 0 and not append:
            result = {
                "status": "error",
                "applied": 0,
                "failed": failed,
                "message": "No edits applied.",
                "title": canvas.title,
                "canvas_id": str(canvas.pk),
                **canvas_size_fields(canvas.content),
            }
            if hint:
                result["hint"] = hint
            return json.dumps(result)

        # Preserve any uncommitted user edits before applying AI edits.
        from chat.services import snapshot_user_edits
        snapshot_user_edits(canvas)

        # Capture the pre-edit baseline for a canvas that has none yet (user-created,
        # never accepted), so the AI edit below surfaces as a reviewable diff.
        pre_ai_cp = None
        if canvas.accepted_checkpoint_id is None:
            pre_ai_cp = canvas.checkpoints.order_by("-order").first()

        from chat.services import (
            clip_to_canvas,
            create_canvas_checkpoint,
            focus_canvas,
            record_agent_canvas_view,
            truncation_result_fields,
        )

        content, stripped_images = _strip_markdown_images(new_content)
        # An edit keeps any existing marker: the canvas is still derived from a
        # truncated copy. An edit that overflows must not shrink it either — the
        # source's length (e.g. 84k) is what matters, not this edit's (75,032).
        content, overflow_chars = clip_to_canvas(content)
        original_chars = (
            max(canvas.truncated_from_chars or 0, overflow_chars) if overflow_chars else None
        )

        canvas.content = content
        update_fields = ["content", "updated_at"]
        if original_chars:
            canvas.truncated_from_chars = original_chars
            update_fields.append("truncated_from_chars")
        canvas.save(update_fields=update_fields)

        record_agent_canvas_view(canvas)  # the agent authored the edited text
        focus_canvas(thread_id, canvas)
        create_canvas_checkpoint(
            canvas, source="ai_edit",
            description=_edit_checkpoint_description(applied, append),
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
            **canvas_size_fields(canvas.content),
        }
        if append:
            result["appended_chars"] = len(append)
        if hint:
            result["hint"] = hint
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        if stripped_images:
            result["warning"] = _MD_IMAGE_WARNING % stripped_images
        return json.dumps(result)


class ReadCanvasInput(ReasonBaseModel):
    canvas_name: str = Field(
        default="",
        description=(
            "Title of the canvas to read. Required when the thread has more than one "
            "canvas; may be omitted when there is exactly one."
        ),
    )
    start_line: int = Field(
        default=1, ge=1,
        description="First line to return (1-based, inclusive). Default 1.",
    )
    end_line: int | None = Field(
        default=None, ge=1,
        description=(
            "Last line to return (inclusive). Omit for 'to the end'. Lines are the "
            "address: the result tells you total_lines and, for a large canvas, an "
            "outline of headings with their line ranges."
        ),
    )
    max_tokens: int | None = Field(
        default=None, ge=500,
        description=(
            "Page size in tokens. Defaults to the per-read cap for this conversation "
            "(the result states it); a larger value is clamped to that cap. When the "
            "requested lines don't fit, you get the first page cut at a line boundary "
            "plus next_start_line."
        ),
    )


def _read_canvas_result(canvas_content: str, *, context, start_line: int, end_line: int | None,
                        max_tokens: int | None) -> tuple[dict, int]:
    """Shared body of canvas_read / subagent_canvas_read: clamp the page size to the
    run's per-read cap and page the content. Returns ``(view_or_error, cap)``."""
    from chat.services import read_canvas_range
    from llm.context_budget import canvas_read_token_cap

    cap = canvas_read_token_cap(getattr(context, "max_context_tokens", None) if context else None)
    page_tokens = min(max_tokens or cap, cap)
    view = read_canvas_range(canvas_content, start_line=start_line, end_line=end_line, max_tokens=page_tokens)
    if view.get("status") == "error":
        return view, cap
    view["max_tokens"] = page_tokens
    if max_tokens and max_tokens > cap:
        note = view.get("note", "")
        view["note"] = (
            f"{note} max_tokens was capped at {page_tokens:,} for this conversation's context budget."
        ).strip()
    return view, cap


def _read_end_label(result: dict) -> str | None:
    if result.get("status") != "ok":
        return "Couldn't read canvas"
    total = result.get("total_lines")
    if not total:
        return None
    return f"Read canvas (lines {result.get('start_line')}–{result.get('end_line')} of {total})"


class ReadCanvasTool(ContextAwareTool):
    """Return a canvas's text, ranged by line and paged by tokens.

    Within a turn the model only sees what it wrote in its own tool calls — and
    those arguments are trimmed when the tool loop compacts older context. This
    tool is how the current text stays reachable. A canvas that doesn't fit the
    per-read cap (``llm.context_budget.canvas_read_token_cap``) comes back as a
    page plus an outline so the model can read the section it needs."""

    name: str = "canvas_read"
    audience: str = "main"
    stub_despite_recency: bool = True  # big, and a re-read is a local query
    section: str = "skills"
    start_label: str = "Reading canvas..."
    end_label: str = "Read canvas"
    description: str = (
        "Read a canvas's text — the whole canvas when it fits the per-read cap, "
        "otherwise a page of lines plus an outline (headings with line ranges, total "
        "lines, table rows) and next_start_line, so you can continue or jump to the "
        "part you need with start_line/end_line. No canvas text is ever in your "
        "context on its own: read a canvas before you edit, summarise or quote it, "
        "and read it again when the canvas list says it changed since you last read it."
    )
    args_schema: type[BaseModel] = ReadCanvasInput

    def end_label_for_result(self, result: dict) -> str | None:
        return _read_end_label(result)

    def _run(self, canvas_name: str = "", start_line: int = 1, end_line: int | None = None,
             max_tokens: int | None = None, **kwargs) -> str:
        from chat.services import resolve_canvas

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        canvas, err = resolve_canvas(thread_id, canvas_name or None)
        if err:
            return json.dumps({"status": "error", "message": err})
        view, cap = _read_canvas_result(
            canvas.content or "", context=self.context,
            start_line=start_line, end_line=end_line, max_tokens=max_tokens,
        )
        if view.get("status") == "error":
            return json.dumps(view)
        from chat.services import record_agent_canvas_view

        record_agent_canvas_view(canvas)
        result = {"status": "ok", "title": canvas.title, "canvas_id": str(canvas.pk), **view}
        if canvas.truncated_from_chars:
            from chat.services import truncation_result_fields

            # Nested: the top-level ``truncated`` now means "this page was cut".
            fields = truncation_result_fields(canvas.truncated_from_chars)
            result["source_truncation"] = {
                "original_chars": fields["original_chars"],
                "kept_chars": fields["kept_chars"],
                "note": fields["truncation_note"],
            }
        logger.info(
            "canvas_read: canvas=%s lines=%d-%d/%d page_tokens=%d returned_tokens=%d cap=%d truncated=%s",
            canvas.pk, result["start_line"], result["end_line"], result["total_lines"],
            result["max_tokens"], result["tokens"], cap, result["truncated"],
        )
        return json.dumps(result)


class DeleteCanvasInput(ReasonBaseModel):
    canvas_name: str = Field(description="Exact title of the canvas to delete.")


class DeleteCanvasTool(ContextAwareTool):
    """Soft-delete a canvas from the conversation (preserved + user-undoable)."""

    name: str = "canvas_delete"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Deleting canvas..."
    end_label: str = "Deleted canvas"
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


class PasteUserTextInput(ReasonBaseModel):
    message_number: int = Field(
        description=(
            "Which of the user's messages to paste, by its number in the "
            "'# Your messages' list in your context (1 = the first user message)."
        ),
    )
    canvas_name: str = Field(
        default="",
        description=(
            "Target canvas by title. Required when the thread has more than one canvas. "
            "If no canvas exists (or the named one doesn't), a new canvas is created."
        ),
    )
    anchor: str = Field(
        default="",
        description=(
            "Optional. Text already in the canvas to paste immediately after. It "
            "must match exactly once; if it's missing or ambiguous the message is "
            "appended at the end instead."
        ),
    )


class PasteUserTextTool(ContextAwareTool):
    """Paste one of the user's own messages, verbatim, into a canvas."""

    name: str = "canvas_paste_user_text"
    audience: str = "main"
    section: str = "skills"
    start_label: str = "Pasting your message..."
    end_label: str = "Pasted your message into the canvas"
    description: str = (
        "Insert the exact text of one of the user's messages into a canvas, "
        "verbatim — use this instead of retyping the user's words, so nothing is "
        "paraphrased or dropped. Pick the message by its number from the "
        "'# Your messages' list in your context. The text is appended to the "
        "canvas (after the optional anchor phrase if it matches once, otherwise at "
        "the end). If the target canvas doesn't exist yet it is created. "
        "Use this when verbatim replication of the prompt matters."
    )
    args_schema: type[BaseModel] = PasteUserTextInput

    def _run(self, message_number: int, canvas_name: str = "", anchor: str = "", **kwargs) -> str:
        from chat.edit_utils import append_with_anchor
        from chat.models import ChatCanvas
        from chat.services import (
            MAX_CANVASES_PER_THREAD,
            canvas_targeting_error,
            clip_to_canvas,
            create_canvas_checkpoint,
            dedupe_canvas_title,
            focus_canvas,
            list_pasteable_user_messages,
            live_canvases,
            snapshot_user_edits,
            truncation_result_fields,
        )

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # Resolve the requested user message by its per-thread number.
        pairs = list_pasteable_user_messages(thread_id)
        if not pairs:
            return json.dumps({
                "status": "error",
                "message": "There are no user messages to paste yet.",
            })
        by_num = {n: m for n, m in pairs}
        msg = by_num.get(message_number)
        if msg is None:
            latest = pairs[-1][0]
            return json.dumps({
                "status": "error",
                "message": f"No user message #{message_number}. Valid numbers are 1-{latest}.",
                "latest_message_number": latest,
            })
        text = msg.content or ""

        # Resolve the target canvas, or create one (auto-create per the tool contract).
        created = False
        if canvas_name:
            try:
                canvas = ChatCanvas.objects.select_related("accepted_checkpoint").get(
                    thread_id=thread_id, title=canvas_name[:255], deleted_at__isnull=True,
                )
            except ChatCanvas.DoesNotExist:
                canvas = None
        else:
            existing = list(live_canvases(thread_id)[: MAX_CANVASES_PER_THREAD + 1])
            if len(existing) > 1:
                return json.dumps({
                    "status": "error",
                    "message": canvas_targeting_error([c.title for c in existing]),
                })
            canvas = existing[0] if existing else None

        if canvas is None:
            if ChatCanvas.objects.filter(
                thread_id=thread_id, deleted_at__isnull=True,
            ).count() >= MAX_CANVASES_PER_THREAD:
                return json.dumps({
                    "status": "error",
                    "message": f"Maximum of {MAX_CANVASES_PER_THREAD} canvases per thread reached.",
                })
            title = dedupe_canvas_title(thread_id, canvas_name or "Untitled document")
            canvas = ChatCanvas.objects.create(thread_id=thread_id, title=title, content="")
            created = True

        was_empty = not canvas.content
        new_content, inserted = append_with_anchor(canvas.content, text, anchor)
        new_content, overflow_chars = clip_to_canvas(new_content)
        # Keep any existing marker (still a truncated copy); never shrink it.
        original_chars = (
            max(canvas.truncated_from_chars or 0, overflow_chars) if overflow_chars else None
        )
        update_fields = ["content", "updated_at"]
        if original_chars:
            canvas.truncated_from_chars = original_chars
            update_fields.append("truncated_from_chars")

        description = f"Pasted your message #{message_number}"
        if created:
            # Fresh canvas: establish an accepted baseline so later edits diff cleanly.
            canvas.content = new_content
            canvas.save(update_fields=update_fields)
            cp = create_canvas_checkpoint(canvas, source="original", description=description)
            canvas.accepted_checkpoint = cp
            canvas.save(update_fields=["accepted_checkpoint"])
        else:
            snapshot_user_edits(canvas)
            pre_ai_cp = None
            if canvas.accepted_checkpoint_id is None:
                pre_ai_cp = canvas.checkpoints.order_by("-order").first()
            canvas.content = new_content
            canvas.save(update_fields=update_fields)
            create_canvas_checkpoint(canvas, source="ai_edit", description=description)
            if pre_ai_cp is not None and canvas.accepted_checkpoint_id is None:
                canvas.accepted_checkpoint = pre_ai_cp
                canvas.save(update_fields=["accepted_checkpoint"])

        focus_canvas(thread_id, canvas)

        result = {
            "status": "ok",
            "canvas_id": str(canvas.pk),
            "title": canvas.title,
            "message_number": message_number,
            "inserted_after_anchor": inserted,
        }
        if anchor and not inserted and not was_empty:
            result["note"] = "Anchor text was not found (or not unique) — appended at the end instead."
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        return json.dumps(result)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(WriteCanvasTool())
_registry.register_tool(EditCanvasTool())
_registry.register_tool(ReadCanvasTool())
_registry.register_tool(DeleteCanvasTool())
_registry.register_tool(PasteUserTextTool())
