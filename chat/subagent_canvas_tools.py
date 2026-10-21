"""Sub-agent canvas tools: the single working document a sub-agent builds.

A sub-agent has exactly one working canvas, stored on its ``SubAgentRun``
(``canvas`` / ``canvas_title``). Unlike the main agent's multi-canvas
``ChatCanvas`` system there is no titling/targeting and no checkpoint/diff
review — just one durable scratch document per run. ``subagent_canvas_read``
reads it (and the orchestrator's shared snapshots, ``SubAgentRun.shared_canvases``)
by line range, paged by the run's per-read token cap, exactly like the main
agent's ``canvas_read``. These tools are ``audience="subagent"`` so they reach
every sub-agent and never the main agent.

Every write is persisted immediately (durability). When the run ends the
orchestrator gets a pointer to the canvas plus the sub-agent's final message and
reads the canvas with ``chat_subagent_report_read`` — see
``chat/subagent_service.py`` (``build_result_notice`` and ``run_subagent``).
"""

from __future__ import annotations

import json
import logging

from django.core.exceptions import ValidationError as DjangoValidationError
from pydantic import BaseModel, Field, field_validator

from chat.canvas_tools import EditItem, _read_canvas_result
from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry
from llm.tools.interfaces import ArgTrim, drop_args, edits_description

logger = logging.getLogger(__name__)


def _resolve_run(
    context, unavailable: str = "The working canvas is only available inside a sub-agent run.",
):
    """Return ``(run, error_json_or_None)`` for the run this tool executes in.

    The tools find their ``SubAgentRun`` via ``context.run_id`` (set to the run
    pk in ``run_subagent``). The main agent's ``run_id`` is a random UUID that
    matches no ``SubAgentRun``, so these tools are inert there even before the
    audience gate.
    """
    from chat.models import SubAgentRun

    run_id = getattr(context, "run_id", None) if context else None
    if not run_id:
        return None, json.dumps(
            {"status": "error", "message": "No sub-agent run context available."}
        )
    try:
        run = SubAgentRun.objects.get(pk=run_id)
    except (SubAgentRun.DoesNotExist, ValueError, DjangoValidationError):
        return None, json.dumps({"status": "error", "message": unavailable})
    return run, None


class SubagentCanvasWriteInput(ReasonBaseModel):
    content: str = Field(description="Full markdown content to write as your working canvas.")
    title: str = Field(
        default="",
        description="Optional title for the canvas (shown to the orchestrator when returned).",
    )


class SubagentCanvasEditInput(ReasonBaseModel):
    edits: list[EditItem] = Field(
        description="List of targeted find-replace edits to apply."
    )

    @field_validator("edits", mode="before")
    @classmethod
    def _coerce_and_clean_edits(cls, value):
        """Tolerate the common ways models malform the ``edits`` array.

        Mirrors ``EditCanvasInput`` in ``chat/canvas_tools.py``: parse an array
        serialized as a JSON string, and drop items missing find/replace fields
        rather than failing the whole call.
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


# Old write/edit arguments are recoverable: the canvas is re-readable with
# ``subagent_canvas_read``, so they are trimmed like the main agent's.
_CANVAS_READ_WHERE = "Call subagent_canvas_read to see the canvas as it is now."


def _read_end_label(result: dict) -> str | None:
    if result.get("status") != "ok":
        return "Couldn't read canvas"
    total = result.get("total_lines")
    if not total:
        return None
    return f"Read canvas (lines {result.get('start_line')}–{result.get('end_line')} of {total})"


class SubagentCanvasLoadTemplateInput(ReasonBaseModel):
    template_name: str = Field(description="Name of the template to load into your working canvas.")


class SubagentCanvasWriteTool(ContextAwareTool):
    """Create or completely rewrite the sub-agent's working canvas."""

    name: str = "subagent_canvas_write"
    audience: str = "subagent"
    start_label: str = "Writing canvas..."
    end_label: str = "Wrote the working canvas"
    description: str = (
        "Replace the entire contents of your working canvas. Use this when "
        "creating your deliverable from scratch or doing a full rewrite. For "
        "targeted changes to existing canvas content, prefer subagent_canvas_edit. "
        "Your working canvas is your deliverable — returned to the orchestrator "
        "alongside your final message — so build substantial structured output "
        "here rather than squeezing it into your final text answer. (For rough "
        "notes you just need to remember while working, use subagent_scratchpad_"
        "append instead; that is private and is not returned.)"
    )
    args_schema: type[BaseModel] = SubagentCanvasWriteInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        content = args.get("content")
        if not isinstance(content, str):
            return None
        return drop_args(args, {"content": f"the {len(content):,}-char content"}, _CANVAS_READ_WHERE)

    def _run(self, content: str, title: str = "", **kwargs) -> str:
        from chat.services import (
            clip_to_canvas,
            oversized_content_rejection,
            truncation_result_fields,
        )

        run, err = _resolve_run(self.context)
        if err:
            return err

        # A body over the cap is a runaway, not a deliverable: refuse it up front.
        rejection = oversized_content_rejection(content)
        if rejection:
            return json.dumps(rejection)

        run.canvas, original_chars = clip_to_canvas(content)
        update_fields = ["canvas"]
        if title:
            run.canvas_title = title[:255]
            update_fields.append("canvas_title")
        run.save(update_fields=update_fields)
        result = {"status": "ok", "chars": len(run.canvas)}
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        return json.dumps(result)


class SubagentCanvasEditTool(ContextAwareTool):
    """Apply targeted find-replace edits to the sub-agent's working canvas."""

    name: str = "subagent_canvas_edit"
    audience: str = "subagent"
    start_label: str = "Editing canvas..."
    end_label: str = "Edited the working canvas"
    description: str = (
        "Make targeted find-replace edits to your working canvas. Prefer this "
        "over subagent_canvas_write when the canvas already has content and you "
        "only need to change specific parts. Each edit specifies old_text to find "
        "and new_text to replace it with. The old_text must match exactly once — "
        "if it appears multiple times, include more surrounding context to make "
        "it unique. If your canvas is empty, use subagent_canvas_write first."
    )
    args_schema: type[BaseModel] = SubagentCanvasEditInput

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> ArgTrim | None:
        if "edits" not in args:
            return None
        return drop_args(args, {"edits": edits_description(args["edits"])}, _CANVAS_READ_WHERE)

    def _run(self, edits: list[dict] | list[EditItem], **kwargs) -> str:
        from chat.edit_utils import apply_unique_text_edits
        from chat.services import (
            clip_to_canvas,
            large_replacement_hint,
            oversized_edit_rejection,
            truncation_result_fields,
        )

        run, err = _resolve_run(self.context)
        if err:
            return err

        pairs = [
            (
                edit.get("old_text", "") if isinstance(edit, dict) else edit.old_text,
                edit.get("new_text", "") if isinstance(edit, dict) else edit.new_text,
            )
            for edit in edits
        ]
        # A replacement text over the cap is a runaway, not an edit.
        rejection = oversized_edit_rejection(pairs, write_tool="subagent_canvas_write")
        if rejection:
            return json.dumps(rejection)

        new_content, applied, failed = apply_unique_text_edits(run.canvas, pairs)
        hint = large_replacement_hint(failed, write_tool="subagent_canvas_write")

        # Nothing matched — leave the canvas untouched and report an error, so the
        # model doesn't believe the edit succeeded (mirrors canvas_edit).
        if applied == 0:
            result = {
                "status": "error",
                "applied": 0,
                "failed": failed,
                "message": "No edits applied.",
            }
            if hint:
                result["hint"] = hint
            return json.dumps(result)

        run.canvas, original_chars = clip_to_canvas(new_content)
        run.save(update_fields=["canvas"])
        result = {"status": "ok", "applied": applied, "failed": failed}
        # Echo the updated canvas so the model re-syncs after anchored edits —
        # but only while it fits the run's per-read cap; a 225k canvas would cost
        # ~56k tokens per edit. Over the cap: its shape, and a pointer to re-read.
        from core.tokens import count_tokens
        from chat.services import canvas_outline
        from llm.context_budget import canvas_read_token_cap

        cap = canvas_read_token_cap(getattr(self.context, "max_context_tokens", None) if self.context else None)
        tokens = count_tokens(run.canvas)
        if tokens <= cap:
            result["content"] = run.canvas
        else:
            result["chars"] = len(run.canvas)
            result["tokens"] = tokens
            result["outline"] = canvas_outline(run.canvas, tokens=tokens)
            result["note"] = (
                f"The canvas is ~{tokens:,} tokens, over the {cap:,}-token echo cap, so its "
                "text is not repeated here; the edit was applied to the full text. Re-read "
                "any part with subagent_canvas_read (start_line/end_line)."
            )
        if hint:
            result["hint"] = hint
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        return json.dumps(result)


class SubagentCanvasReadInput(ReasonBaseModel):
    canvas_name: str = Field(
        default="",
        description=(
            "Leave empty to read your own working canvas. Otherwise the title of a "
            "canvas from the '# Canvases shared by the orchestrator' list in your "
            "instructions."
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
            "Page size in tokens. Defaults to the per-read cap for this run (the result "
            "states it); a larger value is clamped to that cap. When the requested lines "
            "don't fit, you get the first page cut at a line boundary plus next_start_line."
        ),
    )


class SubagentCanvasReadTool(ContextAwareTool):
    """Read the working canvas or one of the orchestrator's shared snapshots,
    ranged by line and paged by the run's per-read token cap."""

    name: str = "subagent_canvas_read"
    audience: str = "subagent"
    start_label: str = "Reading canvas..."
    end_label: str = "Read canvas"
    description: str = (
        "Read your working canvas (leave canvas_name empty) or a canvas the "
        "orchestrator shared with you (by its title from the '# Canvases shared by "
        "the orchestrator' list). The whole canvas comes back when it fits the "
        "per-read cap; otherwise a page of lines plus an outline (headings with line "
        "ranges, total lines, table rows) and next_start_line, so you can continue or "
        "jump to a section with start_line/end_line. Shared snapshots are read-only and "
        "frozen at the time you were created."
    )
    args_schema: type[BaseModel] = SubagentCanvasReadInput

    def end_label_for_result(self, result: dict) -> str | None:
        return _read_end_label(result)

    def _run(self, canvas_name: str = "", start_line: int = 1, end_line: int | None = None,
             max_tokens: int | None = None, **kwargs) -> str:
        run, err = _resolve_run(
            self.context, unavailable="Canvases are only readable inside a sub-agent run.",
        )
        if err:
            return err
        name = (canvas_name or "").strip()
        if not name:
            title = run.canvas_title or "Working document"
            content = run.canvas or ""
        else:
            shared = run.shared_canvases or []
            entry = next((e for e in shared if (e.get("title") or "") == name), None)
            if entry is None:
                return json.dumps({
                    "status": "error",
                    "message": f"No shared canvas titled '{name}'. Leave canvas_name empty for your own working canvas.",
                    "shared_canvases": [e.get("title") for e in shared],
                })
            title = entry.get("title") or name
            content = entry.get("content") or ""
        view, cap = _read_canvas_result(
            content, context=self.context,
            start_line=start_line, end_line=end_line, max_tokens=max_tokens,
        )
        if view.get("status") == "error":
            return json.dumps(view)
        logger.info(
            "subagent_canvas_read: run=%s canvas=%r lines=%d-%d/%d page_tokens=%d cap=%d truncated=%s",
            run.pk, "" if not name else name, view["start_line"], view["end_line"],
            view["total_lines"], view["max_tokens"], cap, view["truncated"],
        )
        return json.dumps({"status": "ok", "title": title, "own_canvas": not name, **view})


class SubagentCanvasLoadTemplateTool(ContextAwareTool):
    """Replace the working canvas with a named template from the specialization."""

    name: str = "subagent_canvas_load_template"
    audience: str = "subagent"
    start_label: str = "Loading template into canvas..."
    end_label: str = "Loaded template into canvas"
    description: str = (
        "Replace your working canvas with a named template from your "
        "specialization. WARNING: this ERASES the current canvas contents, so "
        "only use it to start fresh from a different template. A template is "
        "already loaded into your canvas automatically, so you usually do not "
        "need this."
    )
    args_schema: type[BaseModel] = SubagentCanvasLoadTemplateInput

    def _run(self, template_name: str, **kwargs) -> str:
        from chat.services import clip_to_canvas, truncation_result_fields
        from chat.subagent_service import get_run_specialization_skill

        run, err = _resolve_run(self.context)
        if err:
            return err

        skill = get_run_specialization_skill(run)
        if skill is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "This sub-agent has no specialization, so it has no templates.",
                }
            )

        tmpl = skill.templates.filter(name=template_name).first()
        if tmpl is None:
            available = list(skill.templates.values_list("name", flat=True))
            return json.dumps(
                {
                    "status": "error",
                    "message": f"No template named '{template_name}' in this specialization.",
                    "available_templates": available,
                }
            )

        run.canvas, original_chars = clip_to_canvas(tmpl.content)
        run.canvas_title = tmpl.name[:255]
        run.save(update_fields=["canvas", "canvas_title"])
        result = {"status": "ok", "template": tmpl.name, "content": run.canvas}
        if original_chars:
            result.update(truncation_result_fields(original_chars))
        return json.dumps(result)


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SubagentCanvasWriteTool())
_registry.register_tool(SubagentCanvasEditTool())
_registry.register_tool(SubagentCanvasReadTool())
_registry.register_tool(SubagentCanvasLoadTemplateTool())
