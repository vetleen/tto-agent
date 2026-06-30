"""Sub-agent canvas tools: the single working document a sub-agent builds.

A sub-agent has exactly one working canvas, stored on its ``SubAgentRun``
(``canvas`` / ``canvas_title``). Unlike the main agent's multi-canvas
``ChatCanvas`` system there is no titling/targeting, no checkpoint/diff review,
and no activation — just one durable scratch document per run. These tools are
``audience="subagent"`` so they reach every sub-agent and never the main agent.

Every write is persisted immediately (durability), and the canvas is returned to
the orchestrator alongside the sub-agent's final message — see
``chat/subagent_service.py`` (``render_canvas_block`` and ``run_subagent``).
"""

from __future__ import annotations

import json
import logging

from django.core.exceptions import ValidationError as DjangoValidationError
from pydantic import BaseModel, Field, field_validator

from chat.canvas_tools import EditItem
from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


def _resolve_run(context):
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
        return None, json.dumps(
            {
                "status": "error",
                "message": "The working canvas is only available inside a sub-agent run.",
            }
        )
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
        "Your working canvas is returned to the orchestrator alongside your final "
        "message, so build substantial structured deliverables here rather than "
        "squeezing them into your final text answer."
    )
    args_schema: type[BaseModel] = SubagentCanvasWriteInput

    def _run(self, content: str, title: str = "", **kwargs) -> str:
        from chat.services import CANVAS_MAX_CHARS

        run, err = _resolve_run(self.context)
        if err:
            return err

        run.canvas = (content or "")[:CANVAS_MAX_CHARS]
        update_fields = ["canvas"]
        if title:
            run.canvas_title = title[:255]
            update_fields.append("canvas_title")
        run.save(update_fields=update_fields)
        return json.dumps({"status": "ok", "chars": len(run.canvas)})


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

    def _run(self, edits: list[dict] | list[EditItem], **kwargs) -> str:
        from chat.edit_utils import apply_unique_text_edits
        from chat.services import CANVAS_MAX_CHARS

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
        new_content, applied, failed = apply_unique_text_edits(run.canvas, pairs)

        # Nothing matched — leave the canvas untouched and report an error, so the
        # model doesn't believe the edit succeeded (mirrors canvas_edit).
        if applied == 0:
            return json.dumps(
                {
                    "status": "error",
                    "applied": 0,
                    "failed": failed,
                    "message": "No edits applied.",
                }
            )

        run.canvas = new_content[:CANVAS_MAX_CHARS]
        run.save(update_fields=["canvas"])
        # Echo the full updated canvas so the model re-syncs after anchored edits.
        return json.dumps(
            {
                "status": "ok",
                "applied": applied,
                "failed": failed,
                "content": run.canvas,
            }
        )


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
        from chat.services import CANVAS_MAX_CHARS
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

        run.canvas = (tmpl.content or "")[:CANVAS_MAX_CHARS]
        run.canvas_title = tmpl.name[:255]
        run.save(update_fields=["canvas", "canvas_title"])
        return json.dumps({"status": "ok", "template": tmpl.name, "content": run.canvas})


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SubagentCanvasWriteTool())
_registry.register_tool(SubagentCanvasEditTool())
_registry.register_tool(SubagentCanvasLoadTemplateTool())
