"""Agent-only scratchpad — one append-only tool for the assistant's private notes.

The notes are injected into every turn (see ``chat/prompts.py``) and are immune to
history windowing/summarization, so findings survive aggressive tool-result pruning.
They are the assistant's own working memory and are **never shown to the user** — no
``canvas.updated``-style content event is emitted (the tool is deliberately absent
from ``CANVAS_UPDATED_TOOLS``).
"""

from __future__ import annotations

import json
import logging

from django.conf import settings
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel, omitted_arg_marker

logger = logging.getLogger(__name__)


SAVED_NOTE_MARKER = omitted_arg_marker("note", "It is in your scratchpad.")


def _trim_note_args(args: dict) -> dict | None:
    """At an edit point the scratchpad is re-shown in full, so an old note's
    text is a duplicate — keep the call, shrink the note."""
    if not isinstance(args.get("note"), str) or args["note"] == SAVED_NOTE_MARKER:
        return None
    return {**args, "note": SAVED_NOTE_MARKER}


class ScratchpadAppendInput(ReasonBaseModel):
    note: str = Field(
        description=(
            "A note to append to your private scratchpad — key facts, findings, "
            "URLs, or intermediate conclusions you'll need on later turns. Tool "
            "results (web pages, document reads) can be cleared from context to "
            "save space, so save anything important here first. Only you can see "
            "the scratchpad; the user never does."
        )
    )


class ScratchpadAppendTool(ContextAwareTool):
    name: str = "scratchpad_append"
    audience: str = "main"
    section: str = "chat"  # always-on (see core.preferences base tool set)
    start_label: str = "Making a note..."
    end_label: str = "Made a note"
    description: str = (
        "Append a note to your private, persistent scratchpad. The scratchpad is "
        "shown back to you on every turn and survives context pruning and "
        "summarization, so use it to retain findings before large tool results are "
        "cleared. It is agent-only — the user never sees it."
    )
    args_schema: type[BaseModel] = ScratchpadAppendInput
    trim_args_min_chars: int = 200

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> dict | None:
        return _trim_note_args(args)

    def _run(self, note: str, **kwargs) -> str:
        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})
        note = (note or "").strip()
        if not note:
            return json.dumps({"status": "error", "message": "Nothing to append (empty note)."})

        from chat.models import ChatThread

        cap = int(getattr(settings, "SCRATCHPAD_MAX_CHARS", 20_000))
        try:
            thread = ChatThread.objects.get(id=thread_id)
        except ChatThread.DoesNotExist:
            return json.dumps({"status": "error", "message": "Thread not found."})

        existing = thread.scratchpad or ""
        combined = f"{existing}\n\n{note}" if existing else note
        truncated = False
        if len(combined) > cap:
            # Keep the most recent notes (append-only, tail-truncate).
            combined = combined[-cap:]
            truncated = True
        thread.scratchpad = combined
        thread.save(update_fields=["scratchpad"])
        # This turn's notes are re-shown by the tool loop at edit points (the
        # pre-turn scratchpad is already in the per-turn preamble).
        if self.context is not None:
            self.context.scratchpad_turn_notes.append(note)

        result = {"status": "ok", "scratchpad_chars": len(combined)}
        if truncated:
            result["note"] = (
                "Scratchpad is full; oldest notes were dropped to make room."
            )
        return json.dumps(result)


class SubagentScratchpadAppendInput(ReasonBaseModel):
    note: str = Field(
        description=(
            "A note to append to your private scratchpad — key facts, findings, "
            "URLs, or intermediate conclusions you'll need later in this run. Tool "
            "results (web pages, document reads) get cleared from your context as "
            "the run grows, so save anything important here first. The scratchpad "
            "is your private working memory: it is NOT returned to the orchestrator "
            "(use your working canvas for the deliverable) and the user never sees it."
        )
    )


class SubagentScratchpadAppendTool(ContextAwareTool):
    """Run-scoped scratchpad for a sub-agent (see ``ScratchpadAppendTool`` for the
    main-agent analogue). Stored on the ``SubAgentRun``; between edit points the
    note stays visible as this call's own argument, and at every edit point the
    tool loop re-shows the whole scratchpad as a trailing user message, so notes
    survive mid-run pruning."""

    name: str = "subagent_scratchpad_append"
    audience: str = "subagent"
    section: str = "chat"  # always-on for sub-agents (see core.preferences)
    start_label: str = "Making a note..."
    end_label: str = "Made a note"
    description: str = (
        "Append a note to your private, persistent scratchpad. The scratchpad is "
        "shown back to you whenever older context is cleared, so use it to "
        "retain findings before large tool results are cleared. It is private "
        "working memory — NOT part of your deliverable (build that in your working "
        "canvas) and the user never sees it."
    )
    args_schema: type[BaseModel] = SubagentScratchpadAppendInput
    trim_args_min_chars: int = 200

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> dict | None:
        return _trim_note_args(args)

    def _run(self, note: str, **kwargs) -> str:
        run_id = getattr(self.context, "run_id", None) if self.context else None
        if not run_id:
            return json.dumps({"status": "error", "message": "No sub-agent run context available."})
        note = (note or "").strip()
        if not note:
            return json.dumps({"status": "error", "message": "Nothing to append (empty note)."})

        from chat.models import SubAgentRun

        cap = int(getattr(settings, "SCRATCHPAD_MAX_CHARS", 20_000))
        try:
            run = SubAgentRun.objects.get(pk=run_id)
        except SubAgentRun.DoesNotExist:
            return json.dumps({"status": "error", "message": "Sub-agent run not found."})

        existing = run.scratchpad or ""
        combined = f"{existing}\n\n{note}" if existing else note
        truncated = False
        if len(combined) > cap:
            # Keep the most recent notes (append-only, tail-truncate).
            combined = combined[-cap:]
            truncated = True
        run.scratchpad = combined
        run.save(update_fields=["scratchpad"])
        # Update the in-memory copy the tool loop re-shows at edit points, so it
        # needs no DB read (see llm/pipelines/edit_points.py).
        if self.context is not None:
            self.context.scratchpad = combined
            self.context.scratchpad_turn_notes.append(note)

        result = {"status": "ok", "scratchpad_chars": len(combined)}
        if truncated:
            result["note"] = (
                "Scratchpad is full; oldest notes were dropped to make room."
            )
        return json.dumps(result)


_registry = None
try:
    from llm.tools.registry import get_tool_registry

    _registry = get_tool_registry()
    _registry.register_tool(ScratchpadAppendTool())
    _registry.register_tool(SubagentScratchpadAppendTool())
except Exception:  # pragma: no cover - registration best-effort at import
    logger.exception("Failed to register scratchpad tool")
