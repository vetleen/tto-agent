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

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)


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
    main-agent analogue). Stored on the ``SubAgentRun`` and re-injected into the
    system prompt every tool-loop iteration, so notes survive mid-run pruning."""

    name: str = "subagent_scratchpad_append"
    audience: str = "subagent"
    section: str = "chat"  # always-on for sub-agents (see core.preferences)
    start_label: str = "Making a note..."
    end_label: str = "Made a note"
    description: str = (
        "Append a note to your private, persistent scratchpad. The scratchpad is "
        "shown back to you every step and survives context pruning, so use it to "
        "retain findings before large tool results are cleared. It is private "
        "working memory — NOT part of your deliverable (build that in your working "
        "canvas) and the user never sees it."
    )
    args_schema: type[BaseModel] = SubagentScratchpadAppendInput

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
        # Update the in-memory copy the tool loop re-injects each iteration, so the
        # append is visible next round without a DB read (see simple_chat pipeline).
        if self.context is not None:
            self.context.scratchpad = combined

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
