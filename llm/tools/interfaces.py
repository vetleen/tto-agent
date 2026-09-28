"""Tool interface for the LLM framework — LangChain BaseTool integration."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from llm.types.context import RunContext


def _tool_validation_error_message(exc: Exception) -> str:
    """Message fed back to the model when tool arguments fail ``args_schema``.

    Wired into every tool via ``handle_validation_error`` below so langchain
    returns this as the tool observation (letting the model self-correct and
    retry) instead of raising — which otherwise surfaces as a recurring,
    non-actionable langchain ``_parse_input`` ValidationError in Sentry (models
    passing a list arg as a JSON string, omitting required fields, or emitting
    malformed JSON: WILFRED-40 / WILFRED-6A). The per-field coercion validators
    on individual tool schemas handle the salvageable cases; this is the
    catch-all for the rest.
    """
    return (
        "Your tool arguments were invalid, so the tool was NOT run. Fix the "
        "arguments and call the tool again: pass list/object arguments as native "
        "JSON values (not strings), use valid JSON, and include all required "
        f"fields. Details: {exc}"
    )


class ReasonBaseModel(BaseModel):
    """Base input schema that adds a ``reason`` field to every tool."""

    reason: str = Field(
        default="",
        description="Brief explanation of why you are calling this tool and what you hope to achieve.",
    )


class ContextAwareTool(BaseTool):
    """Base class for all project tools. Holds RunContext for access control."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    context: RunContext | None = None
    section: str = "chat"

    # Per-audience override of ``section`` for sub-agents. When set (e.g.
    # "chat"), the sub-agent tool resolver uses THIS instead of ``section`` —
    # letting a tool be skills-gated for the main agent yet always-on for
    # sub-agents (used by the document and web tools). None => sub-agents fall
    # back to ``section``. Consumed in core/preferences.py's sub-agent base list.
    subagent_section: str | None = None

    # Which agent kinds may run this tool. "shared" (default) = both the main
    # assistant and sub-agents; "main" = main assistant only (e.g. canvas,
    # loops, skill management, spawning sub-agents); "subagent" = sub-agents
    # only. Orthogonal to ``section`` (which gates activation, not audience).
    # Enforced upstream when assembling the tool list (preferences /
    # resolve_subagent_tools) and defensively in the pipeline's _resolve_tools.
    audience: str = "shared"

    # When the model calls a tool with arguments that fail the args_schema — a
    # list passed as a JSON string, a missing required field, malformed JSON, … —
    # langchain catches the ValidationError and returns this message as the tool
    # observation instead of raising, so the model self-corrects and retries
    # rather than the call erroring out and storming Sentry (recurring langchain
    # _parse_input ValidationErrors: WILFRED-40 / WILFRED-6A).
    handle_validation_error = staticmethod(_tool_validation_error_message)

    # UI display labels for the chat surface. The pipeline ships these to the
    # client in the tool_start/tool_end events as ``display_label`` so the
    # frontend stays a dumb renderer (no per-tool label logic in the template).
    start_label: str = "Working..."  # present tense, shown while the tool runs
    end_label: str = "Done"  # past tense, static fallback when finished

    def set_context(self, ctx: RunContext) -> "ContextAwareTool":
        self.context = ctx
        return self

    # Calls whose serialized arguments are shorter than this are never trimmed
    # (not worth an edit). See trim_args_at_edit_point.
    trim_args_min_chars: int = 2000

    def end_label_for_result(self, result: dict) -> str | None:
        """Dynamic past-tense label derived from the (best-effort parsed) result
        dict. Return None to fall back to ``end_label``. Override in tools whose
        completion label depends on the result (counts, names, status, etc.)."""
        return None

    def trim_args_at_edit_point(self, args: dict, *, later_calls: list) -> dict | None:
        """Return a shrunken copy of an OLD call's arguments, or None to keep them.

        Called only at edit points (the tool loop is already editing history) and
        on between-turn history, for calls outside the recent window whose
        serialized args exceed ``trim_args_min_chars``. Override in tools whose
        big arguments stay reachable another way (a read tool, a later call that
        supersedes them, the scratchpad block) — replace the big fields with a
        short marker saying where the content lives. Must return a JSON object.
        ``later_calls`` are the ToolCalls made after this one, oldest first, so a
        tool can tell whether this call was superseded.
        """
        return None


OMITTED_ARG_PREFIX = "[Omitted from this transcript to save space"


def omitted_arg_marker(what: str, where: str) -> str:
    """Marker replacing a trimmed call argument (see trim_args_at_edit_point).

    Worded so the model reads it as an omission from its own history — NOT as the
    value the call ran with (a bare "[4,200 chars saved]" was read by a live
    model as "I saved a placeholder", and it re-did the work)."""
    return f"{OMITTED_ARG_PREFIX} — this call ran with the full {what}. {where}]"


def is_omitted_arg(value) -> bool:
    return isinstance(value, str) and value.startswith(OMITTED_ARG_PREFIX)


# Backward-compat alias
Tool = ContextAwareTool

__all__ = ["ContextAwareTool", "Tool", "omitted_arg_marker", "is_omitted_arg"]
