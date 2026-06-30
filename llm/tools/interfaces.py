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

    def end_label_for_result(self, result: dict) -> str | None:
        """Dynamic past-tense label derived from the (best-effort parsed) result
        dict. Return None to fall back to ``end_label``. Override in tools whose
        completion label depends on the result (counts, names, status, etc.)."""
        return None


# Backward-compat alias
Tool = ContextAwareTool

__all__ = ["ContextAwareTool", "Tool"]
