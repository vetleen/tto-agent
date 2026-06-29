"""Tool interface for the LLM framework — LangChain BaseTool integration."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from llm.types.context import RunContext


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
