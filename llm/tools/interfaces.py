"""Tool interface for the LLM framework — LangChain BaseTool integration."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from llm.types.context import RunContext


class ContextAwareTool(BaseTool):
    """Base class for all project tools. Holds RunContext for access control."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    context: RunContext | None = None

    def set_context(self, ctx: RunContext) -> "ContextAwareTool":
        self.context = ctx
        return self


# Backward-compat alias
Tool = ContextAwareTool

__all__ = ["ContextAwareTool", "Tool"]
