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

    def set_context(self, ctx: RunContext) -> "ContextAwareTool":
        self.context = ctx
        return self


# Backward-compat alias
Tool = ContextAwareTool

__all__ = ["ContextAwareTool", "Tool"]
