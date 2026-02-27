"""Tool interface for the LLM framework."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from llm.types.context import RunContext


@runtime_checkable
class Tool(Protocol):
    """Protocol for a callable tool that can be invoked by pipelines."""

    name: str

    def run(self, args: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        """Execute the tool with the given arguments and run context."""
        ...


__all__ = ["Tool"]
