"""Registry for tools by name."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from llm.tools.interfaces import Tool


@dataclass
class ToolRegistry:
    """Holds tools by name for lookup by pipelines."""

    _tools: Dict[str, Tool] = field(default_factory=dict)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool by its name."""
        if not tool.name:
            raise ValueError("Tool name must be non-empty")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Return the tool with the given name, or None if not registered."""
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, Tool]:
        """Return a copy of the name -> tool mapping."""
        return dict(self._tools)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()


_global_registry: Optional[ToolRegistry] = None
_global_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Return the process-wide ToolRegistry singleton (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = ToolRegistry()
    return _global_registry


__all__ = ["ToolRegistry", "get_tool_registry"]
