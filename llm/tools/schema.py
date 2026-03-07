"""Schema conversion — no longer needed with BaseTool.

LangChain's bind_tools() accepts BaseTool instances directly.
This module is kept for backward compatibility but the function is a no-op pass-through.
"""

from __future__ import annotations

from typing import Any, List


def tools_to_langchain_schemas(tools: List[Any]) -> List[Any]:
    """Pass through — BaseTool instances are accepted directly by bind_tools()."""
    return list(tools)


__all__ = ["tools_to_langchain_schemas"]
