"""Convert Tool objects to LangChain/OpenAI-compatible tool schemas for bind_tools()."""

from __future__ import annotations

from typing import Any, Dict, List

from llm.tools.interfaces import Tool


def tools_to_langchain_schemas(tools: List[Tool]) -> List[Dict[str, Any]]:
    """Convert our Tool objects to OpenAI-format dicts accepted by LangChain bind_tools().

    Returns a list of dicts: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        })
    return result


__all__ = ["tools_to_langchain_schemas"]
