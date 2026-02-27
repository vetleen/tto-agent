"""Built-in tools for testing and demos."""

from __future__ import annotations

from typing import Any, Dict

from llm.types.context import RunContext

from .interfaces import Tool
from .registry import get_tool_registry


class AddNumberTool:
    """Tool that adds two numbers. Used for testing tool wiring."""

    name = "add_number"
    description = "Add two numbers together and return the sum."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    }

    def run(self, args: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        a = args.get("a")
        b = args.get("b")
        if a is None or b is None:
            raise ValueError("add_number requires 'a' and 'b' arguments")
        try:
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
        except (TypeError, ValueError) as e:
            raise ValueError(f"add_number expects numeric a and b: {e}") from e
        return {"result": a_num + b_num}


# Register on import so pipelines can resolve by name.
_registry = get_tool_registry()
_registry.register_tool(AddNumberTool())


__all__ = ["AddNumberTool"]
