"""Completion label + error flag for a finished tool call.

Shared by the live pipeline (``tool_end`` events) and the chat history view, so a
reloaded thread shows exactly the labels the user saw while it streamed.
"""

from __future__ import annotations

import json
from typing import Optional

from llm.tools.interfaces import ContextAwareTool

FALLBACK_END_LABEL = "Done"


def parse_tool_result(result_str) -> dict | None:
    """Best-effort parse of a tool result into a dict.

    Returns None for non-JSON or non-dict results (e.g. tools that return
    markdown), in which case labels fall back to the tool's static ``end_label``.
    """
    try:
        value = json.loads(result_str)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def tool_end_display(tool: Optional[ContextAwareTool], result_str) -> tuple[str, bool]:
    """Return ``(display_label, is_error)`` for a finished call of ``tool``.

    ``tool`` may be None (unknown or retired tool) → the generic "Done" label.
    Tools report failure as ``{"status": "error", ...}``.
    """
    parsed = parse_tool_result(result_str)
    label = FALLBACK_END_LABEL
    if tool:
        dynamic = tool.end_label_for_result(parsed) if parsed is not None else None
        label = dynamic or tool.end_label
    is_error = bool(parsed) and parsed.get("status") == "error"
    return label, is_error
