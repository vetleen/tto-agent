"""Shared LangChain message conversion used by all providers."""

from __future__ import annotations

from typing import List

from llm.types.messages import Message, ToolCall

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
except Exception:
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore[assignment]


def _normalize_tool_call(tc: object) -> ToolCall:
    """Convert LangChain tool call (dict or object) to our ToolCall."""
    if isinstance(tc, dict):
        return ToolCall(
            id=tc.get("id", ""),
            name=tc["name"],
            arguments=tc.get("args", {}),
        )
    return ToolCall(
        id=getattr(tc, "id", ""),
        name=getattr(tc, "name", ""),
        arguments=getattr(tc, "args", None) or {},
    )


def parse_tool_calls_from_ai_message(ai_message: object) -> list[ToolCall] | None:
    """Extract our ToolCall list from a LangChain AIMessage (or similar)."""
    raw = getattr(ai_message, "tool_calls", None) or []
    if not raw:
        return None
    return [_normalize_tool_call(tc) for tc in raw]


def to_langchain_messages(messages: List[Message]):
    """Convert internal Message objects to LangChain message types.

    Role mapping:
        system    → SystemMessage
        assistant → AIMessage (with optional tool_calls)
        user      → HumanMessage
        tool      → ToolMessage if tool_call_id is set, else HumanMessage (backward compat)
    """
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "assistant":
            tool_calls_lc = None
            if m.tool_calls:
                tool_calls_lc = [
                    {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    for tc in m.tool_calls
                ]
            lc_messages.append(AIMessage(content=m.content, tool_calls=tool_calls_lc or []))
        elif m.role == "tool" and m.tool_call_id:
            lc_messages.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id))
        else:
            # user, or tool without tool_call_id (backward compat)
            lc_messages.append(HumanMessage(content=m.content))
    return lc_messages
