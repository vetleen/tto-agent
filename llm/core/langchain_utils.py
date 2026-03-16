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


def _apply_anthropic_cache_control(lc_messages: list, system_content: str | None) -> list:
    """Add cache_control breakpoints for Anthropic prompt caching.

    Places cache_control at two points:
    1. System message -- always cached (stable across turns)
    2. Second-to-last message -- caches the entire conversation prefix

    This means each new turn only pays full price for the latest user message;
    everything before hits the cache. Requires 1024+ tokens to activate --
    short conversations simply won't cache (no harm).
    """
    result = []
    for i, msg in enumerate(lc_messages):
        if i == 0 and isinstance(msg, SystemMessage) and system_content is not None:
            # Convert system message to content-block format with cache_control
            result.append(SystemMessage(content=[{
                "type": "text",
                "text": system_content,
                "cache_control": {"type": "ephemeral"},
            }]))
        elif len(lc_messages) >= 3 and i == len(lc_messages) - 2:
            # Second-to-last message: add cache_control
            msg_copy = msg.model_copy()
            additional = dict(getattr(msg_copy, "additional_kwargs", {}) or {})
            additional["cache_control"] = {"type": "ephemeral"}
            msg_copy.additional_kwargs = additional
            result.append(msg_copy)
        else:
            result.append(msg)
    return result


def to_langchain_messages(messages: List[Message], *, provider: str | None = None):
    """Convert internal Message objects to LangChain message types.

    Role mapping:
        system    -> SystemMessage
        assistant -> AIMessage (with optional tool_calls)
        user      -> HumanMessage
        tool      -> ToolMessage if tool_call_id is set, else HumanMessage (backward compat)

    When ``provider="anthropic"``, cache_control breakpoints are added to the
    system message and second-to-last message for Anthropic prompt caching.
    """
    lc_messages = []
    system_content = None
    for m in messages:
        if m.role == "system":
            system_content = m.content
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

    if provider == "anthropic":
        lc_messages = _apply_anthropic_cache_control(lc_messages, system_content)

    return lc_messages
