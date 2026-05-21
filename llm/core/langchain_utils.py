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
    """Add an explicit cache_control breakpoint to the system message.

    The system message is static and never changes within a conversation,
    so it gets a long-lived 1h cache. Conversation history caching is
    handled automatically by langchain-anthropic's cache_control kwarg
    (bound in AnthropicChatModel._get_streaming_client), which uses
    Anthropic's lookback window for incremental prefix caching.
    """
    if not lc_messages or not isinstance(lc_messages[0], SystemMessage) or system_content is None:
        return lc_messages
    result = [SystemMessage(content=[{
        "type": "text",
        "text": system_content,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }])]
    result.extend(lc_messages[1:])
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
