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


def signed_text_blocks(blocks: list | None) -> list[dict] | None:
    """A reply's text blocks when the provider signed one, else None.

    Gemini attaches a thought signature to a text part (``extras.signature``,
    sometimes on an empty-text part). Google recommends sending it back in that
    same part on later turns for reasoning quality; it is not validated.
    """
    text_blocks = [
        b for b in blocks or []
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    if any(isinstance(b.get("extras"), dict) and b["extras"].get("signature") for b in text_blocks):
        return text_blocks
    return None


def parse_tool_calls_from_ai_message(ai_message: object) -> list[ToolCall] | None:
    """Extract our ToolCall list from a LangChain AIMessage (or similar)."""
    raw = getattr(ai_message, "tool_calls", None) or []
    if not raw:
        return None
    return [_normalize_tool_call(tc) for tc in raw]


# Message.metadata flag on the user message that carries this turn's per-turn
# context preamble (set by the chat consumer). The preamble is gone from that
# message on the next turn, so the request's tail is never a reusable cache
# entry; the history breakpoints are placed relative to it instead.
TURN_ANCHOR_KEY = "wf_turn_anchor"
# 1h, not the default 5m: users often pause 5-60 min between turns, and a 5m
# entry is gone by then (prod 2026-10: ~20% hit after such a gap vs ~70% under
# 5 min). A 1h write costs 2x input instead of 1.25x, but only the newest
# exchange is written each turn while the whole history is re-read. Anthropic
# requires longer TTLs before shorter ones; the automatic end breakpoint (5m)
# always comes after these.
_HISTORY_BREAKPOINT = {"type": "ephemeral", "ttl": "1h"}


def _with_cache_breakpoint(msg):
    """``msg`` with a cache_control breakpoint on its last text block, or None
    when it has no text block that can carry one (tool results and
    tool-call-only turns are skipped: the caller walks further back)."""
    if isinstance(msg, ToolMessage):
        return None
    content = msg.content
    if isinstance(content, str):
        if not content.strip():
            return None
        blocks = [{"type": "text", "text": content, "cache_control": _HISTORY_BREAKPOINT}]
        return msg.model_copy(update={"content": blocks})
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            blk = content[i]
            if isinstance(blk, dict) and blk.get("type") == "text" and str(blk.get("text") or "").strip():
                new = list(content)
                new[i] = {**blk, "cache_control": _HISTORY_BREAKPOINT}
                return msg.model_copy(update={"content": new})
    return None


def _mark_before(lc_messages: list, index: int) -> None:
    """Put a history breakpoint on the nearest message before ``index`` that
    can carry one (deterministic, so the turn that writes an entry and the
    turn that reads it pick the same block)."""
    for i in range(index - 1, 0, -1):
        if isinstance(lc_messages[i], SystemMessage):
            return
        marked = _with_cache_breakpoint(lc_messages[i])
        if marked is not None:
            lc_messages[i] = marked
            return


def _apply_history_breakpoints(lc_messages: list, messages: List[Message]) -> list:
    """Two explicit breakpoints so consecutive turns share cached history.

    The automatic breakpoint sits at the end of each request, which includes
    the turn's user message with its per-turn context preamble; the next turn
    sends that message without the preamble, so nothing written there is ever
    read again. Instead:

    - *write*: the end of the stable history, just before this turn's anchor
      message — the next turn sends exactly this prefix again;
    - *read*: the same spot one turn back (before the previous turn's anchor,
      found as the user message that opened the previous assistant turn) — an
      exact hit even when the previous turn's tool loop is longer than
      Anthropic's 20-block lookback.

    With the system prompt's 1h breakpoint and the automatic one at the end
    that makes four, Anthropic's maximum. Requests without an anchor (sub-
    agents, one-off calls) are left as they were.
    """
    anchor = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if messages[i].role == "user" and (messages[i].metadata or {}).get(TURN_ANCHOR_KEY)),
        None,
    )
    if anchor is None:
        return lc_messages
    out = list(lc_messages)
    # The previous turn: the last assistant message before the anchor, then the
    # user message that opened it.
    prev_anchor = None
    i = anchor - 1
    while i > 0 and messages[i].role != "assistant":
        i -= 1
    while i > 0 and messages[i].role != "user":
        i -= 1
    if i > 0 and messages[i].role == "user":
        prev_anchor = i
    _mark_before(out, anchor)
    if prev_anchor is not None:
        _mark_before(out, prev_anchor)
    return out


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


def _sanitize_content(content):
    """Drop internal ``_wf_*`` markers from multimodal content blocks so they
    never reach a provider. Defensive backstop — the send-time native-limit
    enforcer normally strips them before conversion."""
    if not isinstance(content, list):
        return content
    cleaned = []
    changed = False
    for blk in content:
        if isinstance(blk, dict) and any(k.startswith("_wf") for k in blk):
            cleaned.append({k: v for k, v in blk.items() if not k.startswith("_wf")})
            changed = True
        else:
            cleaned.append(blk)
    return cleaned if changed else content


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
            # Only the FIRST system message is the static prompt that carries the
            # 1h breakpoint. Later ones (the conversation summary) must not
            # replace it: langchain-anthropic merges consecutive system messages,
            # so keeping the last one here sent [summary, summary] and no prompt.
            if system_content is None:
                system_content = m.content
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "assistant":
            tool_calls_lc = None
            if m.tool_calls:
                tool_calls_lc = [
                    {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    for tc in m.tool_calls
                ]
            # Echo the provider's native reasoning blocks (thinking /
            # redacted_thinking, with signatures) when captured, instead of a
            # plain-text rebuild. Anthropic 400s on a thinking+tool-use turn whose
            # thinking blocks were dropped; langchain-anthropic re-appends the
            # tool_use blocks from tool_calls, so [thinking, text] + tool_calls
            # serializes to [thinking, text, tool_use] (thinking first, as
            # required). Gemini's blocks carry thought signatures on text parts.
            # Other providers send plain-string content.
            content_blocks = (m.metadata or {}).get("content_blocks")
            ai_content = content_blocks if content_blocks else m.content
            # Restore provider replay state (e.g. Gemini function-call thought
            # signatures) that our Message abstraction otherwise drops.
            extra_kwargs = (m.metadata or {}).get("additional_kwargs") or {}
            lc_messages.append(AIMessage(
                content=ai_content,
                tool_calls=tool_calls_lc or [],
                additional_kwargs=extra_kwargs,
            ))
        elif m.role == "tool" and m.tool_call_id:
            lc_messages.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id))
        else:
            # user, or tool without tool_call_id (backward compat)
            lc_messages.append(HumanMessage(content=_sanitize_content(m.content)))

    if provider == "anthropic":
        lc_messages = _apply_history_breakpoints(lc_messages, messages)
        lc_messages = _apply_anthropic_cache_control(lc_messages, system_content)

    return lc_messages
