"""Shared LangChain message conversion used by all providers."""

from __future__ import annotations

from typing import List

from llm.types.messages import Message

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except Exception:
    AIMessage = HumanMessage = SystemMessage = None  # type: ignore[assignment]


def to_langchain_messages(messages: List[Message]):
    """Convert internal Message objects to LangChain message types.

    Role mapping:
        system    → SystemMessage
        assistant → AIMessage
        user / *  → HumanMessage  (tool and unknown roles treated as human for v1)
    """
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        else:
            # Treat "user", "tool", and any other roles as human messages for v1
            lc_messages.append(HumanMessage(content=m.content))
    return lc_messages
