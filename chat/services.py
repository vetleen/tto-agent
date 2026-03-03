"""Chat services — summarisation and related helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chat.models import ChatMessage

logger = logging.getLogger(__name__)

SUMMARY_TARGET_TOKENS = 2_000


async def generate_summary(
    messages: list[ChatMessage],
    existing_summary: str = "",
    *,
    user_id: int,
    conversation_id,
) -> str:
    """Summarise *messages* into a concise rolling summary.

    When *existing_summary* is provided it is folded into the new summary
    so that the LLM produces a single coherent summary covering all prior
    history.

    Uses ``openai/gpt-5-mini`` (same cheap model used for title generation).
    """
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    parts: list[str] = []
    if existing_summary:
        parts.append(
            f"Previous conversation summary:\n{existing_summary}\n"
        )
    parts.append("Messages to summarise:")
    for msg in messages:
        parts.append(f"[{msg.role}]: {msg.content}")

    user_prompt = "\n".join(parts)

    system_prompt = (
        "Produce a concise summary for an LLM chatbot"
        f"of its conversation with a user. Target ~{SUMMARY_TARGET_TOKENS} tokens.  "
        "Preserve key facts, decisions, important points as well as"
        "any context the assistant would need to continue the conversation "
        "coherently. Do NOT include any preamble — output only the summary."
    )

    context = RunContext.create(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        model="openai/gpt-5-mini",
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    response = await service.arun("simple_chat", request)
    return response.message.content.strip()
