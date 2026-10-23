from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


StreamEventType = Literal[
    "message_start",
    "token",
    "thinking",
    "message_end",
    "tool_start",
    "tool_end",
    "error",
    "meta",
    # Server-side only: what an edit point / round rewrote in earlier messages
    # (tool results, call arguments, dropped calls), so the consumer can save it
    # and later turns replay the same bytes. Never forwarded to the client.
    "context_edit",
    # Status: the provider layer is backing off before retrying a rate limit /
    # overload ({provider, error_code, wait_seconds, attempt, max_attempts}).
    # Forwarded to the client; the collapser and log accumulator ignore it.
    "provider_retry",
    # A new text message started inside the same model response (OpenAI
    # Responses: a ``commentary`` progress note, then the ``final_answer``).
    # The text streamed so far is a progress note, not part of the answer.
    "text_break",
]


class StreamEvent(BaseModel):
    """Single streaming event emitted during a chat run."""

    event_type: StreamEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    sequence: int
    run_id: str


__all__ = ["StreamEventType", "StreamEvent"]

