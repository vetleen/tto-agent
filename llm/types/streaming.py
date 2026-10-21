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
]


class StreamEvent(BaseModel):
    """Single streaming event emitted during a chat run."""

    event_type: StreamEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    sequence: int
    run_id: str


__all__ = ["StreamEventType", "StreamEvent"]

