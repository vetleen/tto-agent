from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


StreamEventType = Literal[
    "message_start",
    "token",
    "message_end",
    "tool_start",
    "tool_end",
    "error",
    "meta",
]


class StreamEvent(BaseModel):
    """Single streaming event emitted during a chat run."""

    event_type: StreamEventType
    data: Dict[str, Any] = Field(default_factory=dict)
    sequence: int
    run_id: str


__all__ = ["StreamEventType", "StreamEvent"]

