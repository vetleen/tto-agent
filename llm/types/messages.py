from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """Generic chat message used across pipelines and providers."""

    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None  # assistant messages requesting tools
    tool_call_id: Optional[str] = None  # tool result messages
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Represents a tool invocation requested or executed by the model."""

    id: str  # correlates call with result
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["Role", "Message", "ToolCall"]

