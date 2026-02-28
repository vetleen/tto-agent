from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .messages import Message


class Usage(BaseModel):
    """Token and cost accounting for a single LLM call."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class ChatResponse(BaseModel):
    """Normalized response from a chat model or pipeline."""

    message: Message
    model: str
    usage: Optional[Usage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["Usage", "ChatResponse"]

