from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .context import RunContext
from .messages import Message


class ChatRequest(BaseModel):
    """Normalized chat request for all pipelines and providers."""

    messages: List[Message]
    stream: bool = False
    model: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    tools: Optional[List[str]] = None
    context: Optional[RunContext] = None


__all__ = ["ChatRequest"]

