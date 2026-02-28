from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    """Per-run context for tracing, attribution, and timeouts."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deadline_seconds: Optional[int] = None

    @classmethod
    def create(
        cls,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
        deadline_seconds: int | None = None,
    ) -> "RunContext":
        return cls(
            user_id=str(user_id) if user_id is not None else None,
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            deadline_seconds=deadline_seconds,
        )


__all__ = ["RunContext"]

