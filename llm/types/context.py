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
    data_room_ids: list[int] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deadline_seconds: Optional[int] = None
    # Which agent kind this run is: "main" (the orchestrator) or "subagent".
    # Lets the pipeline defensively drop tools whose audience excludes this kind.
    agent_kind: str = "main"
    # Image assets a tool asked to surface to the model this turn. The chat
    # pipeline drains these into a user message — native image blocks when the
    # model supports vision, else their text descriptions. Each item is a dict:
    # {"asset_id", "b64", "media_type", "description"}.
    pending_image_assets: list = Field(default_factory=list)
    # Skill activation the agent triggered mid-turn (via chat_skill_attach).
    # The chat pipeline drains these each tool-loop iteration so a skill the
    # agent attaches takes effect on the very next step of the SAME turn — not
    # the next user turn. Mirrors the pending_image_assets drain pattern.
    #   added_tool_names           — tool names to union into the live tool set.
    #   pending_skill_instructions — rendered instruction blocks to inject.
    added_tool_names: list = Field(default_factory=list)
    pending_skill_instructions: list = Field(default_factory=list)
    # slug -> org-filtered tool_names, stashed by the consumer from
    # prefs.allowed_skills so chat_skill_attach can resolve a newly-attached
    # skill's tools without re-deriving org tool-toggle filtering.
    skill_tool_map: dict = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        user_id: Any | None = None,
        conversation_id: Any | None = None,
        deadline_seconds: int | None = None,
        data_room_ids: list[int] | None = None,
    ) -> "RunContext":
        return cls(
            user_id=str(user_id) if user_id is not None else None,
            conversation_id=str(conversation_id) if conversation_id is not None else None,
            deadline_seconds=deadline_seconds,
            data_room_ids=data_room_ids or [],
        )


__all__ = ["RunContext"]
