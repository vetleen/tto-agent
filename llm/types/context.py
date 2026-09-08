from __future__ import annotations

import threading

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

# Per-run cap on native-asset bytes, measured on base64 length (that is what
# actually occupies memory in the message history and the provider payload).
# Cumulative for the whole run — it deliberately does NOT reset when the
# pipeline drains pending_native_assets each tool-loop iteration.
NATIVE_ASSET_BUDGET_B64_CHARS = 20 * 1024 * 1024


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
    # Files a tool asked to surface to the model this turn (document_view_native
    # queues images/pdf). The chat pipeline drains these into a user message as
    # native content blocks when the model supports the modality, else a text
    # fallback. Each item is a dict keyed by "kind" ("image" default, or "pdf"):
    #   image: {"asset_id", "b64", "media_type", "description"}
    #   pdf:   {"kind": "pdf", "b64", "filename", "description", "extracted_text"}
    pending_native_assets: list = Field(default_factory=list)
    # Skill activation the agent triggered mid-turn (via chat_skill_attach).
    # The chat pipeline drains these each tool-loop iteration so a skill the
    # agent attaches takes effect on the very next step of the SAME turn — not
    # the next user turn. Mirrors the pending_native_assets drain pattern.
    #   added_tool_names           — tool names to union into the live tool set.
    #   pending_skill_instructions — rendered instruction blocks to inject.
    added_tool_names: list = Field(default_factory=list)
    pending_skill_instructions: list = Field(default_factory=list)
    # slug -> org-filtered tool_names, stashed by the consumer from
    # prefs.allowed_skills so chat_skill_attach can resolve a newly-attached
    # skill's tools without re-deriving org tool-toggle filtering.
    skill_tool_map: dict = Field(default_factory=dict)
    # Web image candidates surfaced by web_fetch(include_images=True), keyed by a
    # run-monotonic handle ("img-1", "img-2", …). web_image_view resolves a
    # handle to its source URL. Each value: {"url", "page_url", "filename", "alt"}.
    # Per-run only (a fresh RunContext per turn/agent) — handles from one run are
    # meaningless in another, so a sub-agent must return the durable [[image:…]]
    # token it mints, never an img-N handle.
    web_image_manifest: dict = Field(default_factory=dict)
    # Handles are allocated under a lock because tools run concurrently
    # (ThreadPoolExecutor): a plain len()+1 would race and collide.
    _web_image_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _web_image_next: int = PrivateAttr(default=1)
    # Native-asset budget bookkeeping — locked for the same reason as above.
    _native_asset_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _native_asset_b64_used: int = PrivateAttr(default=0)

    def try_add_native_asset(self, item: dict) -> bool:
        """Queue a native asset if the run's byte budget allows it.

        Atomically reserves ``len(item["b64"])`` against the per-run budget and
        appends to ``pending_native_assets``; returns False (and appends
        nothing) once the budget is exhausted. Every tool that surfaces
        images/PDFs to the model MUST go through this instead of appending
        directly, so one run can never hold unbounded base64 in history.
        """
        size = len(item.get("b64") or "")
        with self._native_asset_lock:
            if self._native_asset_b64_used + size > NATIVE_ASSET_BUDGET_B64_CHARS:
                return False
            self._native_asset_b64_used += size
            self.pending_native_assets.append(item)
            return True

    def native_asset_budget_remaining(self) -> int:
        """Base64 chars still available in this run's native-asset budget."""
        with self._native_asset_lock:
            return NATIVE_ASSET_BUDGET_B64_CHARS - self._native_asset_b64_used

    def allocate_web_image_handle(self, entry: dict) -> str:
        """Register a web image candidate under a fresh monotonic handle and
        return it (e.g. ``"img-7"``). Thread-safe."""
        with self._web_image_lock:
            handle = f"img-{self._web_image_next}"
            self._web_image_next += 1
            self.web_image_manifest[handle] = entry
            return handle

    def remaining_seconds(self) -> Optional[float]:
        """Seconds left before this run's deadline, or None if no deadline is set.

        Negative once the deadline has passed. Lets tool retry/backoff loops —
        which have no cancel signal — avoid sleeping past the run's deadline.
        """
        if self.deadline_seconds is None:
            return None
        elapsed = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        return self.deadline_seconds - elapsed

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
