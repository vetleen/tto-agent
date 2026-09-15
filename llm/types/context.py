from __future__ import annotations

import threading

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr

# Native-asset budget, measured on base64 length (that is what actually occupies
# memory in the message history and the provider payload). Three pathways surface
# PDF/image assets to the model — user chat attachments, data-room documents, and
# skill resources — and they share one pool. The pool is the ceiling on total
# native base64 in a single outgoing request; the send-time enforcer
# (llm/core/providers/base.py) prunes to min(pool, provider_ceiling) with skill
# priority. This add-time budget is a fair-share + memory-bail guard only.
#
# Fallbacks used when Django settings can't be read; the live values come from
# settings so ops can tune them without a deploy.
_DEFAULT_NATIVE_ASSET_BUDGET_B64_BYTES = 50 * 1024 * 1024
_DEFAULT_NATIVE_ASSET_SKILL_FRACTION = 0.5

# Pathway tags. Skill assets are prioritized (evicted last); everything else is
# non-skill and evicted oldest-first when a request overflows.
PATHWAY_SKILL = "skill"
PATHWAY_DATAROOM = "dataroom"
PATHWAY_ATTACHMENT = "attachment"


def native_asset_pool_bytes() -> int:
    """Ceiling (base64 chars) on total native-asset bytes in one request."""
    try:
        from django.conf import settings

        return int(getattr(
            settings, "NATIVE_ASSET_BUDGET_B64_BYTES",
            _DEFAULT_NATIVE_ASSET_BUDGET_B64_BYTES,
        ))
    except Exception:
        return _DEFAULT_NATIVE_ASSET_BUDGET_B64_BYTES


def native_asset_skill_fraction() -> float:
    """Fraction of the pool a skill-pathway asset may occupy (hard cap)."""
    try:
        from django.conf import settings

        return float(getattr(
            settings, "NATIVE_ASSET_SKILL_FRACTION",
            _DEFAULT_NATIVE_ASSET_SKILL_FRACTION,
        ))
    except Exception:
        return _DEFAULT_NATIVE_ASSET_SKILL_FRACTION


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
    # The org/user context aim for this run. Read by chat_skill_attach so a
    # mid-turn skill attach respects the same aim-relative skill budget as the UI
    # (agent_skills.resources.attach_token_budget). None → the fixed budget.
    max_context_tokens: Optional[int] = None
    # Sub-agent private scratchpad (subagent_scratchpad_append). Held in memory so
    # the tool loop can re-inject it into the system prompt every iteration without
    # a DB read; the durable copy lives on SubAgentRun.scratchpad. Empty for the
    # main agent (which uses the thread-scoped scratchpad instead).
    scratchpad: str = ""
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
    # _native_asset_b64_used is the running total; _b64_used_by_pathway breaks it
    # down per pathway (skill/dataroom/attachment) to enforce the skill sub-cap.
    _native_asset_lock: Any = PrivateAttr(default_factory=threading.Lock)
    _native_asset_b64_used: int = PrivateAttr(default=0)
    _b64_used_by_pathway: dict = PrivateAttr(default_factory=dict)
    # The acting User instance, resolved once and reused for the whole run so the
    # per-instance memoization on accounts.models.get_membership /
    # get_user_preferences_dict holds across every tool call (otherwise each tool
    # loads a fresh User and re-queries UserSettings + Membership). Seeded at run
    # setup on a single thread before tools spawn; tools only read it. Left None on
    # the main pipeline (which does not seed it) so its behaviour is unchanged.
    _cached_user: Any = PrivateAttr(default=None)
    # Per-turn observability counters the pipeline accumulates across the tool
    # loop; the logger surfaces them onto LLMCallLog (tool_call_count,
    # prune_count, tool_result_tokens) so we can reason about tool-loop cost and
    # how often mid-turn pruning fires over time — the data behind a future
    # tool-result budget. Locked because tools execute concurrently.
    _stats_lock: Any = PrivateAttr(default_factory=threading.Lock)
    observability: dict = Field(default_factory=dict)

    def bump_stat(self, key: str, amount: int = 1) -> None:
        """Add ``amount`` to a per-turn observability counter (thread-safe)."""
        with self._stats_lock:
            self.observability[key] = self.observability.get(key, 0) + amount

    def reserve_native_asset(
        self, size_b64: int, pathway: str = PATHWAY_DATAROOM
    ) -> bool:
        """Atomically reserve ``size_b64`` base64 chars for ``pathway``.

        Skill-pathway assets are hard-capped at ``skill_fraction * pool``; every
        other (non-skill) pathway is collectively bounded by the pool. This is a
        fair-share + memory-bail guard — the true per-request ceiling (and skill
        priority) is enforced at send time by pruning. Returns False without
        reserving when the pathway's allowance is exhausted.
        """
        pool = native_asset_pool_bytes()
        with self._native_asset_lock:
            used = self._b64_used_by_pathway.get(pathway, 0)
            if pathway == PATHWAY_SKILL:
                cap = int(pool * native_asset_skill_fraction())
                if used + size_b64 > cap:
                    return False
            else:
                nonskill_used = sum(
                    v for k, v in self._b64_used_by_pathway.items()
                    if k != PATHWAY_SKILL
                )
                if nonskill_used + size_b64 > pool:
                    return False
            self._b64_used_by_pathway[pathway] = used + size_b64
            self._native_asset_b64_used += size_b64
            return True

    def try_add_native_asset(
        self, item: dict, pathway: str = PATHWAY_DATAROOM
    ) -> bool:
        """Queue a native asset if the pathway's byte budget allows it.

        Reserves ``len(item["b64"])`` via :meth:`reserve_native_asset`, tags the
        item with its ``pathway`` (so the drain can classify the emitted content
        block for send-time priority pruning), and appends to
        ``pending_native_assets``. Every tool that surfaces images/PDFs to the
        model MUST go through this instead of appending directly.
        """
        size = len(item.get("b64") or "")
        if not self.reserve_native_asset(size, pathway):
            return False
        item["_pathway"] = pathway
        self.pending_native_assets.append(item)
        return True

    def native_asset_budget_remaining(self, pathway: str = PATHWAY_DATAROOM) -> int:
        """Base64 chars still available to ``pathway`` in this run.

        Skill → its 50% reserve; any non-skill pathway → the pool minus what
        non-skill pathways have already taken.
        """
        pool = native_asset_pool_bytes()
        with self._native_asset_lock:
            if pathway == PATHWAY_SKILL:
                cap = int(pool * native_asset_skill_fraction())
                return max(0, cap - self._b64_used_by_pathway.get(PATHWAY_SKILL, 0))
            nonskill_used = sum(
                v for k, v in self._b64_used_by_pathway.items()
                if k != PATHWAY_SKILL
            )
            return max(0, pool - nonskill_used)

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


__all__ = [
    "RunContext",
    "PATHWAY_SKILL",
    "PATHWAY_DATAROOM",
    "PATHWAY_ATTACHMENT",
    "native_asset_pool_bytes",
    "native_asset_skill_fraction",
]
