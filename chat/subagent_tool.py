"""Sub-agent tools: create and check sub-agent runs."""

from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


def _is_waiting(run_id) -> bool:
    """True while the run is still queued for an execution slot (PENDING, never dispatched)."""
    from chat.models import SubAgentRun

    return SubAgentRun.objects.filter(
        pk=run_id, status=SubAgentRun.Status.PENDING, dispatched_at__isnull=True,
    ).exists()


# --- Input schemas ---

class CreateSubagentInput(ReasonBaseModel):
    prompt: str = Field(description="The task description for the sub-agent.")
    type: str | None = Field(
        default=None,
        description=(
            "Optional specialization for the sub-agent — pass the slug of one of "
            "the specializations listed in your instructions (under 'Sub-agent "
            "specializations'). Gives the sub-agent extra instructions and tools "
            "for that role. Omit for a general-purpose sub-agent."
        ),
    )
    model_tier: str = Field(
        default="mid",
        description='Model tier: "mid" (default) for research, "top" for deep analysis.',
    )
    timeout: int = Field(
        default=0,
        description="Seconds to wait for the result (0-540). 0=background, 30-60=quick tasks, 120=research.",
    )



# --- Tools ---

class CreateSubagentTool(ContextAwareTool):
    """Create a sub-agent to handle a delegated task."""

    name: str = "chat_subagent_create"
    audience: str = "main"
    start_label: str = "Running sub-agent..."
    end_label: str = "Sub-agent dispatched"

    def end_label_for_result(self, result: dict) -> str | None:
        return {
            "queued": "Sub-agent queued",
            "started": "Sub-agent dispatched",
            "completed": "Sub-agent finished",
            "timeout": "Sub-agent dispatched",
            "error": "Sub-agent failed",
        }.get(result.get("status"))

    description: str = (
        "Delegate a task to an independent sub-agent that runs with its own context and tools. "
        "Use for tasks requiring extensive research, parallel analysis, or focused work. "
        "Set timeout=0 for background, 30-60 for quick tasks, 120 for research."
    )
    args_schema: type[BaseModel] = CreateSubagentInput

    def _run(
        self,
        prompt: str,
        model_tier: str = "mid",
        timeout: int = 0,
        **kwargs,
    ) -> str:
        from django.contrib.auth import get_user_model

        from chat.models import SubAgentRun
        from chat.subagent_limits import (
            create_subagent_run_if_allowed,
            dispatch_pending_subagents,
            get_queue_depth,
        )

        context = self.context
        if not context or not context.user_id:
            return json.dumps({"status": "error", "message": "No user context available."})

        thread_id = context.conversation_id if context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # Validate model_tier
        if model_tier not in ("mid", "top"):
            model_tier = "mid"

        # Clamp timeout to [0, 540]
        timeout = max(0, min(timeout, 540))

        # Load user
        User = get_user_model()
        try:
            user = User.objects.get(pk=context.user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        data_room_ids = context.data_room_ids if context else []

        # Enforce sequential subagent policy if parallel is disabled
        from core.preferences import get_preferences
        prefs = get_preferences(user)

        # Resolve an optional specialization ("type"). Validate against the
        # user's available specializations and feed an actionable error back to
        # the model (it can retry without a type, or with a valid one) rather
        # than silently ignoring an unknown slug.
        specialization = (kwargs.get("type") or "").strip()
        if specialization:
            valid_slugs = [s["slug"] for s in prefs.allowed_specializations]
            if specialization not in valid_slugs:
                available = ", ".join(valid_slugs) if valid_slugs else "(none available)"
                return json.dumps({
                    "status": "error",
                    "message": (
                        f"Unknown sub-agent type '{specialization}'. "
                        f"Available types: {available}. "
                        "Omit 'type' to spawn a general-purpose sub-agent."
                    ),
                })

        if not prefs.parallel_subagents:
            active = SubAgentRun.objects.filter(
                thread_id=thread_id,
                status__in=[SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING],
            ).exists()
            if active:
                return json.dumps({"status": "error", "message": "Sub-agents must run one at a time. Wait for the current one to complete."})

        # Atomically check limits and create the run record.
        # The service resolves model_used and tool_names when it executes.
        run, err_msg = create_subagent_run_if_allowed(
            user,
            thread_id=thread_id,
            prompt=prompt,
            model_tier=model_tier,
            timeout=timeout,
            data_room_ids=data_room_ids,
            skill_slug=specialization,
        )
        if run is None:
            return json.dumps({"status": "error", "message": err_msg})

        # Hand it to Celery now if an execution slot is free; otherwise it waits
        # in line and is dispatched when a running sub-agent finishes. A
        # dispatcher error is not a spawn error: the row exists and the sweeper
        # backstop will dispatch it.
        try:
            dispatch_pending_subagents()
        except Exception:
            logger.exception("Dispatch after creating sub-agent run %s failed", run.id)
        # Read the outcome from the row, not from this call's return value:
        # sibling chat_subagent_create calls in the same turn run in parallel
        # threads, and whichever dispatcher runs first hands out every waiting
        # run — including ours — leaving our own call nothing to return.
        started = not _is_waiting(run.id)

        if timeout == 0:
            if started:
                return json.dumps({
                    "status": "started",
                    "run_id": str(run.id),
                    "message": f"Sub-agent started (model: {model_tier}). Its result will appear in the conversation automatically.",
                })
            queue = get_queue_depth()
            ahead = SubAgentRun.objects.filter(
                status=SubAgentRun.Status.PENDING,
                dispatched_at__isnull=True,
                created_at__lt=run.created_at,
            ).count()
            return json.dumps({
                "status": "queued",
                "run_id": str(run.id),
                "queue_position": ahead + 1,
                "ahead_in_queue": ahead,
                "currently_running": queue["running"],
                "worker_slots": queue["worker_slots"],
                "message": (
                    f"Sub-agent queued (position {ahead + 1}): all {queue['worker_slots']} "
                    f"execution slots are busy ({queue['running']} running, {ahead} waiting ahead of it). "
                    "It starts automatically when a slot frees up and its result will appear "
                    "in the conversation — do not create it again."
                ),
            })

        # Poll for result until timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            run.refresh_from_db()
            if run.status == SubAgentRun.Status.COMPLETED:
                # We deliver the result inline this turn, so mark the run reported.
                # Otherwise the consumer's _claim_unreported_subagents sees the
                # hidden result message as unreported and fires a duplicate seeded
                # orchestrator turn (double delivery + LLM cost).
                from django.utils import timezone

                SubAgentRun.objects.filter(pk=run.id).update(reported_at=timezone.now())
                if run.result or run.canvas:
                    from chat.subagent_service import render_canvas_block

                    body = run.result or (
                        "The sub-agent produced no final message, but left the "
                        "following working canvas."
                    )
                    return body + render_canvas_block(run)
                return json.dumps({
                    "status": "completed",
                    "run_id": str(run.id),
                    "message": "Sub-agent completed but returned no text content. "
                    "Its findings may have been lost. Consider doing this research directly.",
                })
            if run.status == SubAgentRun.Status.FAILED:
                return json.dumps({
                    "status": "error",
                    "message": f"Sub-agent failed: {run.error}",
                })

        # Timeout exceeded — still waiting for a slot, or still running
        queue = get_queue_depth()
        if _is_waiting(run.id):
            return json.dumps({
                "status": "queued",
                "run_id": str(run.id),
                "currently_running": queue["running"],
                "ahead_in_queue": queue["waiting"],
                "worker_slots": queue["worker_slots"],
                "message": (
                    f"Sub-agent is still waiting for a free execution slot after {timeout}s "
                    f"({queue['running']} running, {queue['waiting']} waiting). It starts "
                    "automatically and its result will appear in the conversation — do not create it again."
                ),
            })
        return json.dumps({
            "status": "started",
            "run_id": str(run.id),
            "currently_running": queue["running"],
            "ahead_in_queue": queue["waiting"],
            "worker_slots": queue["worker_slots"],
            "message": f"Sub-agent is still running after {timeout}s. Its result will appear in the conversation automatically.",
        })



# Register on import
registry = get_tool_registry()
registry.register_tool(CreateSubagentTool())
