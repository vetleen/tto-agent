"""Sub-agent tools: create and check sub-agent runs."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, get_tool_registry

logger = logging.getLogger(__name__)


# --- Input schemas ---

class CreateSubagentInput(BaseModel):
    prompt: str = Field(description="The task description for the sub-agent.")
    skill_slug: str = Field(
        default="",
        description="Optional skill slug to load for the sub-agent (must be available to the user).",
    )
    model_tier: str = Field(
        default="mid",
        description='Model tier: "fast" for simple lookups, "mid" (default) for research, "top" for deep analysis.',
    )
    blocking: bool = Field(
        default=False,
        description="If true, wait for the sub-agent to finish and return its result inline. If false, start it in the background.",
    )


class CheckSubagentStatusInput(BaseModel):
    run_id: str = Field(
        default="",
        description="Optional sub-agent run ID. If omitted, returns all runs for this thread.",
    )


# --- Tools ---

class CreateSubagentTool(ContextAwareTool):
    """Create a sub-agent to handle a delegated task."""

    name: str = "create_subagent"
    description: str = (
        "Delegate a task to an independent sub-agent that runs with its own context and tools. "
        "Use for tasks requiring extensive research, parallel analysis, or focused work. "
        "Set blocking=true to wait for the result, or blocking=false to run in background."
    )
    args_schema: type[BaseModel] = CreateSubagentInput

    def _run(
        self,
        prompt: str,
        skill_slug: str = "",
        model_tier: str = "mid",
        blocking: bool = False,
    ) -> str:
        from django.contrib.auth import get_user_model

        from chat.models import ChatThread, SubAgentRun
        from chat.subagent_limits import check_subagent_limits
        from chat.subagent_service import resolve_subagent_model, resolve_subagent_tools, run_subagent
        from core.preferences import get_preferences

        context = self.context
        if not context or not context.user_id:
            return json.dumps({"status": "error", "message": "No user context available."})

        thread_id = context.conversation_id if context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # Validate model_tier
        if model_tier not in ("fast", "mid", "top"):
            model_tier = "mid"

        # Load user
        User = get_user_model()
        try:
            user = User.objects.get(pk=context.user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        # Check concurrency limits
        allowed, err_msg = check_subagent_limits(user)
        if not allowed:
            return json.dumps({"status": "error", "message": err_msg})

        # Resolve preferences, tools, model
        prefs = get_preferences(user)
        data_room_ids = context.data_room_ids if context else []

        # Load skill if specified
        skill = None
        if skill_slug:
            from agent_skills.services import get_available_skills
            for s in get_available_skills(user):
                if s.slug == skill_slug:
                    skill = s
                    break

        tool_list = resolve_subagent_tools(prefs, data_room_ids, skill=skill)
        model = resolve_subagent_model(model_tier, prefs)

        # Create run record
        run = SubAgentRun.objects.create(
            thread_id=thread_id,
            user=user,
            prompt=prompt,
            skill_slug=skill_slug,
            model_tier=model_tier,
            model_used=model,
            blocking=blocking,
            data_room_ids=data_room_ids,
            tool_names=tool_list,
        )

        if blocking:
            # Execute synchronously with a deadline so the LLM service
            # will abort if the sub-agent takes too long.
            BLOCKING_TIMEOUT = 270  # seconds, matches Celery soft_time_limit
            try:
                run_subagent(run.id, deadline_seconds=BLOCKING_TIMEOUT)
            except Exception:
                # run_subagent sets PENDING on error (for Celery retry), but
                # blocking calls don't retry — mark FAILED permanently.
                from django.utils import timezone as tz
                run.refresh_from_db()
                if run.status != SubAgentRun.Status.COMPLETED:
                    run.status = SubAgentRun.Status.FAILED
                    run.completed_at = tz.now()
                    run.save(update_fields=["status", "completed_at"])

            run.refresh_from_db()
            if run.status == SubAgentRun.Status.COMPLETED:
                return run.result
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Sub-agent failed: {run.error}",
                })
        else:
            # Dispatch to Celery
            from chat.tasks import run_subagent_task
            task = run_subagent_task.delay(str(run.id))
            run.celery_task_id = task.id
            run.save(update_fields=["celery_task_id"])

            return json.dumps({
                "status": "started",
                "run_id": str(run.id),
                "message": f"Sub-agent has been started in the background (model: {model_tier}). Use check_subagent_status to check its progress.",
            })


class CheckSubagentStatusTool(ContextAwareTool):
    """Check the status of background sub-agent runs."""

    name: str = "check_subagent_status"
    description: str = (
        "Check the status and results of background sub-agent runs. "
        "Provide a specific run_id or omit to see all runs for this thread."
    )
    args_schema: type[BaseModel] = CheckSubagentStatusInput

    def _run(self, run_id: str = "") -> str:
        from chat.models import SubAgentRun

        context = self.context
        thread_id = context.conversation_id if context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        if run_id:
            try:
                run = SubAgentRun.objects.get(pk=run_id, thread_id=thread_id)
            except SubAgentRun.DoesNotExist:
                return json.dumps({"status": "error", "message": f"Sub-agent run {run_id} not found."})

            result = {
                "run_id": str(run.id),
                "status": run.status,
                "prompt": run.prompt[:200],
                "model_tier": run.model_tier,
            }
            if run.status == SubAgentRun.Status.COMPLETED:
                result["result"] = run.result
            elif run.status == SubAgentRun.Status.FAILED:
                result["error"] = run.error
            return json.dumps(result)

        # Return all runs for this thread
        runs = SubAgentRun.objects.filter(thread_id=thread_id).order_by("-created_at")[:20]
        results = []
        for run in runs:
            entry = {
                "run_id": str(run.id),
                "status": run.status,
                "prompt": run.prompt[:200],
                "model_tier": run.model_tier,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            }
            if run.status == SubAgentRun.Status.COMPLETED:
                entry["result"] = run.result
            elif run.status == SubAgentRun.Status.FAILED:
                entry["error"] = run.error
            results.append(entry)

        return json.dumps({"runs": results, "count": len(results)})


# Register on import
registry = get_tool_registry()
registry.register_tool(CreateSubagentTool())
registry.register_tool(CheckSubagentStatusTool())
