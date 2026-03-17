"""Sub-agent tools: create and check sub-agent runs."""

from __future__ import annotations

import json
import logging
import time

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
    timeout: int = Field(
        default=0,
        description="Seconds to wait for the result (0-270). 0=background, 30-60=quick tasks, 120=research.",
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
        "Set timeout=0 for background, 30-60 for quick tasks, 120 for research."
    )
    args_schema: type[BaseModel] = CreateSubagentInput

    def _run(
        self,
        prompt: str,
        skill_slug: str = "",
        model_tier: str = "mid",
        timeout: int = 0,
    ) -> str:
        from django.contrib.auth import get_user_model

        from chat.models import SubAgentRun
        from chat.subagent_limits import create_subagent_run_if_allowed

        context = self.context
        if not context or not context.user_id:
            return json.dumps({"status": "error", "message": "No user context available."})

        thread_id = context.conversation_id if context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        # Validate model_tier
        if model_tier not in ("fast", "mid", "top"):
            model_tier = "mid"

        # Clamp timeout to [0, 270]
        timeout = max(0, min(timeout, 270))

        # Load user
        User = get_user_model()
        try:
            user = User.objects.get(pk=context.user_id)
        except User.DoesNotExist:
            return json.dumps({"status": "error", "message": "User not found."})

        data_room_ids = context.data_room_ids if context else []

        # Atomically check limits and create the run record.
        # The service resolves model_used and tool_names when it executes.
        run, err_msg = create_subagent_run_if_allowed(
            user,
            thread_id=thread_id,
            prompt=prompt,
            skill_slug=skill_slug,
            model_tier=model_tier,
            timeout=timeout,
            data_room_ids=data_room_ids,
        )
        if run is None:
            return json.dumps({"status": "error", "message": err_msg})

        # Always dispatch to Celery
        from chat.tasks import run_subagent_task
        task = run_subagent_task.delay(str(run.id))
        run.celery_task_id = task.id
        run.save(update_fields=["celery_task_id"])

        if timeout == 0:
            return json.dumps({
                "status": "started",
                "run_id": str(run.id),
                "message": f"Sub-agent has been started in the background (model: {model_tier}). Use check_subagent_status to check its progress.",
            })

        # Poll for result until timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            run.refresh_from_db()
            if run.status == SubAgentRun.Status.COMPLETED:
                return run.result
            if run.status == SubAgentRun.Status.FAILED:
                return json.dumps({
                    "status": "error",
                    "message": f"Sub-agent failed: {run.error}",
                })

        # Timeout exceeded — still running
        return json.dumps({
            "status": "started",
            "run_id": str(run.id),
            "message": f"Sub-agent is still running after {timeout}s. Use check_subagent_status to check its progress.",
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
        if not context or not context.user_id:
            return json.dumps({"status": "error", "message": "No user context available."})
        thread_id = context.conversation_id if context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        if run_id:
            try:
                run = SubAgentRun.objects.get(
                    pk=run_id, thread_id=thread_id, user_id=context.user_id,
                )
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

        # Return all runs for this thread belonging to the current user
        runs = SubAgentRun.objects.filter(
            thread_id=thread_id, user_id=context.user_id,
        ).order_by("-created_at")[:20]
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
