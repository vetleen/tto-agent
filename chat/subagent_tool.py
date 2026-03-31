"""Sub-agent tools: create and check sub-agent runs."""

from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


# --- Input schemas ---

class CreateSubagentInput(ReasonBaseModel):
    prompt: str = Field(description="The task description for the sub-agent.")
    model_tier: str = Field(
        default="mid",
        description='Model tier: "fast" for simple lookups, "mid" (default) for research, "top" for deep analysis.',
    )
    timeout: int = Field(
        default=0,
        description="Seconds to wait for the result (0-540). 0=background, 30-60=quick tasks, 120=research.",
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
        model_tier: str = "mid",
        timeout: int = 0,
        **kwargs,
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
                "message": f"Sub-agent has been started in the background (model: {model_tier}). Its status will appear in the conversation automatically.",
            })

        # Poll for result until timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(2)
            run.refresh_from_db()
            if run.status == SubAgentRun.Status.COMPLETED:
                if run.result:
                    return run.result
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

        # Timeout exceeded — still running
        return json.dumps({
            "status": "started",
            "run_id": str(run.id),
            "message": f"Sub-agent is still running after {timeout}s. Its status will appear in the conversation automatically.",
        })



# Register on import
registry = get_tool_registry()
registry.register_tool(CreateSubagentTool())
