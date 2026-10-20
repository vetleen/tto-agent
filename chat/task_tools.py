"""Task plan tools: the main agent's per-thread task list, and a sub-agent's
per-run plan (never the thread's)."""

from __future__ import annotations

import json
import uuid as uuid_mod

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry


class TaskItem(BaseModel):
    id: str = Field(default="", description="UUID of an existing task. Leave empty for new tasks.")
    title: str = Field(description="Short, action-oriented task title.")
    status: str = Field(
        default="pending",
        description="Task status: pending, in_progress, or completed.",
    )


class UpdateTasksInput(ReasonBaseModel):
    tasks: list[TaskItem] = Field(
        description="The complete desired task list. Omitted tasks are deleted.",
    )


def _parse_incoming(tasks: list[dict] | list[TaskItem]) -> list[dict]:
    """Normalize the model's task list: valid statuses (else pending), parsed
    UUIDs (else None), titles capped, order = list position."""
    from chat.models import ThreadTask

    valid_statuses = {s.value for s in ThreadTask.Status}

    # Parse incoming tasks
    incoming = []
    for i, t in enumerate(tasks):
        if isinstance(t, dict):
            tid = t.get("id", "")
            title = t.get("title", "")
            status = t.get("status", "pending")
        else:
            tid = t.id
            title = t.title
            status = t.status

        if status not in valid_statuses:
            status = "pending"

        # Validate UUID
        existing_id = None
        if tid:
            try:
                existing_id = str(uuid_mod.UUID(tid))
            except (ValueError, AttributeError):
                existing_id = None

        incoming.append({
            "existing_id": existing_id,
            "title": title[:512],
            "status": status,
            "order": i,
        })

    return incoming


def _summary(result_tasks: list[dict]) -> str:
    counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for t in result_tasks:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    parts = []
    if counts["completed"]:
        parts.append(f"{counts['completed']} done")
    if counts["in_progress"]:
        parts.append(f"{counts['in_progress']} active")
    if counts["pending"]:
        parts.append(f"{counts['pending']} pending")
    return f"{len(result_tasks)} tasks ({', '.join(parts)})" if parts else "0 tasks"


class UpdateTasksTool(ContextAwareTool):
    """Create, update, or replace the task plan for this conversation."""

    name: str = "chat_task_update"
    start_label: str = "Updating task plan..."
    end_label: str = "Updated task plan"
    description: str = (
        "Create or update the task plan for this conversation. Use proactively "
        "whenever work involves multiple steps — do not wait to be asked. "
        "Send the complete list of tasks each time; omitted tasks are deleted. "
        "Each task has a title and status (pending, in_progress, completed). "
        "Include the id field for existing tasks to preserve them. "
        "Mark tasks in_progress as you start them and completed when done."
    )
    args_schema: type[BaseModel] = UpdateTasksInput
    section: str = "chat"
    # The thread's plan belongs to the main agent: a sub-agent writing it wiped
    # the orchestrator's plan (full-list replace). Sub-agents get
    # subagent_task_update, stored on their run.
    audience: str = "main"

    def _run(self, tasks: list[dict] | list[TaskItem], **kwargs) -> str:
        from chat.models import ThreadTask

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

        incoming = _parse_incoming(tasks)

        # Fetch current tasks for this thread
        current_tasks = {
            str(t.id): t
            for t in ThreadTask.objects.filter(thread_id=thread_id)
        }

        # Build set of incoming IDs that match existing tasks
        incoming_existing_ids = {
            item["existing_id"]
            for item in incoming
            if item["existing_id"] and item["existing_id"] in current_tasks
        }

        # Delete tasks not in incoming set
        to_delete = [
            tid for tid in current_tasks if tid not in incoming_existing_ids
        ]
        if to_delete:
            ThreadTask.objects.filter(id__in=to_delete).delete()

        # Update existing and create new — batched so N incoming tasks cost one
        # bulk_update + one bulk_create, not one query per task.
        from django.utils import timezone

        now = timezone.now()
        result_tasks = []
        to_update = []
        to_create = []
        for item in incoming:
            if item["existing_id"] and item["existing_id"] in current_tasks:
                # Update existing
                task = current_tasks[item["existing_id"]]
                task.title = item["title"]
                task.status = item["status"]
                task.order = item["order"]
                # bulk_update does not fire auto_now, so stamp updated_at ourselves.
                task.updated_at = now
                to_update.append(task)
            else:
                # Create new. ThreadTask.id has a uuid4 default applied at
                # instantiation, so task.id is usable in the result before insert;
                # bulk_create fills created_at/updated_at (auto_now(_add)) on insert.
                task = ThreadTask(
                    thread_id=thread_id,
                    title=item["title"],
                    status=item["status"],
                    order=item["order"],
                )
                to_create.append(task)
            result_tasks.append({
                "id": str(task.id),
                "title": task.title,
                "status": task.status,
                "order": task.order,
            })

        if to_update:
            ThreadTask.objects.bulk_update(
                to_update, fields=["title", "status", "order", "updated_at"]
            )
        if to_create:
            ThreadTask.objects.bulk_create(to_create)

        return json.dumps({
            "status": "ok",
            "tasks": result_tasks,
            "summary": _summary(result_tasks),
        })


class SubagentTaskUpdateTool(ContextAwareTool):
    """Run-scoped task plan for a sub-agent, stored on ``SubAgentRun.task_plan``
    (see ``UpdateTasksTool`` for the main agent's per-thread plan). Carried over
    when the run is continued; never shown to the orchestrator or the user."""

    name: str = "subagent_task_update"
    audience: str = "subagent"
    section: str = "chat"  # always-on for sub-agents (see core.preferences)
    start_label: str = "Updating task plan..."
    end_label: str = "Updated task plan"
    description: str = (
        "Create or update your task plan for this run. It is private to you: the "
        "orchestrator and the user do not see it. Use it proactively whenever the "
        "task involves multiple steps. Send the complete list of tasks each time; "
        "omitted tasks are deleted. Each task has a title and status (pending, "
        "in_progress, completed). Include the id field for existing tasks to "
        "preserve them. Mark tasks in_progress as you start them and completed when done."
    )
    args_schema: type[BaseModel] = UpdateTasksInput

    def _run(self, tasks: list[dict] | list[TaskItem], **kwargs) -> str:
        run_id = getattr(self.context, "run_id", None) if self.context else None
        if not run_id or getattr(self.context, "agent_kind", "") != "subagent":
            return json.dumps({"status": "error", "message": "No sub-agent run context available."})

        from chat.models import SubAgentRun

        # One run executes on one worker thread, so a read-modify-write is safe.
        current = SubAgentRun.objects.filter(pk=run_id).values_list("task_plan", flat=True).first()
        if current is None:
            return json.dumps({"status": "error", "message": "Sub-agent run not found."})
        known_ids = {str(t.get("id")) for t in current if isinstance(t, dict)}

        result_tasks = [
            {
                "id": item["existing_id"] if item["existing_id"] in known_ids else str(uuid_mod.uuid4()),
                "title": item["title"],
                "status": item["status"],
                "order": item["order"],
            }
            for item in _parse_incoming(tasks)
        ]
        SubAgentRun.objects.filter(pk=run_id).update(task_plan=result_tasks)
        return json.dumps({
            "status": "ok",
            "tasks": result_tasks,
            "summary": _summary(result_tasks),
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(UpdateTasksTool())
_registry.register_tool(SubagentTaskUpdateTool())
