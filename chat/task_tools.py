"""Task plan tools: manage a per-thread task list."""

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


class UpdateTasksTool(ContextAwareTool):
    """Create, update, or replace the task plan for this conversation."""

    name: str = "update_tasks"
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

    def _run(self, tasks: list[dict] | list[TaskItem], **kwargs) -> str:
        from chat.models import ThreadTask

        thread_id = self.context.conversation_id if self.context else None
        if not thread_id:
            return json.dumps({"status": "error", "message": "No thread context available."})

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

        # Update existing and create new
        result_tasks = []
        for item in incoming:
            if item["existing_id"] and item["existing_id"] in current_tasks:
                # Update existing
                task = current_tasks[item["existing_id"]]
                task.title = item["title"]
                task.status = item["status"]
                task.order = item["order"]
                task.save(update_fields=["title", "status", "order", "updated_at"])
            else:
                # Create new
                task = ThreadTask.objects.create(
                    thread_id=thread_id,
                    title=item["title"],
                    status=item["status"],
                    order=item["order"],
                )
            result_tasks.append({
                "id": str(task.id),
                "title": task.title,
                "status": task.status,
                "order": task.order,
            })

        # Build summary
        counts = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in result_tasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        total = len(result_tasks)
        parts = []
        if counts["completed"]:
            parts.append(f"{counts['completed']} done")
        if counts["in_progress"]:
            parts.append(f"{counts['in_progress']} active")
        if counts["pending"]:
            parts.append(f"{counts['pending']} pending")
        summary = f"{total} tasks ({', '.join(parts)})" if parts else "0 tasks"

        return json.dumps({
            "status": "ok",
            "tasks": result_tasks,
            "summary": summary,
        })


# Register on import
_registry = get_tool_registry()
_registry.register_tool(UpdateTasksTool())
