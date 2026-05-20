"""Core sub-agent execution service."""

from __future__ import annotations

import logging
import uuid

from django.utils import timezone

from core.preferences import ResolvedPreferences

logger = logging.getLogger(__name__)


def resolve_subagent_model(tier: str, prefs: ResolvedPreferences) -> str:
    """Map a tier name to the user's configured model for that tier."""
    mapping = {
        "mid": prefs.mid_model,
        "top": prefs.top_model,
    }
    return mapping.get(tier, prefs.mid_model)


def resolve_subagent_tools(
    prefs: ResolvedPreferences,
    data_room_ids: list[int],
) -> list[str]:
    """Return the tool list for a sub-agent.

    Starts with the user's allowed chat tools, then:
    - Removes canvas tools (orchestrator-only)
    - Removes sub-agent tools (prevents recursion)
    - Removes document tools if no data rooms attached
    """
    excluded = {
        "active_canvas", "write_canvas", "edit_canvas",
        "create_subagent",
        "attach_skills", "create_skill", "edit_skill", "delete_skill",
        "save_canvas_to_skill_field", "show_skill_field_in_canvas",
        "view_template", "load_template_to_canvas",
        "list_skill_tools", "inspect_tool",
    }
    tools = [t for t in prefs.allowed_tools if t not in excluded]

    if not data_room_ids:
        doc_tools = {"search_documents", "read_document"}
        tools = [t for t in tools if t not in doc_tools]

    return tools


def run_subagent(run_id: uuid.UUID, *, deadline_seconds: int | None = None) -> None:
    """Execute a sub-agent run. Called by the Celery task."""
    from chat.models import SubAgentRun
    from chat.subagent_prompts import build_subagent_system_prompt
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    run = SubAgentRun.objects.select_related("user", "thread").get(pk=run_id)

    try:
        run.status = SubAgentRun.Status.RUNNING
        run.save(update_fields=["status"])

        user = run.user
        prefs = get_preferences(user)

        # Resolve model
        model = resolve_subagent_model(run.model_tier, prefs)
        run.model_used = model

        # Resolve tools
        data_room_ids = run.data_room_ids or []
        tool_list = resolve_subagent_tools(prefs, data_room_ids)
        run.tool_names = tool_list
        run.save(update_fields=["model_used", "tool_names"])

        # Get org name for prompt
        from accounts.models import Membership
        membership = Membership.objects.filter(user=user).select_related("org").first()
        org_name = membership.org.name if membership else None

        # Build data room info for prompt
        data_rooms_info = None
        if data_room_ids:
            from documents.models import DataRoom
            from documents.views import _user_can_access_data_room

            candidate_rooms = DataRoom.objects.filter(pk__in=data_room_ids)
            data_rooms_info = [
                {"id": r.pk, "name": r.name, "description": r.description or ""}
                for r in candidate_rooms
                if _user_can_access_data_room(user, r)
            ]

        # Load thread tasks for sub-agent context
        from chat.models import ThreadTask
        thread_tasks = list(
            ThreadTask.objects.filter(thread_id=run.thread_id)
            .order_by("order", "created_at")
            .values("id", "title", "status")
        )

        has_task_tool = "update_tasks" in (tool_list or [])
        system_prompt = build_subagent_system_prompt(
            data_rooms=data_rooms_info,
            organization_name=org_name,
            tasks=thread_tasks if thread_tasks else None,
            has_task_tool=has_task_tool,
        )

        # Build LLM request
        context = RunContext.create(
            user_id=user.pk,
            conversation_id=str(run.thread_id),
            data_room_ids=data_room_ids,
            deadline_seconds=deadline_seconds,
        )
        context.run_id = str(run_id)

        # Cooperative cancellation: check if the run has been marked FAILED
        def _is_cancelled():
            return SubAgentRun.objects.filter(pk=run_id, status=SubAgentRun.Status.FAILED).exists()

        request = ChatRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=run.prompt),
            ],
            model=model,
            stream=False,
            tools=tool_list if tool_list else None,
            context=context,
            params={"_cancel_check": _is_cancelled},
        )

        # Execute
        service = get_llm_service()
        response = service.run("simple_chat", request)

        # Store result
        run.result = response.message.content or ""
        if response.usage:
            run.tokens_used = response.usage.total_tokens or 0
            run.cost_usd = response.usage.cost_usd or 0.0
        if run.result:
            run.status = SubAgentRun.Status.COMPLETED
        else:
            run.status = SubAgentRun.Status.FAILED
            run.error = "Sub-agent produced no text output despite using tools."
            logger.warning(
                "Sub-agent run %s finished with no content (tokens_used=%d); marking FAILED",
                run_id, run.tokens_used,
            )
        run.completed_at = timezone.now()
        run.save(update_fields=[
            "result", "tokens_used", "cost_usd", "status", "error", "completed_at",
        ])

        # Persist the result as a hidden ChatMessage so it survives
        # across reconnects and failed LLM streams.  Uses role="user"
        # (not "tool") to avoid orphan tool_call_ids that OpenAI rejects.
        if run.result:
            from chat.models import ChatMessage
            from core.tokens import count_tokens

            short_id = str(run.id)[:8]
            wrapped = f"[Sub-agent result: {short_id}]\n{run.result}"
            ChatMessage.objects.create(
                thread_id=run.thread_id,
                role="user",
                content=wrapped,
                metadata={"source": "subagent", "subagent_run_id": str(run.id)},
                token_count=count_tokens(wrapped),
                is_hidden_from_user=True,
            )

        # Notify the WebSocket consumer so it auto-triggers the orchestrator
        from chat.tasks import _notify_consumer

        for _attempt in range(2):
            try:
                _notify_consumer(str(run.id), str(run.thread_id))
                break
            except Exception:
                if _attempt == 0:
                    import time
                    time.sleep(0.5)
                    continue
                logger.warning(
                    "Failed to notify consumer of sub-agent %s completion after 2 attempts",
                    run.id,
                )

    except Exception as exc:
        logger.exception("Sub-agent run %s failed", run_id)
        run.status = SubAgentRun.Status.FAILED
        run.error = str(exc)
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "error", "completed_at"])
        raise
