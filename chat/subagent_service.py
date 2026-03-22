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
        "fast": prefs.cheap_model,
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
    excluded = {"write_canvas", "edit_canvas", "create_subagent"}
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

        # Build system prompt — no skill injection; the orchestrator writes
        # task-specific instructions directly in run.prompt.
        system_prompt = build_subagent_system_prompt(
            run.prompt,
            data_rooms=data_rooms_info,
            organization_name=org_name,
            tasks=thread_tasks if thread_tasks else None,
        )

        # Build LLM request
        context = RunContext.create(
            user_id=user.pk,
            conversation_id=str(run.thread_id),
            data_room_ids=data_room_ids,
            deadline_seconds=deadline_seconds,
        )

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
        if not run.result and response.message.tool_calls:
            logger.warning(
                "Sub-agent run %s completed with unresolved tool calls and no content",
                run_id,
            )
        if response.usage:
            run.tokens_used = response.usage.total_tokens or 0
            run.cost_usd = response.usage.cost_usd or 0.0
        run.status = SubAgentRun.Status.COMPLETED
        run.completed_at = timezone.now()
        run.save(update_fields=[
            "result", "tokens_used", "cost_usd", "status", "completed_at",
        ])

    except Exception as exc:
        logger.exception("Sub-agent run %s failed", run_id)
        # Set back to PENDING so Celery retries don't show a premature
        # "failed" status to the user.  The Celery on_failure handler
        # will set FAILED permanently once all retries are exhausted.
        run.status = SubAgentRun.Status.PENDING
        run.error = str(exc)
        run.save(update_fields=["status", "error"])
        raise
