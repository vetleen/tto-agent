"""Core sub-agent execution service."""

from __future__ import annotations

import logging
import uuid
from typing import Any

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
    skill: Any = None,
) -> list[str]:
    """Return the tool list for a sub-agent.

    Starts with the user's allowed chat tools, then:
    - Removes canvas tools (orchestrator-only)
    - Removes sub-agent tools (prevents recursion)
    - Removes document tools if no data rooms attached
    - Adds skill tools if a skill is provided
    """
    excluded = {"write_canvas", "edit_canvas", "create_subagent", "check_subagent_status"}
    tools = [t for t in prefs.allowed_tools if t not in excluded]

    if not data_room_ids:
        doc_tools = {"search_documents", "read_document"}
        tools = [t for t in tools if t not in doc_tools]

    if skill and prefs.allowed_skills:
        # Find the skill in allowed_skills to get its filtered tool_names
        for s in prefs.allowed_skills:
            if s["slug"] == skill.slug:
                for t in s["tool_names"]:
                    if t not in tools and t not in excluded:
                        tools.append(t)
                break

    return tools


def run_subagent(run_id: uuid.UUID, *, deadline_seconds: int | None = None, blocking: bool = False) -> None:
    """Execute a sub-agent run. Called by both blocking tool and Celery task."""
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

        # Load skill if specified
        skill = None
        if run.skill_slug:
            from agent_skills.services import get_available_skills
            for s in get_available_skills(user):
                if s.slug == run.skill_slug:
                    skill = s
                    break

        # Resolve tools
        data_room_ids = run.data_room_ids or []
        tool_list = resolve_subagent_tools(prefs, data_room_ids, skill=skill)
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
            data_rooms_info = list(
                DataRoom.objects.filter(pk__in=data_room_ids, created_by=user)
                .values("id", "name", "description")
            )

        # Build system prompt
        system_prompt = build_subagent_system_prompt(
            run.prompt,
            skill=skill,
            data_rooms=data_rooms_info,
            organization_name=org_name,
        )

        # Build LLM request
        context = RunContext.create(
            user_id=user.pk,
            conversation_id=str(run.thread_id),
            data_room_ids=data_room_ids,
            deadline_seconds=deadline_seconds,
        )

        request = ChatRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=run.prompt),
            ],
            model=model,
            stream=False,
            tools=tool_list if tool_list else None,
            context=context,
        )

        # Execute
        service = get_llm_service()
        response = service.run("simple_chat", request)

        # Store result
        run.result = response.message.content or ""
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
        if blocking:
            # Blocking calls don't retry — mark FAILED permanently.
            run.status = SubAgentRun.Status.FAILED
            run.error = str(exc)
            run.completed_at = timezone.now()
            run.save(update_fields=["status", "error", "completed_at"])
        else:
            # Set back to PENDING so Celery retries don't show a premature
            # "failed" status to the user.  The Celery on_failure handler
            # will set FAILED permanently once all retries are exhausted.
            run.status = SubAgentRun.Status.PENDING
            run.error = str(exc)
            run.save(update_fields=["status", "error"])
        raise
