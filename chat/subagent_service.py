"""Core sub-agent execution service."""

from __future__ import annotations

import logging
import uuid

from django.utils import timezone

from core.preferences import ResolvedPreferences

logger = logging.getLogger(__name__)


def is_retryable_subagent_error(exc: BaseException) -> bool:
    """Whether a sub-agent failure is transient and worth retrying the whole run.

    Single source of truth shared by :func:`run_subagent` (which leaves the run
    RUNNING instead of marking it FAILED for these, so a retry can re-enter
    cleanly) and ``chat.tasks.run_subagent_task`` (which calls ``self.retry``).
    Covers transient LLM provider errors (rate-limit / overload / timeout /
    connection) and low-level network errors. A Postgres "too many connections"
    error is treated as terminal — retrying immediately would just hit the cap
    again.
    """
    from django.db.utils import OperationalError
    from llm.service.errors import (
        LLMConnectionError,
        LLMOverloadedError,
        LLMRateLimitError,
        LLMTimeoutError,
    )

    if isinstance(exc, OperationalError):
        return "too many connections" not in str(exc).lower()
    return isinstance(exc, (
        LLMRateLimitError,
        LLMOverloadedError,
        LLMTimeoutError,
        LLMConnectionError,
        ConnectionError,
        TimeoutError,
        OSError,
    ))


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
    specialization_slug: str = "",
) -> list[str]:
    """Return the tool list for a sub-agent.

    Base = the user's sub-agent-audience chat tools
    (``prefs.allowed_subagent_tools``). Tools that must never run in a sub-agent
    — canvas, loops, skill management, spawning sub-agents — are declared
    ``audience="main"`` and so are absent from that list (replacing the old
    hard-coded denylist). Document tools are dropped when no data rooms are
    attached. A specialization ("type") contributes its skill-section tools,
    already audience-filtered and org-toggle-filtered in
    ``prefs.allowed_specializations``.
    """
    tools = list(prefs.allowed_subagent_tools)

    if not data_room_ids:
        doc_tools = {"document_search", "document_read"}
        tools = [t for t in tools if t not in doc_tools]

    if specialization_slug:
        spec = next(
            (s for s in prefs.allowed_specializations if s["slug"] == specialization_slug),
            None,
        )
        if spec:
            for t in spec["tool_names"]:
                if t not in tools:
                    tools.append(t)

    return tools


def get_run_specialization_skill(run):
    """Resolve the ``AgentSkill`` backing a run's specialization, or ``None``.

    Re-resolves through the same access gate / shadowing as spawn time, so a
    skill revoked mid-run is dropped gracefully. Shared by ``run_subagent``'s
    template auto-load and the ``subagent_canvas_load_template`` tool.
    """
    if not run.skill_slug:
        return None
    from agent_skills.services import get_subagent_skills

    return next(
        (s for s in get_subagent_skills(run.user) if s.slug == run.skill_slug),
        None,
    )


def render_canvas_block(run) -> str:
    """The delimited working-canvas block appended to a sub-agent's return.

    Empty string when the run has no canvas, so callers can append it
    unconditionally.
    """
    if not run.canvas:
        return ""
    title = run.canvas_title or "Working document"
    return f"\n\n=== SUBAGENT CANVAS: {title} ===\n{run.canvas}"


def _subagent_result_message_exists(run) -> bool:
    """Whether a hidden result message already exists for this run."""
    from chat.models import ChatMessage

    return ChatMessage.objects.filter(
        thread_id=run.thread_id,
        metadata__source="subagent",
        metadata__subagent_run_id=str(run.id),
    ).exists()


def _create_subagent_result_message(run) -> None:
    """Persist the sub-agent's result (+ working canvas) as a hidden ChatMessage.

    Hidden, ``role="user"`` (not ``"tool"``, to avoid orphan tool_call_ids that
    OpenAI rejects). The canvas is appended clearly delimited so the orchestrator
    sees it automatically on its next turn (``_load_history`` replays the content
    verbatim). Survives reconnects and failed LLM streams.
    """
    from chat.models import ChatMessage
    from core.tokens import count_tokens

    short_id = str(run.id)[:8]
    if run.result:
        wrapped = f"[Sub-agent result: {short_id}]\n{run.result}"
    else:
        wrapped = (
            f"[Sub-agent result: {short_id}]\n"
            "The sub-agent produced no final message, but left the following working canvas."
        )
    wrapped += render_canvas_block(run)
    ChatMessage.objects.create(
        thread_id=run.thread_id,
        role="user",
        content=wrapped,
        metadata={"source": "subagent", "subagent_run_id": str(run.id)},
        token_count=count_tokens(wrapped),
        is_hidden_from_user=True,
    )


def _notify_consumer_safely(run_id: str, thread_id: str) -> None:
    """Notify the WebSocket consumer of a finished run; best-effort with one retry."""
    from chat.tasks import _notify_consumer

    for attempt in range(2):
        try:
            _notify_consumer(run_id, thread_id)
            return
        except Exception:
            if attempt == 0:
                import time

                time.sleep(0.5)
                continue
            logger.warning(
                "Failed to notify consumer of sub-agent %s completion after 2 attempts",
                run_id,
            )


def run_subagent(run_id: uuid.UUID, *, deadline_seconds: int | None = None) -> None:
    """Execute a sub-agent run. Called by the Celery task."""
    from chat.models import SubAgentRun
    from chat.subagent_prompts import build_subagent_system_prompt
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    run = SubAgentRun.objects.select_related("user", "thread").get(pk=run_id)

    try:
        # Guarded transition: if the user cancelled while the run was PENDING
        # (status already FAILED), don't resurrect it.
        started = SubAgentRun.objects.filter(pk=run_id).exclude(
            status=SubAgentRun.Status.FAILED,
        ).update(status=SubAgentRun.Status.RUNNING, started_at=timezone.now())
        if not started:
            logger.info("Sub-agent run %s was cancelled before starting; skipping", run_id)
            return

        user = run.user
        prefs = get_preferences(user)

        # Resolve model
        model = resolve_subagent_model(run.model_tier, prefs)
        run.model_used = model

        # Resolve an optional specialization ("type"). The carrying skill is
        # re-resolved through the same access gate / shadowing as at spawn time,
        # so a skill deleted or revoked in between is dropped gracefully.
        specialization_skill = get_run_specialization_skill(run)
        if run.skill_slug and specialization_skill is None:
            logger.info(
                "Sub-agent run %s referenced specialization '%s' that is no "
                "longer available; running without it",
                run_id, run.skill_slug,
            )

        # Auto-load the first template (name-ordered, matching the prompt's
        # bullet list) into the working canvas as a starting point. Persisted
        # immediately so the canvas is durable from the very start of the run.
        if specialization_skill is not None:
            first_tmpl = specialization_skill.templates.all().first()
            if first_tmpl is not None:
                from chat.services import CANVAS_MAX_CHARS

                run.canvas = (first_tmpl.content or "")[:CANVAS_MAX_CHARS]
                run.canvas_title = first_tmpl.name[:255]
                run.save(update_fields=["canvas", "canvas_title"])

        # Resolve tools
        data_room_ids = run.data_room_ids or []
        tool_list = resolve_subagent_tools(
            prefs, data_room_ids, specialization_slug=run.skill_slug,
        )
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

        has_task_tool = "chat_task_update" in (tool_list or [])
        system_prompt = build_subagent_system_prompt(
            data_rooms=data_rooms_info,
            organization_name=org_name,
            tasks=thread_tasks if thread_tasks else None,
            has_task_tool=has_task_tool,
            specialization_skill=specialization_skill,
            canvas_content=run.canvas,
            canvas_title=run.canvas_title,
        )

        # Build LLM request
        context = RunContext.create(
            user_id=user.pk,
            conversation_id=str(run.thread_id),
            data_room_ids=data_room_ids,
            deadline_seconds=deadline_seconds,
        )
        context.run_id = str(run_id)
        context.agent_kind = "subagent"

        # Cooperative cancellation: check if the run has been marked FAILED
        def _is_cancelled():
            return SubAgentRun.objects.filter(pk=run_id, status=SubAgentRun.Status.FAILED).exists()

        request = ChatRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=run.prompt),
            ],
            model=model,
            stream=True,
            tools=tool_list if tool_list else None,
            context=context,
            params={"_cancel_check": _is_cancelled},
        )

        # Execute via streaming so a long generation never trips a non-streaming
        # read timeout (Anthropic sends no bytes until a non-streamed completion
        # is fully generated; streamed tokens reset the read timeout). Tokens are
        # not surfaced live — run_via_stream collapses the stream into the same
        # ChatResponse shape run() returns.
        service = get_llm_service()
        response = service.run_via_stream("simple_chat", request)

        # Store result
        run.result = response.message.content or ""
        # Tools wrote the working canvas straight to the DB row during the run;
        # refresh so the delivered/return logic below sees the latest content.
        run.refresh_from_db(fields=["canvas", "canvas_title"])
        if response.usage:
            run.tokens_used = response.usage.total_tokens or 0
            run.cost_usd = response.usage.cost_usd or 0.0
        # A populated canvas is a deliverable in its own right, so a run that
        # built one but emitted no final text still counts as completed.
        delivered = bool(run.result) or bool(run.canvas)
        if delivered:
            run.status = SubAgentRun.Status.COMPLETED
        else:
            run.status = SubAgentRun.Status.FAILED
            run.error = "Sub-agent produced no text output and no canvas despite using tools."
            logger.warning(
                "Sub-agent run %s finished with no content (tokens_used=%d); marking FAILED",
                run_id, run.tokens_used,
            )
        run.completed_at = timezone.now()
        # Guarded write: if the user cancelled mid-run (status flipped to
        # FAILED by _cancel_active_subagents), don't overwrite the
        # cancellation with COMPLETED — and don't report the result.
        finished = SubAgentRun.objects.filter(pk=run_id).exclude(
            status=SubAgentRun.Status.FAILED,
        ).update(
            result=run.result,
            tokens_used=run.tokens_used,
            cost_usd=run.cost_usd,
            status=run.status,
            error=run.error,
            completed_at=run.completed_at,
        )
        if not finished:
            logger.info(
                "Sub-agent run %s was cancelled mid-run; discarding its result", run_id,
            )
            return

        # Persist the result (and working canvas) as a hidden ChatMessage so it
        # survives across reconnects and failed LLM streams.
        if delivered:
            _create_subagent_result_message(run)

        # Notify the WebSocket consumer so it auto-triggers the orchestrator
        _notify_consumer_safely(str(run.id), str(run.thread_id))

    except Exception as exc:
        if is_retryable_subagent_error(exc):
            # Transient failure (Gemini rate-limit/overload/timeout, network or
            # DB blip). Leave the run RUNNING so the Celery retry re-enters via
            # the guarded RUNNING transition above; on_failure records FAILED
            # only once retries are exhausted. Writing FAILED here would make the
            # retry a no-op (the guard skips FAILED runs) and would break the
            # "FAILED == cancelled-or-terminal" invariant a user cancel relies on.
            logger.warning(
                "Sub-agent run %s hit a transient error (%s); will retry",
                run_id, type(exc).__name__,
            )
            raise
        logger.exception("Sub-agent run %s failed", run_id)
        # Durability: a terminal (non-cancel) failure shouldn't discard a canvas
        # the sub-agent already built. Surface it to the orchestrator before
        # marking FAILED. A cancelled run is already FAILED, so it's skipped — and
        # the hidden-message guard avoids a duplicate if one somehow exists.
        try:
            fresh = SubAgentRun.objects.get(pk=run_id)
            if (
                fresh.status != SubAgentRun.Status.FAILED
                and fresh.canvas
                and not _subagent_result_message_exists(fresh)
            ):
                _create_subagent_result_message(fresh)
                _notify_consumer_safely(str(fresh.id), str(fresh.thread_id))
        except Exception:
            logger.warning(
                "Failed to surface canvas for terminally-failed sub-agent run %s",
                run_id, exc_info=True,
            )
        # Guarded: keep an earlier failure reason (e.g. "Cancelled by user.")
        # instead of clobbering it with this exception's message.
        SubAgentRun.objects.filter(pk=run_id).exclude(
            status=SubAgentRun.Status.FAILED,
        ).update(
            status=SubAgentRun.Status.FAILED,
            error=str(exc),
            completed_at=timezone.now(),
        )
        raise
