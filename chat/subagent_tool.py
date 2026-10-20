"""Sub-agent tools: create and check sub-agent runs."""

from __future__ import annotations

import json
import logging

from typing import Literal

from pydantic import BaseModel, Field

from core.preferences import DEFAULT_SUBAGENT_TIER, SUBAGENT_TIERS
from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry

logger = logging.getLogger(__name__)


def _is_waiting(run_id) -> bool:
    """True while the run is still queued for an execution slot (PENDING, never dispatched)."""
    from chat.models import SubAgentRun

    return SubAgentRun.objects.filter(
        pk=run_id, status=SubAgentRun.Status.PENDING, dispatched_at__isnull=True,
    ).exists()


# How many canvases one sub-agent may be handed. Size is no longer a concern:
# the snapshot is not pasted into the sub-agent's prompt, it is listed there and
# read on demand with subagent_canvas_read (ranged, token-paged), so this is
# only a sanity bound on the listing.
MAX_SHARED_CANVASES = 5


def _snapshot_canvases(thread_id, titles: list[str]) -> tuple[list[dict], str]:
    """Copy the named canvases' current text for a sub-agent.

    Returns ``(snapshot, "")`` or ``([], error_message)``. The error is
    actionable: too many titles states the limit, and an unknown title lists
    the canvases that do exist. Each entry carries ``chars``/``tokens`` so the
    sub-agent prompt can list the snapshot without re-counting it.
    """
    from core.tokens import count_tokens
    from chat.models import ChatCanvas
    from chat.services import resolve_canvas

    wanted = list(dict.fromkeys(t.strip() for t in titles if t and t.strip()))
    if len(wanted) > MAX_SHARED_CANVASES:
        return [], (
            f"You may share up to {MAX_SHARED_CANVASES} canvases with a sub-agent; "
            f"you asked for {len(wanted)}."
        )

    snapshot: list[dict] = []
    for title in wanted:
        canvas, err = resolve_canvas(thread_id, title)
        if err:
            available = list(
                ChatCanvas.objects.filter(thread_id=thread_id, deleted_at__isnull=True)
                .order_by("title")
                .values_list("title", flat=True)
            )
            listing = ", ".join(f'"{t}"' for t in available) if available else "(none)"
            return [], f"{err} Canvases in this conversation: {listing}."
        content = canvas.content or ""
        snapshot.append({
            "title": canvas.title, "content": content,
            "chars": len(content), "tokens": count_tokens(content),
        })
    return snapshot, ""


def _shareable(att) -> bool:
    """Sent (not a composer draft) and not removed by the content safety system."""
    return att.message_id is not None and not att.message.is_redacted


def _resolve_shared_attachments(thread_id, numbers: list[int]) -> tuple[list[dict], str]:
    """Map the orchestrator's attachment numbers to ``[{"id", "number"}, ...]``.

    Returns ``(entries, "")`` or ``([], error_message)``. Any sent attachment of
    the thread may be shared — the same set the main agent can view. An email
    brings the files split out of it along (a child is skipped silently when it
    can't be shared itself). Ids are stored, not numbers: drafts count in the
    numbering and can be removed, which shifts later numbers.
    """
    from chat.services import list_thread_attachments

    wanted = list(dict.fromkeys(int(n) for n in numbers))
    if not wanted:
        return [], ""
    pairs = list_thread_attachments(thread_id)
    by_number = dict(pairs)
    number_of = {att.pk: n for n, att in pairs}

    entries: list[dict] = []
    seen: set = set()

    def _add(n, att):
        if att.pk not in seen:
            seen.add(att.pk)
            entries.append({"id": str(att.pk), "number": n})

    for n in wanted:
        att = by_number.get(n)
        if att is None:
            available = [f"#{m} {a.original_filename}" for m, a in pairs if _shareable(a)]
            listing = ", ".join(available) if available else "(none)"
            return [], f"No attachment #{n}. Attachments in this conversation: {listing}."
        if att.message_id is None:
            return [], (
                f"Attachment #{n} ('{att.original_filename}') hasn't been sent yet — it "
                "is still a draft in the message composer."
            )
        if att.message.is_redacted:
            return [], (
                f"Attachment #{n} ('{att.original_filename}') was removed by the content "
                "safety system and can't be shared."
            )
        _add(n, att)
        children = sorted(
            (a for _, a in pairs if a.parent_id == att.pk),
            key=lambda a: (a.email_ordinal or 0, a.created_at),
        )
        for child in children:
            if _shareable(child):
                _add(number_of[child.pk], child)
    return entries, ""


def _valid_round_budget(name) -> str:
    """The round budget name to store; anything unknown becomes the default."""
    from chat.subagent_service import DEFAULT_ROUND_BUDGET, ROUND_BUDGETS

    return name if name in ROUND_BUDGETS else DEFAULT_ROUND_BUDGET


# --- Input schemas ---

_ROUND_BUDGET_DESCRIPTION = (
    'How many tool rounds the sub-agent gets: "quick" (6 rounds — a few lookups), '
    '"standard" (15 rounds, default), "deep" (30 rounds — broad research). A round '
    "is one reply that calls tools (parallel calls count once)."
)
_MODEL_TIER_DESCRIPTION = (
    'Model quality: "low" for simple, well-defined tasks (lookups, narrow searches, '
    'summaries); "medium" (default) for most tasks; "high" only for tasks that '
    "require exceptional intelligence (rarely relevant)."
)


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
    canvases: list[str] = Field(
        default_factory=list,
        description=(
            "Optional titles of canvases in this conversation to share with the "
            "sub-agent as a read-only snapshot of their current text (up to 5). Use "
            "when the task concerns a draft (review it, fact-check it, research gaps "
            "in it). The sub-agent reads them on demand with its own canvas read "
            "tool, so large canvases are fine. It cannot edit your canvases or see "
            "later changes. Omit when the task doesn't need it."
        ),
    )
    attachments: list[int] = Field(
        default_factory=list,
        description=(
            "Optional numbers of files the user attached to this conversation (from "
            "the '# Attachments' list) to share with the sub-agent. It can view those "
            "files itself — text, PDF pages, presentation slides, images — but no "
            "other attachment. Sharing an email also shares the files attached to it."
        ),
    )
    model_tier: str = Field(default=DEFAULT_SUBAGENT_TIER, description=_MODEL_TIER_DESCRIPTION)
    round_budget: Literal["quick", "standard", "deep"] = Field(
        default="standard", description=_ROUND_BUDGET_DESCRIPTION,
    )


class ContinueSubagentInput(ReasonBaseModel):
    run_id: str = Field(
        description=(
            "The id of the finished sub-agent run to continue — the short id from its "
            '"[Sub-agent result: <id> …]" header (or the full id).'
        ),
    )
    round_budget: Literal["quick", "standard", "deep"] = Field(
        description="Rounds for the continuation. " + _ROUND_BUDGET_DESCRIPTION,
    )
    instructions: str = Field(
        default="",
        description=(
            "Optional direction for the continuation: what to focus on, what to skip, "
            "what was missing. Omit to just let it carry on with its \"Not covered\" list."
        ),
    )
    model_tier: str | None = Field(
        default=None,
        description="Optional model quality; defaults to the previous run's. " + _MODEL_TIER_DESCRIPTION,
    )


# --- Spawning (shared by create and continue) ---

def _load_user(context):
    """``(user, None)`` or ``(None, error_json)`` for the tool's run context."""
    from django.contrib.auth import get_user_model

    if not context or not context.user_id:
        return None, json.dumps({"status": "error", "message": "No user context available."})
    if not context.conversation_id:
        return None, json.dumps({"status": "error", "message": "No thread context available."})
    User = get_user_model()
    try:
        return User.objects.get(pk=context.user_id), None
    except User.DoesNotExist:
        return None, json.dumps({"status": "error", "message": "User not found."})


def _parallel_policy_error(prefs, thread_id) -> str | None:
    """The error JSON when the org runs sub-agents one at a time and one is active."""
    from chat.models import SubAgentRun

    if prefs.parallel_subagents:
        return None
    active = SubAgentRun.objects.filter(
        thread_id=thread_id,
        status__in=[SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING],
    ).exists()
    if active:
        return json.dumps({"status": "error", "message": "Sub-agents must run one at a time. Wait for the current one to complete."})
    return None


def _spawn_run(user, thread_id, **run_kwargs) -> str:
    """Admit, dispatch and announce a new run; return the tool's JSON result.

    Admission goes through the per-user / system caps. The run is handed to
    Celery now if an execution slot is free, else it waits in line. The result
    is always "started" or "queued" (or an admission error): sub-agents run in
    the background and their outcome reaches the orchestrator as a hidden
    message that wakes it.
    """
    from chat.models import SubAgentRun
    from chat.subagent_limits import (
        create_subagent_run_if_allowed,
        dispatch_pending_subagents,
        get_queue_depth,
    )

    run, err_msg = create_subagent_run_if_allowed(user, thread_id=thread_id, **run_kwargs)
    if run is None:
        return json.dumps({"status": "error", "message": err_msg})

    # A dispatcher error is not a spawn error: the row exists and the sweeper
    # backstop will dispatch it.
    try:
        dispatch_pending_subagents()
    except Exception:
        logger.exception("Dispatch after creating sub-agent run %s failed", run.id)
    # Read the outcome from the row, not from this call's return value:
    # sibling spawn calls in the same turn run in parallel threads, and
    # whichever dispatcher runs first hands out every waiting run — including
    # ours — leaving our own call nothing to return.
    started = not _is_waiting(run.id)
    # Show the run in the user's sub-agent panel now (queued or dispatched).
    from chat.tasks import notify_subagent_status

    notify_subagent_status(str(thread_id), str(run.id))

    short_id = str(run.id)[:8]
    if started:
        return json.dumps({
            "status": "started",
            "run_id": str(run.id),
            "message": (
                f"Sub-agent {short_id} started (model quality: {run.model_tier}, round "
                f"budget: {run.round_budget}). Its result will appear in the "
                "conversation automatically."
            ),
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
            f"Sub-agent {short_id} queued (position {ahead + 1}): all {queue['worker_slots']} "
            f"execution slots are busy ({queue['running']} running, {ahead} waiting ahead of it). "
            "It starts automatically when a slot frees up and its result will appear "
            "in the conversation — do not create it again."
        ),
    })


def _spawn_end_label(result: dict) -> str | None:
    return {
        "queued": "Sub-agent queued",
        "started": "Sub-agent dispatched",
        "error": "Sub-agent failed",
    }.get(result.get("status"))


# --- Tools ---

class CreateSubagentTool(ContextAwareTool):
    """Create a sub-agent to handle a delegated task."""

    name: str = "chat_subagent_create"
    audience: str = "main"
    start_label: str = "Running sub-agent..."
    end_label: str = "Sub-agent dispatched"

    def end_label_for_result(self, result: dict) -> str | None:
        return _spawn_end_label(result)

    description: str = (
        "Delegate a task to an independent sub-agent that runs in the background with its "
        "own context and tools. Use for tasks requiring extensive research, parallel "
        "analysis, or focused work. Size round_budget to the task. Pass canvases to give "
        "it a read-only copy of canvas text, and attachments to let it view files the user "
        "attached. Its result arrives in the conversation automatically."
    )
    args_schema: type[BaseModel] = CreateSubagentInput

    def _run(
        self,
        prompt: str,
        model_tier: str = DEFAULT_SUBAGENT_TIER,
        **kwargs,
    ) -> str:
        from core.preferences import get_preferences

        context = self.context
        user, err = _load_user(context)
        if err:
            return err
        thread_id = context.conversation_id

        if model_tier not in SUBAGENT_TIERS:
            model_tier = DEFAULT_SUBAGENT_TIER
        prefs = get_preferences(user)

        # Resolve an optional specialization ("type"). Validate against the
        # user's available specializations and feed an actionable error back to
        # the model (it can retry without a type, or with a valid one) rather
        # than silently ignoring an unknown slug.
        specialization = (kwargs.get("type") or "").strip()
        if specialization:
            # Same safety-scan gate as chat_skill_attach: a specialization the
            # scan hasn't passed is not on offer.
            valid_slugs = [
                s["slug"] for s in prefs.allowed_specializations
                if s.get("approved", True)
            ]
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

        err = _parallel_policy_error(prefs, thread_id)
        if err:
            return err

        # Snapshot shared canvases now, not when the run executes: a queued or
        # retried run must see the text the orchestrator delegated against.
        shared_canvases, err_msg = _snapshot_canvases(thread_id, kwargs.get("canvases") or [])
        if err_msg:
            return json.dumps({"status": "error", "message": err_msg})

        shared_attachments, err_msg = _resolve_shared_attachments(
            thread_id, kwargs.get("attachments") or [],
        )
        if err_msg:
            return json.dumps({"status": "error", "message": err_msg})

        # The service resolves model_used and tool_names when it executes.
        return _spawn_run(
            user, thread_id,
            prompt=prompt,
            reason=str(kwargs.get("reason") or "").strip()[:500],
            model_tier=model_tier,
            round_budget=_valid_round_budget(kwargs.get("round_budget")),
            data_room_ids=context.data_room_ids or [],
            skill_slug=specialization,
            shared_canvases=shared_canvases,
            shared_attachments=shared_attachments,
        )


# A continuation chain (original run + continuations) may hold at most this many
# runs, so a task can't be continued forever.
MAX_CHAIN_LENGTH = 10


def _resolve_run_for_continuation(thread_id, run_id: str):
    """``(run, None)`` or ``(None, error_message)`` for an id within the thread."""
    from django.core.exceptions import ValidationError

    from chat.models import SubAgentRun

    ref = (run_id or "").strip().strip("[]").lower()
    if not ref:
        return None, "Give the id of the sub-agent run to continue."
    runs = SubAgentRun.objects.filter(thread_id=thread_id)
    try:
        if len(ref) == 36:
            match = list(runs.filter(pk=ref)[:2])
        else:
            match = [r for r in runs.only("id") if str(r.id).startswith(ref)][:2]
    except ValidationError:
        match = []
    if not match:
        return None, f"No sub-agent run '{run_id}' in this conversation."
    if len(match) > 1:
        return None, f"'{run_id}' matches more than one sub-agent run; give more of its id."
    return SubAgentRun.objects.get(pk=match[0].pk), None


class ContinueSubagentTool(ContextAwareTool):
    """Continue a finished sub-agent run in a new run that picks up its work."""

    name: str = "chat_subagent_continue"
    audience: str = "main"
    start_label: str = "Continuing sub-agent..."
    end_label: str = "Sub-agent continued"

    def end_label_for_result(self, result: dict) -> str | None:
        status = result.get("status")
        if status in ("started", "queued"):
            return "Sub-agent continued" if status == "started" else "Sub-agent queued"
        return _spawn_end_label(result)

    description: str = (
        "Continue a finished sub-agent's task with a fresh round budget — e.g. when its "
        'result is PARTIAL (its round budget ran out) or its "Not covered" list matters. '
        "A new sub-agent run starts with the same task, specialization, data rooms and "
        "shared files, plus the previous run's working canvas, private notes and final "
        "reply, and picks up where it left off. Its result arrives in the conversation "
        "automatically, like any sub-agent's."
    )
    args_schema: type[BaseModel] = ContinueSubagentInput

    def _run(self, run_id: str, round_budget: str = "standard", **kwargs) -> str:
        from chat.models import SubAgentRun
        from core.preferences import get_preferences

        context = self.context
        user, err = _load_user(context)
        if err:
            return err
        thread_id = context.conversation_id

        previous, err_msg = _resolve_run_for_continuation(thread_id, run_id)
        if err_msg:
            return json.dumps({"status": "error", "message": err_msg})
        if previous.status not in (SubAgentRun.Status.COMPLETED, SubAgentRun.Status.FAILED):
            return json.dumps({
                "status": "error",
                "message": (
                    f"Sub-agent {str(previous.id)[:8]} is still {previous.status}; wait for "
                    "its result before continuing it."
                ),
            })
        if previous.chain_depth + 1 >= MAX_CHAIN_LENGTH:
            return json.dumps({
                "status": "error",
                "message": (
                    f"This task has already been continued {previous.chain_depth} times "
                    "(the limit). Start a new sub-agent with a fresh, narrower task instead."
                ),
            })

        model_tier = kwargs.get("model_tier") or previous.model_tier
        if model_tier not in SUBAGENT_TIERS:
            model_tier = DEFAULT_SUBAGENT_TIER
        prefs = get_preferences(user)
        err = _parallel_policy_error(prefs, thread_id)
        if err:
            return err

        return _spawn_run(
            user, thread_id,
            prompt=previous.prompt,
            reason=(str(kwargs.get("reason") or "").strip() or previous.reason)[:500],
            model_tier=model_tier,
            round_budget=_valid_round_budget(round_budget),
            data_room_ids=previous.data_room_ids or [],
            skill_slug=previous.skill_slug,
            shared_canvases=previous.shared_canvases,
            shared_attachments=previous.shared_attachments,
            canvas=previous.canvas,
            canvas_title=previous.canvas_title,
            scratchpad=previous.scratchpad,
            continued_from=previous,
            chain_depth=previous.chain_depth + 1,
            continuation_instructions=str(kwargs.get("instructions") or "").strip()[:4000],
        )


# Register on import
registry = get_tool_registry()
registry.register_tool(CreateSubagentTool())
registry.register_tool(ContinueSubagentTool())
