"""Core sub-agent execution service."""

from __future__ import annotations

import logging
import uuid

from django.utils import timezone

from chat.services import nest_heading_levels
from core.preferences import MIN_CONTEXT_TOKENS, SUBAGENT_TIER_FEATURES, ResolvedPreferences

logger = logging.getLogger(__name__)

# A sub-agent run is bounded by a round budget the orchestrator picks per task
# (``round_budget`` on chat_subagent_create). A round is one model reply that
# calls tools; the pipeline counts them down to the sub-agent every round
# (llm/pipelines/simple_chat.py) and forces a tool-less final reply after the
# last one. There is no wall-clock limit: hangs are caught by per-tool timeouts
# and the heartbeat sweeper.
ROUND_BUDGETS = {"quick": 10, "standard": 20, "deep": 40}
DEFAULT_ROUND_BUDGET = "standard"
# Sanity clamp on any budget (long runs are the worker's main memory spike).
SUBAGENT_MAX_ROUNDS = 40


def round_budget_rounds(name: str) -> int:
    """Rounds for a budget name; an unknown name gets the default budget."""
    rounds = ROUND_BUDGETS.get(name) or ROUND_BUDGETS[DEFAULT_ROUND_BUDGET]
    return min(rounds, SUBAGENT_MAX_ROUNDS)

def subagent_context_budget(prefs: ResolvedPreferences, tier: str) -> int:
    """Context aim (tokens) for a sub-agent of *tier*.

    The org's per-tier sub-agent budget when set, else the org's context limit
    (default 200k) — never the model's full hard window, which let a long run
    re-send ~1M-token histories every round. Floored at MIN_CONTEXT_TOKENS.
    """
    budget = prefs.subagent_context_budgets.get(tier, 0) or prefs.max_context_tokens
    return max(budget, MIN_CONTEXT_TOKENS)


def is_retryable_subagent_error(exc: BaseException) -> bool:
    """Whether a sub-agent failure is transient and worth retrying the whole run.

    Single source of truth shared by :func:`run_subagent` (which leaves the run
    RUNNING instead of marking it FAILED for these, so a retry can re-enter
    cleanly) and ``chat.tasks.run_subagent_task`` (which calls ``self.retry``).
    Covers transient LLM provider errors (timeout / connection) and low-level
    network errors. Rate limits and overloads are NOT retried here: the provider
    layer already backed off 4 times (30/60/120 s) before raising, and a whole-run
    restart replays every paid round from scratch — the orchestrator gets a
    failure notice instead and can pick another model. A Postgres "too many
    connections" error is treated as terminal — retrying immediately would just
    hit the cap again.
    """
    from django.db.utils import OperationalError
    from llm.service.errors import LLMConnectionError, LLMTimeoutError

    if isinstance(exc, OperationalError):
        return "too many connections" not in str(exc).lower()
    return isinstance(exc, (
        LLMTimeoutError,
        LLMConnectionError,
        ConnectionError,
        TimeoutError,
        OSError,
    ))


_PANEL_PROMPT_FALLBACK_CHARS = 120


def active_subagent_runs_payload(thread_id) -> dict:
    """The thread's active sub-agent runs for the user's sub-agent panel.

    Same scope as the consumer's active/waiting counts: PENDING or RUNNING runs
    of the thread; a PENDING run not yet handed to Celery (``dispatched_at``
    NULL) is "queued". Returns ``{"runs": [...], "server_now": iso}`` —
    ``server_now`` lets the browser correct its clock before counting each
    running row's timer up from ``started_at``. Sync (DB); oldest first.
    """
    from chat.models import SubAgentRun

    qs = (
        SubAgentRun.objects.filter(
            thread_id=thread_id,
            status__in=[SubAgentRun.Status.PENDING, SubAgentRun.Status.RUNNING],
        )
        .order_by("created_at")
        .values(
            "id", "status", "reason", "prompt", "created_at", "dispatched_at",
            "started_at", "chain_depth",
        )
    )
    runs = []
    for r in qs:
        queued = r["status"] == SubAgentRun.Status.PENDING and r["dispatched_at"] is None
        reason = (r["reason"] or "").strip()
        if not reason:
            prompt = " ".join((r["prompt"] or "").split())
            reason = prompt[:_PANEL_PROMPT_FALLBACK_CHARS] + (
                "…" if len(prompt) > _PANEL_PROMPT_FALLBACK_CHARS else ""
            )
        runs.append({
            "id": str(r["id"]),
            "reason": reason,
            "state": "queued" if queued else "running",
            # Dispatched but not yet picked up by the worker: count from dispatch.
            "started_at": (
                None if queued
                else (r["started_at"] or r["dispatched_at"] or r["created_at"]).isoformat()
            ),
            "created_at": r["created_at"].isoformat(),
            # >0 for a chat_subagent_continue run: the panel marks it "continued".
            "continuation": r["chain_depth"],
        })
    return {"runs": runs, "server_now": timezone.now().isoformat()}


def resolve_subagent_model(tier: str, prefs: ResolvedPreferences) -> str:
    """Map a tier name to the model configured for that sub-agent tier.

    Honors the org's per-tier override (the ``subagent_low`` / ``subagent_medium``
    / ``subagent_high`` feature models), falling back to the tier's slot model
    (``cheap_model`` / ``medium_model`` / ``top_model``) when no override is set —
    which is also what ``feature_models`` resolves to by default. The slot
    fallback also covers callers that build a bare ``ResolvedPreferences``
    without the subagent feature keys. An unknown tier resolves as medium.
    """
    feature = SUBAGENT_TIER_FEATURES.get(tier)
    if feature:
        model = prefs.feature_models.get(feature)
        if model:
            return model
    slot_model = {"low": prefs.cheap_model, "medium": prefs.medium_model, "high": prefs.top_model}
    return slot_model.get(tier) or prefs.medium_model


def specialization_model(skill, prefs: ResolvedPreferences) -> tuple[str | None, str | None]:
    """(model, reasoning level) a specialization always runs on, or (None, None).

    A specialization naming an Automatic category (chat/auto_model.py) runs on
    the org's model for that category — with the skill's own reasoning level
    when that model offers it, else the category's — whatever ``model_tier``
    the orchestrator asked for.
    """
    from chat.auto_model import CATEGORIES_BY_KEY, resolve_category

    category = getattr(skill, "preferred_model_category", "") if skill is not None else ""
    if category not in CATEGORIES_BY_KEY:
        return None, None
    return resolve_category(prefs, category, getattr(skill, "preferred_reasoning_level", "") or None)


def subagent_thinking_level(prefs: ResolvedPreferences, tier: str, model: str) -> str | None:
    """The org's reasoning level for a sub-agent of *tier* running on *model*.

    A level is stored with the model it was chosen for and only applies while
    the tier still runs on that exact model — any model change (override, slot
    default, allow-list fallback, retirement) reverts to the registry default.
    Returns None for "use the model's default".
    """
    from llm.model_registry import get_model_info

    entry = prefs.subagent_reasoning.get(tier)
    if not entry or entry.get("model") != model:
        return None
    info = get_model_info(model)
    level = entry.get("level")
    if not info or level not in info.reasoning_levels:
        return None
    return level


def resolve_subagent_tools(
    prefs: ResolvedPreferences,
    data_room_ids: list[int],
    specialization_slug: str = "",
    has_shared_attachments: bool = False,
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
    from chat.tool_groups import DATA_ROOM_TOOL_NAMES

    tools = list(prefs.allowed_subagent_tools)
    spec = next(
        (s for s in prefs.allowed_specializations if s["slug"] == specialization_slug),
        None,
    ) if specialization_slug else None

    if spec:
        for t in spec["tool_names"]:
            if t not in tools:
                tools.append(t)

    # The template loader resolves against the run's specialization. Hide it
    # from general agents and from runs whose specialization was revoked after
    # they were queued.
    if spec is None:
        tools = [t for t in tools if t != "subagent_canvas_load_template"]

    # Apply context gates after specialization tools are merged so a custom
    # specialization cannot reintroduce tools that are unusable without rooms.
    if not data_room_ids:
        tools = [t for t in tools if t not in DATA_ROOM_TOOL_NAMES]
    # Like chat_attachment_view on the main agent: only when there is a file to view.
    if not has_shared_attachments:
        tools = [t for t in tools if t != "subagent_attachment_view"]

    return tools


def shared_attachment_rows(run) -> list[tuple[int, object]]:
    """``[(number, ChatAttachment), ...]`` for the files shared with *run*, in
    the order shared. Skips files deleted or redacted since the spawn."""
    from chat.models import ChatAttachment

    entries = run.shared_attachments or []
    if not entries:
        return []
    rows = {
        str(a.pk): a
        for a in ChatAttachment.objects.select_related("message").filter(
            thread_id=run.thread_id, pk__in=[e["id"] for e in entries],
        )
    }
    out = []
    for e in entries:
        att = rows.get(e["id"])
        if att is None or att.message_id is None or att.message.is_redacted:
            continue
        out.append((e["number"], att))
    return out


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


def _is_untouched_template(run) -> bool:
    """Whether the run's canvas is still exactly its specialization's auto-loaded
    template — i.e. the sub-agent never wrote to it, so it is not a deliverable."""
    if not run.canvas or not run.skill_slug:
        return False
    skill = get_run_specialization_skill(run)
    first = skill.templates.all().first() if skill is not None else None
    if first is None:
        return False
    from chat.services import CANVAS_MAX_CHARS

    return run.canvas == (first.content or "")[:CANVAS_MAX_CHARS]


def report_source(run, *, untouched: bool | None = None) -> tuple[str, str]:
    """``(text, source)`` of a run's deliverable: its working canvas
    (``"canvas"``), or its final message (``"final_message"``) when it left no
    canvas or only the untouched specialization template. ``untouched`` is the
    result of ``_is_untouched_template(run)`` when the caller already has it (two
    queries a finishing run would otherwise repeat); None computes it here."""
    if run.canvas:
        if untouched is None:
            untouched = _is_untouched_template(run)
        if not untouched:
            return run.canvas, "canvas"
    return run.result or "", "final_message"


def _cap_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """``(text, was_cut)``: *text* cut to about *max_tokens*, at a line or word
    boundary when one is near."""
    from core.tokens import count_tokens

    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text, False
    cut = text[: max(1, int(len(text) * max_tokens / tokens))]
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary > len(cut) * 0.8:
        cut = cut[:boundary]
    return cut.rstrip(), True


def _subagent_result_message_exists(run) -> bool:
    """Whether a hidden result message already exists for this run."""
    from chat.models import ChatMessage

    return ChatMessage.objects.filter(
        thread_id=run.thread_id,
        metadata__source="subagent",
        metadata__subagent_run_id=str(run.id),
    ).exists()


def _format_elapsed(run) -> str:
    """``m:ss`` (or ``h:mm:ss``) from start to finish, ``""`` when unknown."""
    if not run.started_at or not run.completed_at:
        return ""
    s = max(0, int((run.completed_at - run.started_at).total_seconds()))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def run_progress(run) -> str:
    """How far the run got, e.g. ``"9/15 rounds (standard), 4:12"``."""
    budget = round_budget_rounds(run.round_budget)
    used = min(run.rounds_used or 0, budget)
    parts = [f"{used}/{budget} rounds ({run.round_budget})"] if run.rounds_used else []
    elapsed = _format_elapsed(run)
    if elapsed:
        parts.append(elapsed)
    return ", ".join(parts)


def forced_reply_reason(run) -> str:
    """Why the run's tool-less final reply was forced, or "" when it answered
    on its own: its round budget ran out, or its output was cut off (output
    cap / repetition guard) twice in a row so the loop gave up on tools."""
    if run.budget_exhausted:
        return "round budget reached"
    if run.stop_reason == "truncation":
        return "its output was cut off repeatedly"
    return ""


def result_header(run) -> str:
    """The header of a sub-agent's result message.

    Says how the run went (rounds used of its budget, time taken) and how to
    continue it. A run whose tool-less final reply was forced (budget ran out,
    or repeatedly cut-off output — ``forced_reply_reason``) says so neutrally:
    the forced reply may still be complete, so the orchestrator checks it.
    """
    short_id = str(run.id)[:8]
    progress = run_progress(run)
    continuation = f" (continuation {run.chain_depth})" if run.chain_depth else ""
    forced = forced_reply_reason(run)
    if forced:
        return (
            f"[Sub-agent result: {short_id}{continuation} — {forced}: its final reply was "
            f"forced{f' after {progress}' if progress else ''}. Check its reply for anything "
            "it did not cover; if more work is needed, consider continuing it.]"
        )
    return (
        f"[Sub-agent result: {short_id}{continuation} — finished"
        f"{f' ({progress})' if progress else ''}. If more work is needed, you may continue "
        f'it with chat_subagent_continue(run_id="{short_id}").]'
    )


def build_result_notice(run, *, untouched: bool | None = None) -> str:
    """The orchestrator-facing result message: a pointer to the report, never
    the report itself.

    The report (working canvas) is read with ``chat_subagent_report_read``, so
    it lives in tool results that age into re-read stubs instead of staying in
    the history in full (where nothing prunes a user message). The sub-agent's
    final message rides along inline, capped at
    ``SUBAGENT_RESULT_INLINE_MAX_TOKENS``; a longer one is read in full with
    the same tool when there is no canvas."""
    from django.conf import settings

    from core.tokens import count_tokens

    short_id = str(run.id)[:8]
    read_call = f'chat_subagent_report_read(run_id="{short_id}")'
    cap = int(getattr(settings, "SUBAGENT_RESULT_INLINE_MAX_TOKENS", 2_000))
    text, source = report_source(run, untouched=untouched)
    # The notice has no heading of its own; keep a report-style "# Findings"
    # from reading as a top-level section of the message.
    final = nest_heading_levels((run.result or "").strip(), 1)

    lines = [result_header(run)]
    if source == "canvas":
        title = run.canvas_title or "Working document"
        line_count = text.count("\n") + 1  # read_canvas_range's line numbering
        lines.append(
            f'Report: canvas "{title}" (~{count_tokens(text):,} tokens, '
            f"{line_count:,} lines). Read it in full with {read_call} "
            "before you rely on its findings."
        )
        if final:
            shown, cut = _cap_tokens(final, cap)
            lines.append(f"Accompanying message:\n{shown}")
            if cut:
                lines.append(
                    f"[Accompanying message cut at ~{cap:,} tokens; the report holds the findings.]"
                )
        else:
            lines.append("The sub-agent left no accompanying message.")
    elif final:
        shown, cut = _cap_tokens(final, cap)
        lines.append(shown)
        if cut:
            lines.append(
                f"[Final message cut at ~{cap:,} tokens of ~{count_tokens(final):,}; "
                f"read it in full with {read_call}.]"
            )
    else:
        lines.append("The sub-agent left no final message and no report.")
    return "\n".join(lines)


def _create_subagent_result_message(run, *, untouched: bool | None = None) -> None:
    """Persist the sub-agent's result notice as a hidden ChatMessage.

    Hidden, ``role="user"`` (not ``"tool"``, to avoid orphan tool_call_ids that
    OpenAI rejects). Content is ``build_result_notice``: a pointer to the report
    plus the capped final message. Survives reconnects and failed LLM streams.
    ``untouched`` is passed through to ``report_source``.
    """
    from chat.models import ChatMessage
    from core.tokens import count_tokens

    wrapped = build_result_notice(run, untouched=untouched)
    ChatMessage.objects.create(
        thread_id=run.thread_id,
        role="user",
        content=wrapped,
        metadata={"source": "subagent", "subagent_run_id": str(run.id)},
        token_count=count_tokens(wrapped),
        is_hidden_from_user=True,
    )


def is_capacity_error(exc: BaseException) -> bool:
    """A provider rate limit / overload that outlasted the provider's retries."""
    from llm.service.errors import LLMOverloadedError, LLMRateLimitError

    return isinstance(exc, (LLMRateLimitError, LLMOverloadedError))


def _create_subagent_failure_message(run, *, capacity: bool = False) -> None:
    """Persist a terminal sub-agent failure as a hidden ChatMessage.

    Mirrors :func:`_create_subagent_result_message` so the orchestrator's
    unreported-claim logic picks the failure up and seeds a continuation turn —
    without this, a run that fails with no result and no canvas is invisible:
    the status bar decrements but the assistant never reacts and the thread
    goes silent. Guarded against duplicates (the terminal-failure canvas path
    may already have written a result message for the same run). Cancelled
    runs never reach this — every caller writes only for rows it newly
    transitioned to FAILED, and a user cancel flips the row to FAILED first.
    """
    from chat.models import ChatMessage
    from core.tokens import count_tokens

    if _subagent_result_message_exists(run):
        return
    from llm.model_registry import get_model_info

    short_id = str(run.id)[:8]
    progress = run_progress(run)
    info = get_model_info(run.model_used) if run.model_used else None
    model_name = info.display_name if info else (run.model_used or "unknown")
    advice = (
        "This is a provider capacity problem (rate limit / overload): a new "
        "sub-agent on the same model will likely fail too. Use a different "
        "model_tier, do the work yourself, or tell the user what happened."
        if capacity else
        "Decide how to proceed: retry with a new sub-agent, do the work "
        "yourself, or tell the user what happened."
    )
    wrapped = (
        f"[Sub-agent failed: {short_id}{f' ({progress})' if progress else ''}]\n"
        f"The sub-agent did not complete its task. Error: {run.error or 'unknown'}\n"
        f"Model: {model_name} ({run.model_tier or 'default'} tier)\n"
        f"{advice}"
    )
    ChatMessage.objects.create(
        thread_id=run.thread_id,
        role="user",
        content=wrapped,
        metadata={"source": "subagent", "subagent_run_id": str(run.id)},
        token_count=count_tokens(wrapped),
        is_hidden_from_user=True,
    )


# At most one heartbeat write per run per this many seconds.
HEARTBEAT_MIN_INTERVAL_SECONDS = 15


def _heartbeat(run_id, rounds=None):
    """A throttled ``RunContext.on_activity`` callback that stamps the run's
    ``last_activity_at`` — and, when ``rounds`` (a callable returning the
    loop's round count so far) is given, ``rounds_used``, so the count is live
    while the run works instead of appearing only at completion. Called from
    the run's thread and from tool worker threads; a worker thread closes the
    DB connection it opened (a stray connection per thread would eat into the
    Postgres/PgBouncer cap)."""
    import threading
    import time

    from django.db import connection

    owner = threading.get_ident()
    lock = threading.Lock()
    last = [time.monotonic()]  # run start already stamped the row

    def beat():
        now = time.monotonic()
        with lock:
            if now - last[0] < HEARTBEAT_MIN_INTERVAL_SECONDS:
                return
            last[0] = now
        from chat.models import SubAgentRun

        fields = {"last_activity_at": timezone.now()}
        if rounds is not None:
            used = rounds()
            if used:
                fields["rounds_used"] = used
        try:
            SubAgentRun.objects.filter(
                pk=run_id, status=SubAgentRun.Status.RUNNING,
            ).update(**fields)
        except Exception:
            logger.debug("Sub-agent heartbeat for run %s failed", run_id, exc_info=True)
        finally:
            if threading.get_ident() != owner:
                connection.close()

    return beat


def task_message(run) -> str:
    """The sub-agent's task message. For a continuation, the original task is
    kept under its own heading and followed by a separate block on the earlier
    run: how it went, its final reply ("Not covered" list included) and the
    orchestrator's direction for this continuation. The canvas and scratchpad
    themselves were carried over on the row."""
    previous = run.continued_from
    if previous is None:
        return run.prompt
    if previous.budget_exhausted:
        outcome = "ran out of rounds (its reply was forced)"
    elif previous.stop_reason == "truncation":
        outcome = "had its output cut off repeatedly (its reply was forced)"
    else:
        outcome = "finished"
    progress = run_progress(previous)
    lines = [
        "# Original task",
        nest_heading_levels(run.prompt, 1),
        "",
        "# Continuing earlier work",
        f"This is a continuation of the original task above. A previous sub-agent {outcome}"
        f"{f' after {progress}' if progress else ''}. Its working canvas and scratchpad were "
        "carried over to you as a starting point. Continue where it left off.",
    ]
    if previous.result:
        lines += ["", "## Its final reply", nest_heading_levels(previous.result.strip(), 2)]
    if run.continuation_instructions:
        lines += [
            "", "## Direction from the orchestrator",
            nest_heading_levels(run.continuation_instructions, 2),
        ]
    return "\n".join(lines)


def run_subagent(run_id: uuid.UUID) -> None:
    """Execute a sub-agent run. Called by the Celery task.

    Bounded by the run's round budget (no wall clock): the tool loop counts the
    rounds down to the sub-agent and forces a tool-less final reply after the
    last one, which marks the run ``budget_exhausted``.
    """
    from chat.models import SubAgentRun
    from chat.services import attachment_kind_label
    from chat.subagent_prompts import build_subagent_system_prompt
    from core.preferences import get_preferences
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    run = SubAgentRun.objects.select_related("user", "thread").get(pk=run_id)

    try:
        # Guarded transition: only (re-)enter from a non-terminal state. PENDING
        # is the first run; RUNNING is a legitimate Celery retry after a retryable
        # mid-run error (the service leaves the row RUNNING). A COMPLETED row must
        # NOT be resurrected — a retry triggered by a transient error AFTER
        # completion (e.g. persisting the result message) would otherwise flip it
        # back to RUNNING and re-run the whole LLM job. FAILED is cancelled/terminal.
        started_at = timezone.now()
        started = SubAgentRun.objects.filter(pk=run_id).exclude(
            status__in=[SubAgentRun.Status.FAILED, SubAgentRun.Status.COMPLETED],
        ).update(
            status=SubAgentRun.Status.RUNNING, started_at=started_at,
            last_activity_at=started_at,
        )
        if not started:
            logger.info("Sub-agent run %s already terminal; skipping re-entry", run_id)
            return
        run.status, run.started_at = SubAgentRun.Status.RUNNING, started_at
        # The user's sub-agent panel flips this row from "Queued" to a live timer.
        from chat.tasks import notify_subagent_status

        notify_subagent_status(str(run.thread_id), str(run_id))

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
        # A specialization with a model category overrides the requested tier.
        spec_model, spec_level = specialization_model(specialization_skill, prefs)
        if spec_model:
            model = spec_model
            run.model_used = model

        # Auto-load the first template (name-ordered, matching the prompt's
        # bullet list) into the working canvas as a starting point. Persisted
        # immediately so the canvas is durable from the very start of the run.
        # A canvas that already has content — carried over by a continuation,
        # or written before a Celery retry — is kept.
        if specialization_skill is not None and not run.canvas:
            first_tmpl = specialization_skill.templates.all().first()
            if first_tmpl is not None:
                from chat.services import CANVAS_MAX_CHARS

                run.canvas = (first_tmpl.content or "")[:CANVAS_MAX_CHARS]
                run.canvas_title = first_tmpl.name[:255]
                run.save(update_fields=["canvas", "canvas_title"])

        # Resolve tools
        data_room_ids = run.data_room_ids or []
        shared_rows = shared_attachment_rows(run)
        tool_list = resolve_subagent_tools(
            prefs, data_room_ids, specialization_slug=run.skill_slug,
            has_shared_attachments=bool(shared_rows),
        )
        run.tool_names = tool_list
        run.save(update_fields=["model_used", "tool_names"])

        # Get org name for prompt (memoized on the user instance — shared with the
        # preference/skill resolvers above so it's one query per run).
        from accounts.models import get_membership
        membership = get_membership(user)
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

        # The run's own plan (a continuation carries the previous run's over);
        # never the thread's ThreadTask rows — those are the orchestrator's.
        thread_tasks = list(run.task_plan or [])

        has_task_tool = "subagent_task_update" in (tool_list or [])
        # Context aim for this run: the org's per-tier budget, else its context
        # limit. It sizes the mid-turn pruner AND the per-read cap of
        # subagent_canvas_read (via context.max_context_tokens).
        from llm.context_budget import canvas_read_token_cap

        budget = subagent_context_budget(prefs, run.model_tier)
        rounds = round_budget_rounds(run.round_budget)
        system_prompt = build_subagent_system_prompt(
            data_rooms=data_rooms_info,
            organization_name=org_name,
            tasks=thread_tasks if thread_tasks else None,
            has_task_tool=has_task_tool,
            specialization_skill=specialization_skill,
            canvas_content=run.canvas,
            canvas_title=run.canvas_title,
            canvas_token_cap=canvas_read_token_cap(budget),
            shared_canvases=run.shared_canvases,
            shared_attachments=[
                {
                    "number": n,
                    "filename": att.original_filename,
                    "kind_label": attachment_kind_label(att),
                }
                for n, att in shared_rows
            ],
            round_budget=rounds,
        )

        # Build LLM request
        context = RunContext.create(
            user_id=user.pk,
            conversation_id=str(run.thread_id),
            data_room_ids=data_room_ids,
        )
        context.run_id = str(run_id)
        context.agent_kind = "subagent"
        # Lets subagent_attachment_view check the model takes images/PDFs natively.
        context.model_id = model
        # subagent_canvas_read derives its per-read page cap from the run's aim.
        context.max_context_tokens = budget
        # Seed the in-memory scratchpad so a resumed run re-reads its prior notes;
        # the tool loop shows context.scratchpad once up front and again at every
        # edit point (llm/pipelines/edit_points.py).
        context.scratchpad = run.scratchpad or ""
        # Heartbeat for the stale-run sweeper (there is no wall-clock limit);
        # also keeps rounds_used current while the run works.
        context.on_activity = _heartbeat(
            run_id, rounds=lambda: context.observability.get("rounds"),
        )
        # Reuse this one (already memoized) User instance for every tool call in the
        # run, so UserSettings/Membership are read once instead of per tool. Set on
        # this single thread before the streaming tool loop spawns its workers.
        context._cached_user = user

        # Cooperative cancellation: check if the run has been marked FAILED
        def _is_cancelled():
            return SubAgentRun.objects.filter(pk=run_id, status=SubAgentRun.Status.FAILED).exists()

        # Context aim the mid-turn pruner (llm/pipelines/edit_points.py) sizes to.
        params = {
            "_cancel_check": _is_cancelled,
            "max_tool_iterations": rounds,
            "max_context_tokens": budget,
        }
        # Org-chosen reasoning level for this tier; absent = registry default
        # (LLMService._apply_default_reasoning).
        thinking_level = (
            spec_level if spec_model else subagent_thinking_level(prefs, run.model_tier, model)
        )
        if thinking_level:
            params["thinking_level"] = thinking_level

        request = ChatRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=task_message(run)),
            ],
            model=model,
            stream=True,
            tools=tool_list if tool_list else None,
            context=context,
            params=params,
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
        obs = context.observability
        run.rounds_used = obs.get("rounds")
        run.stop_reason = obs.get("loop_stop") or ""
        run.budget_exhausted = run.stop_reason == "budget"
        # A populated canvas is a deliverable in its own right, so a run that
        # built one but emitted no final text still counts as completed — but
        # not the specialization's template the sub-agent never touched.
        # Decided once (two queries) and reused for the result message.
        untouched = _is_untouched_template(run)
        delivered = bool(run.result) or (bool(run.canvas) and not untouched)
        if delivered:
            run.status = SubAgentRun.Status.COMPLETED
        elif _is_cancelled():
            # An empty finish because the user stopped the run early (often 0
            # tokens) is expected, not a failure to report to Sentry (WILFRED-8E).
            run.status = SubAgentRun.Status.FAILED
            run.error = "Cancelled by user."
            logger.info("Sub-agent run %s cancelled before producing output", run_id)
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
            rounds_used=run.rounds_used,
            stop_reason=run.stop_reason,
            budget_exhausted=run.budget_exhausted,
            status=run.status,
            error=run.error,
            completed_at=run.completed_at,
        )
        if not finished:
            logger.info(
                "Sub-agent run %s was cancelled mid-run; discarding its result", run_id,
            )
            return

        # Persist the outcome as a hidden ChatMessage so it survives across
        # reconnects and failed LLM streams — a no-content failure gets a
        # failure message so the orchestrator still reacts to it.
        if delivered:
            _create_subagent_result_message(run, untouched=untouched)
        else:
            _create_subagent_failure_message(run)

        # Notify the WebSocket consumer so it auto-triggers the orchestrator.
        # _notify_consumer retries once internally and never raises.
        from chat.tasks import _notify_consumer

        _notify_consumer(str(run.id), str(run.thread_id))

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
        capacity = is_capacity_error(exc)
        if capacity:
            # The provider already reported "retries exhausted" to Sentry.
            logger.warning(
                "Sub-agent run %s failed on provider capacity (%s)", run_id, type(exc).__name__,
            )
        else:
            logger.exception("Sub-agent run %s failed", run_id)
        # Guarded: keep an earlier failure reason (e.g. "Cancelled by user.")
        # instead of clobbering it with this exception's message. The row count
        # doubles as the "we ended it, not a user cancel" test below.
        finished = SubAgentRun.objects.filter(pk=run_id).exclude(
            status=SubAgentRun.Status.FAILED,
        ).update(
            status=SubAgentRun.Status.FAILED,
            error=str(exc),
            completed_at=timezone.now(),
        )
        if finished:
            # Durability: a terminal (non-cancel) failure shouldn't discard a
            # canvas the sub-agent already built — surface it as the outcome;
            # otherwise a failure message so the orchestrator still reacts.
            # Status flips FIRST: the notified consumer's claim only looks at
            # terminal rows, so notifying before the update left this outcome
            # for the watchdog. A cancelled run is already FAILED and gets
            # nothing here; both writers are deduplicated by the hidden-message
            # guard. (Celery's on_failure is a no-op once the row is FAILED.)
            try:
                fresh = SubAgentRun.objects.get(pk=run_id)
                fresh_untouched = _is_untouched_template(fresh)
                if (
                    fresh.canvas and not fresh_untouched
                    and not _subagent_result_message_exists(fresh)
                ):
                    _create_subagent_result_message(fresh, untouched=fresh_untouched)
                else:
                    _create_subagent_failure_message(fresh, capacity=capacity)
                from chat.tasks import _notify_consumer

                _notify_consumer(str(fresh.id), str(fresh.thread_id))
            except Exception:
                logger.warning(
                    "Failed to surface outcome for terminally-failed sub-agent run %s",
                    run_id, exc_info=True,
                )
        raise
