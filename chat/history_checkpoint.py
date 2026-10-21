"""Between-turn history: conversation first, saved stubs, checkpoints.

A thread's history is ``[summary] + every message after the summary cutoff``
(``ChatThread.summary_up_to_message_id``). Between checkpoints it is
append-only: each turn's request starts with the previous turn's history byte
for byte, so the provider's prompt cache keeps hitting (Anthropic caches the
prefix; a sliding window rewrote it every turn — cache-rate drop, 2026-10-20).

**Saved edits.** Whatever the tool loop rewrote in earlier messages (a pruning
stub, a dedup placeholder, trimmed call arguments, a dropped call) is saved on
the row's metadata (``CTX_*`` keys below, ``persist_context_edit``) and replayed
by later turns, so a stub stays a stub. ``content`` stays raw: the chat UI and
branch copies read it.

**Budgets** (``llm.context_budget.history_budgets``). The conversation — the
summary plus user/assistant messages, i.e. everything but tool noise — has first
claim on its own share of the history budget. Tool output gets the rest.

**A checkpoint** is the one deliberate rewrite, with three independent actions:

1. *summarise* — only when the conversation overflows its budget (or on
   ``/compact``): everything before the last ``CONTEXT_CHECKPOINT_KEEP_ROUNDS``
   user rounds is folded into the summary and the cutoff moves past it. The
   folded tool calls are listed under the summary (``TOOL_INDEX_HEADER``).
2. *stub* — when the history passes ``CONTEXT_CHECKPOINT_FRACTION`` of its
   budget: every tool result before the kept rounds becomes a saved stub (big
   call arguments are trimmed too). The conversation stays raw.
3. *drop* — when saved stubs pass the stub cap: the oldest stubbed calls go
   entirely until the stubs are at ``1 − CONTEXT_STUB_DROP_FRACTION`` of it.

It runs after a turn (``ChatConsumer._schedule_checkpoint``) and, as a backstop,
synchronously at turn start when the history no longer fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.conf import settings

logger = logging.getLogger(__name__)

# Row metadata: the replayed body of a tool result, its token count, and why.
CTX_BODY = "wf_ctx_body"
CTX_TOKENS = "wf_ctx_tokens"
CTX_REASON = "wf_ctx_reason"
# Row metadata: the row is no longer replayed (a call dropped by the stub cap).
CTX_DROPPED = "wf_ctx_dropped"
# Assistant-row metadata: the replayed tool_calls list (arguments trimmed).
CTX_CALLS = "wf_ctx_tool_calls"

# Heads the folded-call index appended to the stored summary. Everything after it
# is the index (not passed to the summariser, rebuilt at each summarisation).
TOOL_INDEX_HEADER = "## Earlier tool calls (results cleared; call the tool again if you need one)"


def checkpoint_fraction() -> float:
    return float(getattr(settings, "CONTEXT_CHECKPOINT_FRACTION", 0.7))


def keep_rounds_setting() -> int:
    return int(getattr(settings, "CONTEXT_CHECKPOINT_KEEP_ROUNDS", 2))


def is_tool_noise(msg) -> bool:
    """Tool results and the hidden assistant messages that only request them."""
    if msg.role == "tool":
        return True
    meta = msg.metadata or {}
    return (
        msg.role == "assistant"
        and getattr(msg, "is_hidden_from_user", False)
        and bool(meta.get("tool_calls"))
    )


def starts_round(msg) -> bool:
    """A user round starts at a message the user sent: visible, user role.
    Hidden user-role rows (sub-agent notices, loop/seed markers) don't."""
    return msg.role == "user" and not getattr(msg, "is_hidden_from_user", False)


def is_dropped(msg) -> bool:
    return bool((msg.metadata or {}).get(CTX_DROPPED))


def effective_tokens(msg) -> int:
    """Tokens the row costs when replayed (saved body, 0 when dropped)."""
    meta = msg.metadata or {}
    if meta.get(CTX_DROPPED):
        return 0
    if CTX_TOKENS in meta:
        return int(meta[CTX_TOKENS] or 0)
    return int(msg.token_count or 0)


def replayed_body(msg) -> str:
    return (msg.metadata or {}).get(CTX_BODY, msg.content or "")


def replayed_calls(msg) -> list:
    meta = msg.metadata or {}
    return meta.get(CTX_CALLS) or meta.get("tool_calls") or []


def _is_stub(body) -> bool:
    from llm.pipelines.edit_points import is_stub

    return is_stub(body)


def unsummarised_messages_qs(thread):
    """Messages after the summary cutoff, the ones the history replays."""
    from chat.models import ChatMessage

    qs = (
        ChatMessage.objects.filter(thread=thread)
        .exclude(is_redacted=True)
        .exclude(metadata__has_key="ui_only")
    )
    if thread.summary_up_to_message_id:
        cutoff = ChatMessage.objects.filter(id=thread.summary_up_to_message_id).first()
        if cutoff is not None:
            qs = qs.filter(created_at__gt=cutoff.created_at)
    return qs


@dataclass
class HistoryUsage:
    history: int        # summary + every replayed row
    conversation: int   # summary + rows that are not tool noise
    stubs: int          # saved stub bodies


def history_usage(thread, msgs=None) -> HistoryUsage:
    """Token use of the replayed history (estimate units). Refreshes the
    thread's summary fields first unless *msgs* is given."""
    if msgs is None:
        thread.refresh_from_db(fields=["summary", "summary_token_count", "summary_up_to_message_id"])
        msgs = list(unsummarised_messages_qs(thread))
    summary = thread.summary_token_count or 0
    history = conversation = stubs = 0
    for m in msgs:
        tokens = effective_tokens(m)
        history += tokens
        if not is_tool_noise(m):
            conversation += tokens
        elif m.role == "tool" and not is_dropped(m) and _is_stub(replayed_body(m)):
            stubs += tokens
    return HistoryUsage(history=summary + history, conversation=summary + conversation, stubs=stubs)


def history_tokens(thread) -> int:
    return history_usage(thread).history


def _budgets(budgets):
    from llm.context_budget import HistoryBudgets

    if budgets is None or isinstance(budgets, HistoryBudgets):
        return budgets
    return HistoryBudgets(**{k: int(budgets[k]) for k in ("history", "conversation", "summary_target", "stub_cap")})


def checkpoint_due(usage: HistoryUsage, budgets) -> bool:
    """Any of the three actions could apply (the plan decides what is possible)."""
    budgets = _budgets(budgets)
    if budgets is None:
        return False
    return (
        usage.conversation > budgets.conversation
        or usage.history > checkpoint_fraction() * budgets.history
        or usage.stubs > budgets.stub_cap
    )


def needs_checkpoint(thread, budgets) -> bool:
    return checkpoint_due(history_usage(thread), budgets)


# -- saved edits ---------------------------------------------------------------


def persist_context_edit(thread_id, edit: dict) -> int:
    """Save a tool-loop ``context_edit`` (``llm.pipelines.edit_points.
    diff_context_edits``) on the stored rows: new result bodies, trimmed call
    arguments, dropped calls. Returns the number of rows written."""
    from chat.models import ChatMessage, ChatThread
    from core.tokens import count_tokens

    results = edit.get("results") or {}
    calls = edit.get("calls") or {}
    dropped = set(edit.get("dropped") or ())
    changed = []
    tool_ids = set(results) | dropped
    if tool_ids:
        for row in ChatMessage.objects.filter(
            thread_id=thread_id, role="tool", tool_call_id__in=tool_ids,
        ):
            meta = dict(row.metadata or {})
            if row.tool_call_id in dropped:
                meta[CTX_DROPPED] = True
                meta[CTX_REASON] = "cap"
            else:
                body = results[row.tool_call_id]
                meta[CTX_BODY] = body
                meta[CTX_TOKENS] = count_tokens(body)
                meta[CTX_REASON] = "midturn"
            row.metadata = meta
            changed.append(row)
    if calls or dropped:
        thread = ChatThread.objects.filter(pk=thread_id).first()
        if thread is not None:
            for row in unsummarised_messages_qs(thread).filter(
                role="assistant", metadata__has_key="tool_calls",
            ):
                current = replayed_calls(row)
                ids = [c.get("id") for c in current]
                if not any(cid in calls or cid in dropped for cid in ids):
                    continue
                meta = dict(row.metadata or {})
                meta[CTX_CALLS] = [
                    {**c, "arguments": calls[c.get("id")]} if c.get("id") in calls else c
                    for c in current
                ]
                if ids and all(cid in dropped for cid in ids):
                    meta[CTX_DROPPED] = True
                row.metadata = meta
                changed.append(row)
    if changed:
        ChatMessage.objects.bulk_update(changed, ["metadata"])
    return len(changed)


# -- checkpoint ----------------------------------------------------------------


def split_summary(text: str) -> tuple[str, list[str]]:
    """``(narrative, index_lines)`` of a stored summary."""
    text = text or ""
    head, sep, tail = text.partition(TOOL_INDEX_HEADER)
    if not sep:
        return text.strip(), []
    return head.strip(), [ln for ln in tail.strip().splitlines() if ln.strip()]


def join_summary(narrative: str, index_lines: list[str]) -> str:
    if not index_lines:
        return narrative
    return f"{narrative}\n\n{TOOL_INDEX_HEADER}\n" + "\n".join(index_lines)


def _index_lines(rows) -> list[str]:
    from chat.tool_stub import _summarise_args

    lines = []
    for m in rows:
        if m.role != "assistant" or is_dropped(m):
            continue
        for c in (m.metadata or {}).get("tool_calls") or []:
            # One line per call: argument values (canvas text …) may hold newlines.
            args = " ".join(_summarise_args(c.get("arguments")).split())
            name = c.get("name") or "tool"
            lines.append(f"- {name}({args})" if args else f"- {name}")
    return lines


@dataclass
class CheckpointPlan:
    expected_cutoff_id: object        # the cutoff this plan was made against
    existing_summary: str
    summary_message_count: int
    summary_target: int = 0
    summarise: bool = False
    to_summarise: list = field(default_factory=list)  # narrative rows, chronological
    new_cutoff_id: object = None      # last row folded in (any role)
    folded_count: int = 0
    index_lines: list = field(default_factory=list)   # the summary's new call index
    stub_rows: list = field(default_factory=list)     # tool rows given a saved stub
    call_rows: list = field(default_factory=list)     # assistant rows with trimmed calls
    drop_rows: list = field(default_factory=list)     # rows no longer replayed

    @property
    def empty(self) -> bool:
        return not (self.summarise or self.stub_rows or self.call_rows or self.drop_rows)


def plan_checkpoint(thread, budgets=None, *, keep_rounds: int,
                    force_summary: bool = False) -> CheckpointPlan | None:
    """What a checkpoint would do now (None when nothing). Rows in the plan carry
    their new metadata; ``apply_checkpoint_rows`` writes them."""
    from chat.tool_stub import build_tool_result_stub
    from core.tokens import count_tokens
    from llm.pipelines.edit_points import trim_history_call_args, with_trim_note

    budgets = _budgets(budgets)
    thread.refresh_from_db(fields=[
        "summary", "summary_token_count", "summary_up_to_message_id", "summary_message_count",
    ])
    msgs = [m for m in unsummarised_messages_qs(thread).order_by("created_at") if not is_dropped(m)]
    if keep_rounds > 0:
        round_starts = [i for i, m in enumerate(msgs) if starts_round(m)]
        split = round_starts[-keep_rounds] if len(round_starts) >= keep_rounds else 0
    else:
        split = len(msgs)
    plan = CheckpointPlan(
        expected_cutoff_id=thread.summary_up_to_message_id,
        existing_summary=thread.summary or "",
        summary_message_count=thread.summary_message_count or 0,
        summary_target=budgets.summary_target if budgets else 0,
    )
    usage = history_usage(thread, msgs)
    older = msgs[:split]

    # 1. Summarise: only a conversation overflow (or /compact) folds the narrative.
    if older and (force_summary or (budgets and usage.conversation > budgets.conversation)):
        _narrative, old_index = split_summary(thread.summary or "")
        cap = int(getattr(settings, "CONTEXT_TOOL_INDEX_MAX_ENTRIES", 20))
        plan.summarise = True
        plan.to_summarise = [m for m in older if not is_tool_noise(m)]
        plan.new_cutoff_id = older[-1].id
        plan.folded_count = len(older)
        plan.index_lines = (old_index + _index_lines(older))[-cap:] if cap > 0 else []
        return plan
    if budgets is None:
        return None

    # 2. Stub: tool pressure turns every older result into a saved stub.
    stub_bodies: dict = {}  # tool_call_id -> replayed body after this plan
    for m in msgs:
        if m.role == "tool" and m.tool_call_id:
            stub_bodies[m.tool_call_id] = replayed_body(m)
    if usage.history > checkpoint_fraction() * budgets.history:
        assistants = [m for m in older if m.role == "assistant" and (m.metadata or {}).get("tool_calls")]
        all_calls = [c for m in msgs for c in (m.metadata or {}).get("tool_calls") or []
                     if m.role == "assistant"]
        older_ids = {c.get("id") for m in assistants for c in m.metadata["tool_calls"] if c.get("id")}
        trims = trim_history_call_args(all_calls, older_ids) if older_ids else {}
        call_meta = {c.get("id"): (c.get("name", ""), c.get("arguments")) for c in all_calls}
        for m in assistants:
            current = replayed_calls(m)
            new_calls = [
                {**c, "arguments": trims[c["id"]].args} if c.get("id") in trims else c
                for c in current
            ]
            if new_calls != current:
                m.metadata = {**(m.metadata or {}), CTX_CALLS: new_calls}
                plan.call_rows.append(m)
        for m in older:
            if m.role != "tool" or not m.tool_call_id or _is_stub(replayed_body(m)):
                continue
            name, args = call_meta.get(m.tool_call_id, ("", None))
            body = build_tool_result_stub(name, args)
            trim = trims.get(m.tool_call_id)
            if trim is not None and trim.note:
                body = with_trim_note(body, trim.note)
            m.metadata = {
                **(m.metadata or {}),
                CTX_BODY: body, CTX_TOKENS: count_tokens(body), CTX_REASON: "checkpoint",
            }
            stub_bodies[m.tool_call_id] = body
            plan.stub_rows.append(m)

    # 3. Drop: past the stub cap, the oldest fully-stubbed calls go entirely.
    stub_tokens = {cid: count_tokens(b) for cid, b in stub_bodies.items() if _is_stub(b)}
    total = sum(stub_tokens.values())
    if total > budgets.stub_cap:
        target = int(budgets.stub_cap * (1 - float(getattr(settings, "CONTEXT_STUB_DROP_FRACTION", 0.75))))
        by_call = {m.tool_call_id: m for m in older if m.role == "tool" and m.tool_call_id}
        for m in older:
            if total <= target:
                break
            if m.role != "assistant":
                continue
            ids = [c.get("id") for c in replayed_calls(m)]
            if not ids or not all(cid in by_call and cid in stub_tokens for cid in ids):
                continue
            for row in [m] + [by_call[cid] for cid in ids]:
                row.metadata = {**(row.metadata or {}), CTX_DROPPED: True, CTX_REASON: "cap"}
                plan.drop_rows.append(row)
            total -= sum(stub_tokens[cid] for cid in ids)
    return None if plan.empty else plan


def apply_checkpoint_rows(plan: CheckpointPlan) -> int:
    from chat.models import ChatMessage

    rows = {r.pk: r for r in plan.stub_rows + plan.call_rows + plan.drop_rows}
    if rows:
        ChatMessage.objects.bulk_update(list(rows.values()), ["metadata"])
    return len(rows)


def save_summary(thread_id, text: str, last_msg_id, count: int, expected_cutoff_id=None) -> bool:
    """Write a new summary + cutoff under an optimistic lock on the cutoff the
    caller planned against (a concurrent checkpoint wins; this one is dropped)."""
    from chat.models import ChatThread
    from core.tokens import count_tokens

    qs = ChatThread.objects.filter(pk=thread_id)
    if expected_cutoff_id is not None:
        qs = qs.filter(summary_up_to_message_id=expected_cutoff_id)
    else:
        qs = qs.filter(summary_up_to_message_id__isnull=True)
    rows = qs.update(
        summary=text,
        summary_token_count=count_tokens(text),
        summary_up_to_message_id=last_msg_id,
        summary_message_count=count,
    )
    return rows > 0


@dataclass
class CheckpointResult:
    saved: bool
    folded_count: int = 0
    summary_tokens: int = 0
    summarised: bool = False
    stubbed: int = 0
    dropped: int = 0


async def run_checkpoint(thread, *, user_id, model: str | None, budgets=None,
                         keep_rounds: int | None = None, force_summary: bool = False,
                         reason: str = "threshold") -> CheckpointResult:
    """Plan and apply a checkpoint. Only the summarise action calls the LLM."""
    from channels.db import database_sync_to_async

    from chat.services import generate_summary
    from core.tokens import count_tokens

    if keep_rounds is None:
        keep_rounds = keep_rounds_setting()
    plan = await database_sync_to_async(plan_checkpoint)(
        thread, budgets, keep_rounds=keep_rounds, force_summary=force_summary,
    )
    if plan is None:
        return CheckpointResult(saved=False)
    if not plan.summarise:
        written = await database_sync_to_async(apply_checkpoint_rows)(plan)
        logger.info(
            "History checkpoint (%s): thread=%s stubbed=%d trimmed_calls=%d dropped_rows=%d",
            reason, thread.id, len(plan.stub_rows), len(plan.call_rows), len(plan.drop_rows),
        )
        return CheckpointResult(
            saved=bool(written), stubbed=len(plan.stub_rows), dropped=len(plan.drop_rows),
        )

    narrative, _old_index = split_summary(plan.existing_summary)
    if plan.to_summarise:
        narrative = await generate_summary(
            plan.to_summarise,
            existing_summary=narrative,
            user_id=user_id,
            conversation_id=str(thread.id),
            model=model,
            target_tokens=plan.summary_target or 5_000,
        )
    summary = join_summary(narrative, plan.index_lines)
    saved = await database_sync_to_async(save_summary)(
        thread.id, summary, plan.new_cutoff_id,
        plan.summary_message_count + len(plan.to_summarise),
        expected_cutoff_id=plan.expected_cutoff_id,
    )
    if saved:
        logger.info(
            "History checkpoint (%s): thread=%s folded=%d summarised=%d summary_tokens=%d "
            "target=%d keep_rounds=%d",
            reason, thread.id, plan.folded_count, len(plan.to_summarise),
            count_tokens(summary), plan.summary_target, keep_rounds,
        )
    else:
        logger.info("History checkpoint (%s) skipped for thread %s: concurrent update", reason, thread.id)
    return CheckpointResult(
        saved=saved, folded_count=plan.folded_count, summarised=saved,
        summary_tokens=count_tokens(summary) if saved else 0,
    )
