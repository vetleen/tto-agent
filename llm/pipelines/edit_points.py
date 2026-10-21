"""Append-only tool loop with edit points.

Between tool-loop rounds the request only ever GROWS: new assistant/tool
messages and notices are appended, nothing earlier is rewritten. That keeps the
request prefix byte-stable, which is what prompt caching needs on every provider
and what Anthropic's "preserved thinking" requires (a replayed thinking block is
bound to everything before it — system, tools, earlier messages; editing that
prefix drops or rejects the block).

Some rounds MUST edit earlier history: the context is over its ceiling (prune),
a skill attach added tools, or native assets overflow the provider limit. Such a
round is an **edit point**, and it does all deferred clean-up at once so each
edit point costs one cache break and (on Anthropic) one reasoning drop:

1. dedup normalization (restore append-only redactions, keep the newest copy),
2. per-tool argument and result trimming of older tool calls
   (``trim_args_at_edit_point`` / ``trim_result_at_edit_point``),
3. pruning of old tool results (forced, or early past a fraction of the ceiling),
4. native-asset eviction with hysteresis (early threshold, low watermark),
5. refreshing the consolidated trailing blocks (old ones removed, fresh ones
   appended: the main agent's current-deck block, then the scratchpad block last).

The same code runs for the main agent and sub-agents and for every provider.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, List

from llm.tools.interfaces import ArgTrim
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest

logger = logging.getLogger(__name__)

# Header of the consolidated scratchpad block (a trailing user message). The
# block also carries ``metadata[SCRATCHPAD_BLOCK_KEY]`` so it can be found and
# replaced at the next edit point.
SCRATCHPAD_BLOCK_HEADER = "# Scratchpad (your private notes)"
SCRATCHPAD_BLOCK_KEY = "wf_scratchpad_block"
# The main agent's consolidated "current deck" block (rendered by the consumer-
# supplied ``RunContext.deck_block_provider``); replaced at the next edit point
# like the scratchpad block, and always placed just before it.
DECK_BLOCK_KEY = "wf_deck_block"
# Assistant-message metadata: ids of tool calls whose args were already trimmed.
TRIMMED_ARGS_KEY = "wf_trimmed_args"
# Assistant-message metadata: ``{call_id: note}`` for trimmed calls whose tool
# described the removed fields (``ArgTrim.note``). Re-attached to the paired
# result at every edit point, because a later prune can replace that result
# with a stub.
TRIM_NOTES_KEY = "wf_trim_notes"
# Field that carries the note when the paired result is a JSON object.
RESULT_NOTE_FIELD = "omitted_args_note"
# Tool-message metadata: set once the result body was shrunk by its tool's
# trim_result_at_edit_point hook, so it is not offered to the hook again.
TRIMMED_RESULT_KEY = "wf_trimmed_result"
# Prefix of chat.tool_stub.build_tool_result_stub — a result already stubbed by
# pruning; never offered to a result hook.
_STUB_PREFIX = "[Earlier result of "

EDIT_KIND_PRUNE = "prune"
EDIT_KIND_TOOLS = "tools"
EDIT_KIND_NATIVE = "native_evict"


def _setting(name: str, default):
    try:
        from django.conf import settings

        return type(default)(getattr(settings, name, default))
    except Exception:
        return default


def _call_triples(messages: List[Message]):
    """Every ToolCall in history, oldest first, as (msg_idx, call_idx, call)."""
    out = []
    for mi, m in enumerate(messages):
        if getattr(m, "role", "") != "assistant":
            continue
        for ci, tc in enumerate(m.tool_calls or []):
            if isinstance(tc, ToolCall):
                out.append((mi, ci, tc))
    return out


def recent_tool_call_ids(messages: List[Message], keep_recent: int) -> set:
    """``tool_call_id`` of the ``keep_recent`` most recent tool results."""
    ids = [m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id]
    return set(ids[-keep_recent:]) if keep_recent > 0 else set()


def default_tool_lookup(name: str):
    from llm.tools.registry import get_tool_registry

    return get_tool_registry().get_tool(name)


def _as_arg_trim(value, args: dict) -> ArgTrim | None:
    """A hook's return value as an ArgTrim (a plain dict = args, no note), or None
    when it keeps the arguments."""
    if isinstance(value, ArgTrim):
        new_args, note = value.args, value.note
    elif isinstance(value, dict):
        new_args, note = value, ""
    else:
        return None
    if not isinstance(new_args, dict) or new_args == args:
        return None
    return ArgTrim(args=new_args, note=note or "")


def with_trim_note(content, note: str):
    """``content`` (a tool result body) carrying ``note``: a JSON object gets it as
    the ``RESULT_NOTE_FIELD`` field; anything else (plain text, a pruning stub)
    gets it appended on its own line. Bodies are only ever appended to, never
    prefixed, so the prefix checks for stubs and dedup placeholders still hold.
    Idempotent."""
    if not note or not isinstance(content, str) or note in content:
        return content
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        data[RESULT_NOTE_FIELD] = note
        return json.dumps(data)
    return f"{content}\n{note}" if content else note


def attach_trim_notes(messages: list, notes: dict) -> list:
    """Attach each ``{call_id: note}`` to its paired tool result. Works on
    ``Message`` objects (mid-turn) and on history dicts (between turns). Returns
    a new list, or the original one when nothing changed."""
    if not notes:
        return messages
    out = None
    for i, m in enumerate(messages):
        is_dict = isinstance(m, dict)
        role = m.get("role") if is_dict else getattr(m, "role", "")
        cid = m.get("tool_call_id") if is_dict else getattr(m, "tool_call_id", None)
        if role != "tool" or not cid or cid not in notes:
            continue
        content = m.get("content") if is_dict else m.content
        new = with_trim_note(content, notes[cid])
        if new == content:
            continue
        if out is None:
            out = list(messages)
        out[i] = {**m, "content": new} if is_dict else m.model_copy(update={"content": new})
    return messages if out is None else out


def _stored_trim_notes(messages: List[Message]) -> dict:
    notes: dict = {}
    for m in messages:
        if getattr(m, "role", "") == "assistant":
            notes.update((m.metadata or {}).get(TRIM_NOTES_KEY) or {})
    return notes


def _trimmed_args(
    calls: List[ToolCall], eligible: Callable[[int, ToolCall], bool], tool_lookup,
    recent_ids=frozenset(),
) -> dict:
    """``{call_index: ArgTrim}`` for eligible calls whose tool shrinks them. A
    call in ``recent_ids`` is skipped unless its tool sets
    ``trim_despite_recency`` (recency protects every other tool's arguments)."""
    out: dict[int, ArgTrim] = {}
    for idx, tc in enumerate(calls):
        if not eligible(idx, tc):
            continue
        tool = tool_lookup(tc.name)
        if tool is None:
            continue
        if tc.id in recent_ids and not getattr(tool, "trim_despite_recency", False):
            continue
        args = tc.arguments or {}
        try:
            size = len(json.dumps(args, default=str))
        except (TypeError, ValueError):
            continue
        if size < int(getattr(tool, "trim_args_min_chars", 2000)):
            continue
        try:
            trim = _as_arg_trim(tool.trim_args_at_edit_point(args, later_calls=calls[idx + 1:]), args)
        except Exception:  # a buggy hook must never break a turn
            logger.exception("trim_args_at_edit_point failed for tool %s", tc.name)
            continue
        if trim is not None:
            out[idx] = trim
    return out


def _calls_from_dicts(call_dicts: list) -> List[ToolCall]:
    """DB-history tool calls (``{"id", "name", "arguments"}`` dicts, chronological)
    as ToolCalls."""
    return [
        ToolCall(id=str(d.get("id") or ""), name=d.get("name") or "", arguments=d.get("arguments") or {})
        for d in call_dicts if isinstance(d, dict)
    ]


def trim_history_call_args(
    call_dicts: list,
    eligible_ids: set,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
    recent_ids=frozenset(),
) -> dict:
    """Between-turn variant for DB history, whose tool calls are dicts
    (``{"id", "name", "arguments"}``) in chronological order. Returns
    ``{call_id: ArgTrim}`` for the eligible calls a tool chose to shrink. The
    caller applies ``.args`` and attaches each ``.note`` to the result
    (``attach_trim_notes``). ``recent_ids`` (calls inside the raw window) are
    offered too, but only a tool with ``trim_despite_recency`` shrinks one of
    those."""
    calls = _calls_from_dicts(call_dicts)
    decided = _trimmed_args(
        calls, lambda _i, tc: tc.id in eligible_ids or tc.id in recent_ids, tool_lookup,
        recent_ids=recent_ids,
    )
    return {calls[i].id: args for i, args in decided.items()}


def _trimmed_results(items, calls: List[ToolCall], tool_lookup, recent_ids) -> dict:
    """``{call_id: new_content}`` for the results a tool chose to shrink. ``items``
    are ``(call_id, content)`` pairs; ``calls`` every ToolCall oldest first (for
    the tool name, the arguments and ``later_calls``). A result in ``recent_ids``
    is skipped unless its tool sets ``trim_despite_recency``."""
    index = {tc.id: i for i, tc in enumerate(calls)}
    out: dict[str, str] = {}
    for cid, content in items:
        if not isinstance(content, str) or content.startswith(_STUB_PREFIX):
            continue
        idx = index.get(cid)
        if idx is None:
            continue
        tc = calls[idx]
        tool = tool_lookup(tc.name)
        if tool is None:
            continue
        if cid in recent_ids and not getattr(tool, "trim_despite_recency", False):
            continue
        if len(content) < int(getattr(tool, "trim_result_min_chars", 2000)):
            continue
        hook = getattr(tool, "trim_result_at_edit_point", None)
        if hook is None:
            continue
        try:
            new = hook(content, tc.arguments or {}, later_calls=calls[idx + 1:])
        except Exception:  # a buggy hook must never break a turn
            logger.exception("trim_result_at_edit_point failed for tool %s", tc.name)
            continue
        if isinstance(new, str) and new != content:
            out[cid] = new
    return out


def trim_history_results(
    results: list,
    call_dicts: list,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
) -> dict:
    """Between-turn variant for DB history: ``results`` are the raw-window
    (unstubbed) tool rows as ``(tool_call_id, content)``, ``call_dicts`` every
    call as dicts in chronological order. All of these results are recent by
    definition, so only a tool with ``trim_despite_recency`` ever shrinks one.
    Returns ``{call_id: new_content}``."""
    calls = _calls_from_dicts(call_dicts)
    return _trimmed_results(results, calls, tool_lookup, recent_ids={cid for cid, _ in results})


def trim_old_tool_call_args(
    messages: List[Message],
    *,
    protected_call_ids: set,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
    recent_call_ids=frozenset(),
) -> tuple[List[Message], int]:
    """Shrink big arguments of older tool calls via each tool's
    ``trim_args_at_edit_point`` hook. Calls in ``protected_call_ids`` (the current
    round) and calls already trimmed are left alone; calls in ``recent_call_ids``
    are left alone unless their tool opts out of recency protection
    (``trim_despite_recency``). Returns ``(messages, trimmed_count)``; the
    original list is returned unchanged when nothing was trimmed."""
    triples = _call_triples(messages)
    if not triples:
        return messages, 0

    def eligible(idx: int, tc: ToolCall) -> bool:
        mi = triples[idx][0]
        already = (messages[mi].metadata or {}).get(TRIMMED_ARGS_KEY) or ()
        return tc.id not in protected_call_ids and tc.id not in already

    decided = _trimmed_args(
        [t[2] for t in triples], eligible, tool_lookup, recent_ids=recent_call_ids,
    )
    replacements: dict[int, dict[int, ToolCall]] = {}
    notes_by_msg: dict[int, dict[str, str]] = {}
    for idx, trim in decided.items():
        mi, ci, tc = triples[idx]
        replacements.setdefault(mi, {})[ci] = tc.model_copy(update={"arguments": trim.args})
        if trim.note:
            notes_by_msg.setdefault(mi, {})[tc.id] = trim.note

    if not replacements:
        return messages, 0
    out = list(messages)
    count = 0
    for mi, by_ci in replacements.items():
        msg = messages[mi]
        calls = [by_ci.get(ci, tc) for ci, tc in enumerate(msg.tool_calls)]
        meta = dict(msg.metadata or {})
        meta[TRIMMED_ARGS_KEY] = list(meta.get(TRIMMED_ARGS_KEY) or []) + [
            calls[ci].id for ci in by_ci
        ]
        if mi in notes_by_msg:
            meta[TRIM_NOTES_KEY] = {**(meta.get(TRIM_NOTES_KEY) or {}), **notes_by_msg[mi]}
        out[mi] = msg.model_copy(update={"tool_calls": calls, "metadata": meta})
        count += len(by_ci)
    return out, count


def trim_old_tool_results(
    messages: List[Message],
    *,
    protected_call_ids: set,
    recent_call_ids,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
) -> tuple[List[Message], int]:
    """Shrink big bodies of older tool results via each tool's
    ``trim_result_at_edit_point`` hook — the result-side twin of
    :func:`trim_old_tool_call_args`. Results of the current round
    (``protected_call_ids``), results already shrunk (``TRIMMED_RESULT_KEY``) and
    results already stubbed by pruning are left alone; ``recent_call_ids`` protect
    unless the tool opts out. Returns ``(messages, count)``; the original list
    when nothing changed — never a metadata-only write, so the edit point's
    no-change guard stays honest."""
    calls = [t[2] for t in _call_triples(messages)]
    items = []
    for i, m in enumerate(messages):
        cid = getattr(m, "tool_call_id", None)
        if getattr(m, "role", "") != "tool" or not cid or cid in protected_call_ids:
            continue
        if (m.metadata or {}).get(TRIMMED_RESULT_KEY):
            continue
        items.append((i, cid, m.content))
    decided = _trimmed_results(
        [(cid, content) for _i, cid, content in items], calls, tool_lookup, recent_call_ids,
    )
    if not decided:
        return messages, 0
    out = list(messages)
    for i, cid, _content in items:
        if cid in decided:
            out[i] = messages[i].model_copy(update={
                "content": decided[cid],
                "metadata": {**(messages[i].metadata or {}), TRIMMED_RESULT_KEY: True},
            })
    return out, len(decided)


def is_stub(content) -> bool:
    """A result body that pruning replaced with a re-read stub."""
    return isinstance(content, str) and content.startswith(_STUB_PREFIX)


def drop_stub_groups(
    messages: List[Message], *, cap: int, drop_fraction: float, eligible_call_ids: set,
) -> tuple[List[Message], int]:
    """Past ``cap`` tokens of stubbed results, remove the oldest stubbed tool
    calls entirely until the stubs are back under ``(1 − drop_fraction) × cap``.

    The unit is an assistant tool-call message together with all its results,
    and only when every one of its results is a stub and every call is in
    ``eligible_call_ids`` — so call/result pairing and role alternation stay
    valid on every provider. Returns ``(messages, dropped_call_count)``; the
    original list when nothing was dropped."""
    from core.tokens import count_tokens

    if cap <= 0:
        return messages, 0
    results = {
        m.tool_call_id: m for m in messages
        if getattr(m, "role", "") == "tool" and m.tool_call_id
    }
    total = sum(count_tokens(m.content) for m in results.values() if is_stub(m.content))
    if total <= cap:
        return messages, 0
    target = int(cap * (1 - drop_fraction))
    drop_ids: set = set()
    for m in messages:
        if total <= target:
            break
        if getattr(m, "role", "") != "assistant" or not m.tool_calls:
            continue
        ids = [tc.id for tc in m.tool_calls]
        if not all(cid in eligible_call_ids and cid in results and is_stub(results[cid].content)
                   for cid in ids):
            continue
        drop_ids.update(ids)
        total -= sum(count_tokens(results[cid].content) for cid in ids)
    if not drop_ids:
        return messages, 0
    out = [
        m for m in messages
        if not (getattr(m, "role", "") == "tool" and m.tool_call_id in drop_ids)
        and not (
            getattr(m, "role", "") == "assistant" and m.tool_calls
            and all(tc.id in drop_ids for tc in m.tool_calls)
        )
    ]
    return out, len(drop_ids)


def diff_context_edits(before: List[Message], after: List[Message]) -> dict | None:
    """What a round rewrote, keyed by ``tool_call_id``: ``results`` (new body of a
    tool result), ``calls`` (new arguments of a call) and ``dropped`` (calls
    removed with their results). The consumer saves this on the stored rows so
    the next turn replays exactly what this one sent (``chat/consumers.py``
    ``_persist_context_edits``). None when nothing changed."""
    def _tool_bodies(msgs):
        return {
            m.tool_call_id: m.content for m in msgs
            if getattr(m, "role", "") == "tool" and m.tool_call_id
        }

    def _call_args(msgs):
        return {
            tc.id: tc.arguments for m in msgs if getattr(m, "role", "") == "assistant"
            for tc in (m.tool_calls or []) if isinstance(tc, ToolCall) and tc.id
        }

    old_bodies, new_bodies = _tool_bodies(before), _tool_bodies(after)
    old_args, new_args = _call_args(before), _call_args(after)
    results = {
        cid: body for cid, body in new_bodies.items()
        if cid in old_bodies and old_bodies[cid] != body
    }
    calls = {
        cid: args for cid, args in new_args.items()
        if cid in old_args and old_args[cid] != args
    }
    dropped = sorted(cid for cid in old_args if cid not in new_args)
    if not (results or calls or dropped):
        return None
    return {"results": results, "calls": calls, "dropped": dropped}


def _provider_view(messages: List[Message]) -> list:
    """What a provider sees of ``messages`` (metadata excluded) — to tell whether
    an edit point changed anything."""
    return [
        (m.role, m.content, m.tool_call_id,
         [(tc.id, tc.name, tc.arguments) for tc in (m.tool_calls or []) if isinstance(tc, ToolCall)])
        for m in messages
    ]


def record_thinking_drops(context, response_metadata) -> None:
    """Count Anthropic ``input_transformations`` thinking drops for this round
    (present only when the preserved-thinking beta header is sent). Unknown
    types/reasons are ignored — the API adds values over time."""
    if context is None or not isinstance(response_metadata, dict):
        return
    entries = response_metadata.get("input_transformations") or []
    if not isinstance(entries, list):
        return
    dropped = [e for e in entries if isinstance(e, dict) and e.get("type") == "thinking_dropped"]
    if dropped:
        context.bump_stat("thinking_dropped", len(dropped))
        logger.info(
            "Provider dropped %d replayed thinking block(s): %s",
            len(dropped), sorted({str(e.get("reason")) for e in dropped}),
        )


class EditPointManager:
    """Owns the per-round history maintenance of one tool-loop invocation.

    The loop appends the assistant message and its tool results, then calls
    :meth:`finish_round`, which appends this round's extras (notice, requested
    files, skill instructions) and — only when a trigger fires — performs an
    edit point. :meth:`finish_final` does the same for the final tool-less call.
    """

    def __init__(self, pipeline, req: ChatRequest, tool_by_name: dict):
        self.pipeline = pipeline
        self.tool_by_name = tool_by_name
        ctx = req.context
        self.agent_kind = getattr(ctx, "agent_kind", "main") if ctx else "main"
        self.nudged = False
        # A sub-agent resumed after a retry starts with its scratchpad seeded
        # from the DB; show it once (append-only) since no call carries it.
        self._seed_pending = bool(
            self.agent_kind == "subagent" and ctx is not None and (ctx.scratchpad or "").strip()
        )
        # Round numbers (0-based) that were edit points — for tests/diagnostics.
        self.edit_rounds: list[int] = []
        self._round = -1
        # What the last finish_round / finish_final rewrote in earlier messages
        # (``diff_context_edits``), for the caller to save so later turns replay
        # the same bytes. None when nothing changed.
        self.last_edit: dict | None = None
        # Calls that came in with the history (earlier turns): the only ones the
        # stub cap may drop — the current turn's calls may carry reasoning or
        # signatures bound to them.
        self._history_call_ids = {
            tc.id for m in req.messages if getattr(m, "role", "") == "assistant"
            for tc in (m.tool_calls or []) if isinstance(tc, ToolCall) and tc.id
        }

    # -- public -----------------------------------------------------------

    def finish_round(
        self,
        new_messages: List[Message],
        req: ChatRequest,
        tools: list,
        *,
        prior_len: int,
        real_input_tokens: int,
        round_call_ids: set,
        notice: Message | None,
    ) -> tuple[List[Message], list]:
        """Complete a tool round. ``new_messages[prior_len:]`` is what this round
        appended (assistant message + tool results)."""
        from chat.dedup import redact_new_duplicates

        self._round += 1
        before_round = new_messages
        new_messages = redact_new_duplicates(new_messages, prior_len)
        if notice is not None:
            new_messages.append(notice)
        before_assets = len(new_messages)
        self.pipeline._append_pending_native_assets(new_messages, req)
        asset_msg_indices = set(range(before_assets, len(new_messages)))
        self.pipeline._append_pending_skill_instructions(new_messages, req)

        kinds: list[str] = []
        grown = self.pipeline._expand_tools_from_context(tools, self.tool_by_name, req)
        if len(grown) != len(tools):
            kinds.append(EDIT_KIND_TOOLS)
        tools = grown

        projected = self._projected_tokens(
            new_messages, prior_len, real_input_tokens, model=req.model,
        )
        ceiling = self.pipeline._midturn_ceiling(req)
        if ceiling and projected > ceiling:
            kinds.append(EDIT_KIND_PRUNE)
        native_total, native_ceiling = self._native_state(new_messages, req)
        if native_ceiling and native_total > native_ceiling:
            kinds.append(EDIT_KIND_NATIVE)

        if kinds:
            new_messages = self._edit_point(
                new_messages, req, kinds,
                projected=projected, ceiling=ceiling,
                native_total=native_total, native_ceiling=native_ceiling,
                protect_call_ids=round_call_ids,
                protect_msg_indices=asset_msg_indices,
            )
        self._append_seed_scratchpad(new_messages, req)  # no-op once shown
        self._maybe_nudge(new_messages, req, projected, ceiling)
        self.last_edit = diff_context_edits(before_round, new_messages)
        return new_messages, tools

    def finish_final(self, final_messages: List[Message], req: ChatRequest) -> List[Message]:
        """Backstop before the final tool-less call: if the assembled request still
        estimates over the ceiling, this is an edit point (prune, no protection
        for a current round — there is none)."""
        from core.tokens import estimate_chat_request_tokens

        self.last_edit = None
        ceiling = self.pipeline._midturn_ceiling(req)
        estimate = estimate_chat_request_tokens(final_messages)
        if not ceiling or estimate <= ceiling:
            return final_messages
        self._round += 1
        native_total, native_ceiling = self._native_state(final_messages, req)
        edited = self._edit_point(
            final_messages, req, [EDIT_KIND_PRUNE],
            projected=estimate, ceiling=ceiling,
            native_total=native_total, native_ceiling=native_ceiling,
            protect_call_ids=set(), protect_msg_indices=set(),
        )
        self.last_edit = diff_context_edits(final_messages, edited)
        return edited

    # -- edit point -------------------------------------------------------

    def _edit_point(
        self, messages, req, kinds, *, projected, ceiling,
        native_total, native_ceiling, protect_call_ids, protect_msg_indices,
    ) -> List[Message]:
        from chat.dedup import normalize_keep_newest
        from llm.pipelines.simple_chat import _prune_tool_messages_midturn

        ctx = req.context
        keep_recent = _setting("CONTEXT_MIDTURN_KEEP_TOOL_RESULTS", 6)
        current_ids = set(protect_call_ids)
        recent_ids = recent_tool_call_ids(messages, keep_recent)
        original = messages
        before = _provider_view(messages)

        # 0. The current-deck block, read BEFORE any trimming so the slide tools
        #    know whether the copy their markers point at will exist (a failed read
        #    makes them keep their payloads this edit point). One provider call per
        #    edit point — also paid by an edit point that turns out to change nothing.
        deck_text, deck_failed = self._deck_block_text(ctx)
        if ctx is not None:
            ctx.deck_block_failed = deck_failed
        if deck_failed:
            logger.warning("Deck block unavailable at this edit point; slide payloads are kept")

        # 1. Dedup: re-decide append-only redactions as keep-newest.
        messages = normalize_keep_newest(messages)

        # 2. Per-tool argument trimming (needs ctx: the trailing blocks below are
        #    what make a trimmed note / deck recoverable). The current round is
        #    always protected; recency protects unless the tool opts out.
        trimmed = 0
        trimmed_results = 0
        if ctx is not None:
            messages, trimmed = trim_old_tool_call_args(
                messages, protected_call_ids=current_ids, tool_lookup=self._lookup,
                recent_call_ids=recent_ids,
            )
            # 2b. ... and result trimming (same protection), before pruning stubs
            #     whatever recency no longer protects.
            messages, trimmed_results = trim_old_tool_results(
                messages, protected_call_ids=current_ids, recent_call_ids=recent_ids,
                tool_lookup=self._lookup,
            )

        # 3. Prune old tool results — forced, or early to push the next edit out.
        pruned = 0
        frac = _setting("CONTEXT_EDIT_POINT_PRUNE_FRACTION", 0.7)
        if EDIT_KIND_PRUNE in kinds or (ceiling and projected > frac * ceiling):
            messages, pruned = _prune_tool_messages_midturn(
                messages, keep_recent=keep_recent, protect_call_ids=set(protect_call_ids),
                tool_lookup=self._lookup,
            )
            if pruned and ctx is not None:
                ctx.bump_stat("prunes", 1)

        # 3a. Stub cap: past CONTEXT_STUB_CAP_FRACTION of the ceiling, the oldest
        #     stubbed calls from earlier turns go entirely (call + results).
        messages, dropped = drop_stub_groups(
            messages,
            cap=int(_setting("CONTEXT_STUB_CAP_FRACTION", 0.05) * (ceiling or 0)),
            drop_fraction=_setting("CONTEXT_STUB_DROP_FRACTION", 0.75),
            eligible_call_ids=self._history_call_ids - current_ids,
        )

        # 3b. Trimmed calls' notes go onto their results, after result trimming
        #     (JSON hooks parse those bodies) and after pruning (a stub replaces
        #     the body, so a note attached at an earlier edit point is re-added).
        messages = attach_trim_notes(messages, _stored_trim_notes(messages))

        # 4. Native assets — evict with hysteresis so the next file doesn't
        #    trigger another edit.
        evicted = 0
        early = _setting("NATIVE_EVICT_EARLY_FRACTION", 0.8)
        if native_ceiling and (
            EDIT_KIND_NATIVE in kinds or native_total > early * native_ceiling
        ):
            from llm.core.native_limits import apply_native_evictions, plan_native_evictions

            low = _setting("NATIVE_EVICT_LOW_FRACTION", 0.6)
            plan = plan_native_evictions(
                messages, int(low * native_ceiling), protect_msg_indices=protect_msg_indices,
            )
            messages = apply_native_evictions(messages, plan)
            evicted = len(plan)

        # Nothing actually changed (e.g. still over the ceiling but nothing left to
        # prune): not an edit point. Keep the prefix untouched — refreshing the
        # scratchpad block alone would edit history every round for nothing.
        if EDIT_KIND_TOOLS not in kinds and _provider_view(messages) == before:
            logger.debug(
                "Edit point triggered (%s) but nothing to change; staying append-only",
                ",".join(kinds),
            )
            return original

        # 5. Consolidated trailing blocks: the current deck (main agent, when the
        #    deck changed this turn — read in step 0), then the scratchpad block,
        #    always last.
        messages, deck_block = self._refresh_trailing_blocks(
            messages, req, deck_text, deck_failed=deck_failed,
        )
        self._seed_pending = False

        self.edit_rounds.append(self._round)
        if ctx is not None:
            ctx.bump_stat("edit_points_total", 1)
            for kind in dict.fromkeys(kinds):
                ctx.bump_stat(f"edit_point:{kind}", 1)
            if deck_block:
                # Not an ``edit_point:*`` key — the logger maps those to trigger kinds.
                ctx.bump_stat("deck_block_refreshes", 1)
        logger.info(
            "Tool-loop edit point (%s): pruned=%d dropped_stub_calls=%d trimmed_args=%d "
            "trimmed_results=%d evicted_native=%d deck_block=%s (projected ~%d / ceiling %d, "
            "model %s, agent %s)",
            ",".join(kinds), pruned, dropped, trimmed, trimmed_results, evicted, deck_block,
            projected, ceiling or 0, req.model, self.agent_kind,
        )
        return messages

    # -- scratchpad -------------------------------------------------------

    def _block_text(self, ctx) -> str:
        if ctx is None:
            return ""
        if self.agent_kind == "subagent":
            notes = (ctx.scratchpad or "").strip()
            if not notes:
                return ""
            return (
                f"{SCRATCHPAD_BLOCK_HEADER}\n"
                "Your own notes from earlier in this run — retained even when tool "
                "results are cleared. Add to it with `subagent_scratchpad_append`.\n"
                f"```\n{notes}\n```"
            )
        notes = "\n\n".join(n for n in (ctx.scratchpad_turn_notes or []) if n)
        if not notes.strip():
            return ""
        return (
            f"{SCRATCHPAD_BLOCK_HEADER}\n"
            "Notes you saved to your scratchpad earlier in this turn (older notes "
            "are in the scratchpad section of your context above). Older context "
            "was just compacted; these notes are retained.\n"
            f"```\n{notes}\n```"
        )

    def _refresh_trailing_blocks(
        self, messages, req, deck: str, *, deck_failed: bool = False,
    ) -> tuple[List[Message], bool]:
        """Drop the previous trailing blocks and append fresh ones: the current
        deck block ``deck`` (when there is one), then the scratchpad block, always
        last. A failed deck re-read (``deck_failed``) keeps the last good deck
        block in place — payloads trimmed at earlier edit points point at it.
        Returns ``(messages, deck_block_appended)``."""
        def keep(m: Message) -> bool:
            meta = m.metadata or {}
            if meta.get(SCRATCHPAD_BLOCK_KEY):
                return False
            if meta.get(DECK_BLOCK_KEY):
                return deck_failed
            return True

        messages = [m for m in messages if keep(m)]
        if deck:
            messages.append(Message(
                role="user", content=deck, metadata={DECK_BLOCK_KEY: True},
            ))
        text = self._block_text(req.context)
        if text:
            messages.append(Message(
                role="user", content=text, metadata={SCRATCHPAD_BLOCK_KEY: True},
            ))
        return messages, bool(deck)

    def _deck_block_text(self, ctx) -> tuple[str, bool]:
        """``(text, failed)`` — the main agent's "current deck" block, rendered by
        the consumer-supplied ``RunContext.deck_block_provider`` (re-reads the
        active deck from the DB; ``""`` when there is none or it is unchanged
        since the preamble; ``None`` or a raise = the read failed). Never raises —
        the block is best-effort."""
        provider = getattr(ctx, "deck_block_provider", None)
        if provider is None or self.agent_kind != "main":
            return "", False
        try:
            text = provider()
        except Exception:  # noqa: BLE001
            logger.exception("deck_block_provider failed")
            return "", True
        if text is None:
            return "", True
        return (text if isinstance(text, str) else ""), False

    def _append_seed_scratchpad(self, messages, req) -> None:
        if not self._seed_pending:
            return
        self._seed_pending = False
        text = self._block_text(req.context)
        if text:
            messages.append(Message(
                role="user", content=text, metadata={SCRATCHPAD_BLOCK_KEY: True},
            ))

    def _maybe_nudge(self, messages, req, projected: int, ceiling: int) -> None:
        """Sub-agents with no notes get ONE appended reminder once the context is
        filling (the main agent's equivalent lives in its per-turn preamble)."""
        ctx = req.context
        if (
            self.nudged or self.agent_kind != "subagent" or ctx is None
            or (ctx.scratchpad or "").strip() or not ceiling
            or projected < 0.7 * ceiling
        ):
            return
        self.nudged = True
        messages.append(Message(role="user", content=(
            "# Context notice\n"
            "Your context is filling up, and older tool results will be cleared "
            "to stay under budget. Save any facts, figures, or URLs you'll need "
            "for your final answer with `subagent_scratchpad_append` now — cleared "
            "results don't come back."
        )))

    # -- helpers ----------------------------------------------------------

    def _lookup(self, name: str):
        return self.tool_by_name.get(name) or default_tool_lookup(name)

    @staticmethod
    def _projected_tokens(
        messages, prior_len: int, real_input_tokens: int, *, model: str | None = None,
    ) -> int:
        """Next request's size in ESTIMATE units (the ceiling's units): the
        provider's real count of the last request is converted back with the
        model's token_estimate_factor so it isn't corrected twice."""
        from core.tokens import estimate_chat_request_tokens
        from llm.context_budget import token_estimate_factor

        appended = messages[prior_len:]
        if real_input_tokens:
            real_as_estimate = int(int(real_input_tokens) / token_estimate_factor(model))
            return real_as_estimate + estimate_chat_request_tokens(appended)
        return estimate_chat_request_tokens(messages)

    @staticmethod
    def _native_state(messages, req) -> tuple[int, int]:
        from llm.core.model_factory import detect_provider
        from llm.core.native_limits import native_b64_total, provider_native_b64_ceiling

        total = native_b64_total(messages)
        if not total:
            return 0, 0
        return total, provider_native_b64_ceiling(detect_provider(req.model or ""))


__all__ = [
    "DECK_BLOCK_KEY",
    "EditPointManager",
    "SCRATCHPAD_BLOCK_HEADER",
    "RESULT_NOTE_FIELD",
    "TRIMMED_RESULT_KEY",
    "attach_trim_notes",
    "diff_context_edits",
    "drop_stub_groups",
    "is_stub",
    "record_thinking_drops",
    "with_trim_note",
    "trim_history_call_args",
    "trim_history_results",
    "trim_old_tool_call_args",
    "trim_old_tool_results",
]
