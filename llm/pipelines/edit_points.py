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
2. per-tool argument trimming of older tool calls (``trim_args_at_edit_point``),
3. pruning of old tool results (forced, or early past a fraction of the ceiling),
4. native-asset eviction with hysteresis (early threshold, low watermark),
5. refreshing the consolidated scratchpad block (old block removed, fresh one
   appended last).

The same code runs for the main agent and sub-agents and for every provider.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, List

from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest

logger = logging.getLogger(__name__)

# Header of the consolidated scratchpad block (a trailing user message). The
# block also carries ``metadata[SCRATCHPAD_BLOCK_KEY]`` so it can be found and
# replaced at the next edit point.
SCRATCHPAD_BLOCK_HEADER = "# Scratchpad (your private notes)"
SCRATCHPAD_BLOCK_KEY = "wf_scratchpad_block"
# Assistant-message metadata: ids of tool calls whose args were already trimmed.
TRIMMED_ARGS_KEY = "wf_trimmed_args"

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


def _trimmed_args(calls: List[ToolCall], eligible: Callable[[int, ToolCall], bool], tool_lookup) -> dict:
    """``{call_index: new_args}`` for eligible calls whose tool shrinks them."""
    out: dict[int, dict] = {}
    for idx, tc in enumerate(calls):
        if not eligible(idx, tc):
            continue
        tool = tool_lookup(tc.name)
        if tool is None:
            continue
        args = tc.arguments or {}
        try:
            size = len(json.dumps(args, default=str))
        except (TypeError, ValueError):
            continue
        if size < int(getattr(tool, "trim_args_min_chars", 2000)):
            continue
        try:
            new_args = tool.trim_args_at_edit_point(args, later_calls=calls[idx + 1:])
        except Exception:  # a buggy hook must never break a turn
            logger.exception("trim_args_at_edit_point failed for tool %s", tc.name)
            continue
        if isinstance(new_args, dict) and new_args != args:
            out[idx] = new_args
    return out


def trim_history_call_args(
    call_dicts: list,
    eligible_ids: set,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
) -> dict:
    """Between-turn variant for DB history, whose tool calls are dicts
    (``{"id", "name", "arguments"}``) in chronological order. Returns
    ``{call_id: new_args}`` for the eligible calls a tool chose to shrink."""
    calls = [
        ToolCall(id=str(d.get("id") or ""), name=d.get("name") or "", arguments=d.get("arguments") or {})
        for d in call_dicts if isinstance(d, dict)
    ]
    decided = _trimmed_args(
        calls, lambda _i, tc: tc.id in eligible_ids, tool_lookup,
    )
    return {calls[i].id: args for i, args in decided.items()}


def trim_old_tool_call_args(
    messages: List[Message],
    *,
    protected_call_ids: set,
    tool_lookup: Callable[[str], object] = default_tool_lookup,
) -> tuple[List[Message], int]:
    """Shrink big arguments of older tool calls via each tool's
    ``trim_args_at_edit_point`` hook. Calls in ``protected_call_ids`` and calls
    already trimmed are left alone. Returns ``(messages, trimmed_count)``; the
    original list is returned unchanged when nothing was trimmed."""
    triples = _call_triples(messages)
    if not triples:
        return messages, 0

    def eligible(idx: int, tc: ToolCall) -> bool:
        mi = triples[idx][0]
        already = (messages[mi].metadata or {}).get(TRIMMED_ARGS_KEY) or ()
        return tc.id not in protected_call_ids and tc.id not in already

    decided = _trimmed_args([t[2] for t in triples], eligible, tool_lookup)
    replacements: dict[int, dict[int, ToolCall]] = {}
    for idx, new_args in decided.items():
        mi, ci, tc = triples[idx]
        replacements.setdefault(mi, {})[ci] = tc.model_copy(update={"arguments": new_args})

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
        out[mi] = msg.model_copy(update={"tool_calls": calls, "metadata": meta})
        count += len(by_ci)
    return out, count


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

        projected = self._projected_tokens(new_messages, prior_len, real_input_tokens)
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
        else:
            self._append_seed_scratchpad(new_messages, req)
        self._maybe_nudge(new_messages, req, projected, ceiling)
        return new_messages, tools

    def finish_final(self, final_messages: List[Message], req: ChatRequest) -> List[Message]:
        """Backstop before the final tool-less call: if the assembled request still
        estimates over the ceiling, this is an edit point (prune, no protection
        for a current round — there is none)."""
        from core.tokens import estimate_chat_request_tokens

        ceiling = self.pipeline._midturn_ceiling(req)
        estimate = estimate_chat_request_tokens(final_messages)
        if not ceiling or estimate <= ceiling:
            return final_messages
        self._round += 1
        native_total, native_ceiling = self._native_state(final_messages, req)
        return self._edit_point(
            final_messages, req, [EDIT_KIND_PRUNE],
            projected=estimate, ceiling=ceiling,
            native_total=native_total, native_ceiling=native_ceiling,
            protect_call_ids=set(), protect_msg_indices=set(),
        )

    # -- edit point -------------------------------------------------------

    def _edit_point(
        self, messages, req, kinds, *, projected, ceiling,
        native_total, native_ceiling, protect_call_ids, protect_msg_indices,
    ) -> List[Message]:
        from chat.dedup import normalize_keep_newest
        from llm.pipelines.simple_chat import _prune_tool_messages_midturn

        ctx = req.context
        keep_recent = _setting("CONTEXT_MIDTURN_KEEP_TOOL_RESULTS", 6)
        protected_ids = set(protect_call_ids) | recent_tool_call_ids(messages, keep_recent)

        # 1. Dedup: re-decide append-only redactions as keep-newest.
        messages = normalize_keep_newest(messages)

        # 2. Per-tool argument trimming (needs ctx: the scratchpad block below is
        #    what makes a trimmed note recoverable).
        trimmed = 0
        if ctx is not None:
            messages, trimmed = trim_old_tool_call_args(
                messages, protected_call_ids=protected_ids, tool_lookup=self._lookup,
            )

        # 3. Prune old tool results — forced, or early to push the next edit out.
        pruned = 0
        frac = _setting("CONTEXT_EDIT_POINT_PRUNE_FRACTION", 0.7)
        if EDIT_KIND_PRUNE in kinds or (ceiling and projected > frac * ceiling):
            messages, pruned = _prune_tool_messages_midturn(
                messages, keep_recent=keep_recent, protect_call_ids=set(protect_call_ids),
            )
            if pruned and ctx is not None:
                ctx.bump_stat("prunes", 1)

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

        # 5. Consolidated scratchpad block, always last.
        messages = self._refresh_scratchpad_block(messages, req)
        self._seed_pending = False

        self.edit_rounds.append(self._round)
        if ctx is not None:
            ctx.bump_stat("edit_points_total", 1)
            for kind in dict.fromkeys(kinds):
                ctx.bump_stat(f"edit_point:{kind}", 1)
        logger.info(
            "Tool-loop edit point (%s): pruned=%d trimmed_args=%d evicted_native=%d "
            "(projected ~%d / ceiling %d, model %s, agent %s)",
            ",".join(kinds), pruned, trimmed, evicted, projected, ceiling or 0,
            req.model, self.agent_kind,
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

    def _refresh_scratchpad_block(self, messages, req) -> List[Message]:
        messages = [
            m for m in messages if not (m.metadata or {}).get(SCRATCHPAD_BLOCK_KEY)
        ]
        text = self._block_text(req.context)
        if text:
            messages.append(Message(
                role="user", content=text, metadata={SCRATCHPAD_BLOCK_KEY: True},
            ))
        return messages

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
    def _projected_tokens(messages, prior_len: int, real_input_tokens: int) -> int:
        from core.tokens import estimate_chat_request_tokens

        appended = messages[prior_len:]
        if real_input_tokens:
            return int(real_input_tokens) + estimate_chat_request_tokens(appended)
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
    "EditPointManager",
    "SCRATCHPAD_BLOCK_HEADER",
    "record_thinking_drops",
    "trim_history_call_args",
    "trim_old_tool_call_args",
]
