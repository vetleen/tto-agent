"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Iterator, List, Tuple

from llm.core.interfaces import ChatModel
from llm.core.model_factory import create_chat_model
from llm.pipelines.base import BasePipeline
from llm.pipelines.edit_points import EditPointManager, record_thinking_drops
from llm.pipelines.registry import get_pipeline_registry
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent
from llm.types.context import RunContext
from llm.tools.display import parse_tool_result, tool_end_display
from llm.tools.interfaces import ContextAwareTool, find_echoed_marker
from llm.tools.registry import get_tool_registry
from llm.tools._throttle import set_call_deadline

logger = logging.getLogger(__name__)

# Hard cap on a single tool result entering the conversation history.
# ~50k tokens — fits comfortably inside the smallest registry context window
# (128k tokens) with room left for history. Enforced centrally in
# _execute_tool_calls so every tool (web, canvas, doc search, skills) is
# covered without per-tool logic.
MAX_TOOL_RESULT_CHARS = 200_000


def _round_stage_line(rounds_left: int, budget: int) -> str:
    """The pacing advice for a sub-agent with *rounds_left* of *budget* rounds.

    Stages, from first to last: explore → converge (more than half the budget
    used) → wrap up (the last ~20%, at least 2 rounds) → final round. Tested at
    the boundaries for the quick/standard/deep budgets.
    """
    if rounds_left <= 1:
        return (
            "This is your final round: finish your canvas and write your final reply. "
            "Do not repeat what is in the canvas in your reply."
        )
    if rounds_left <= max(2, math.ceil(0.2 * budget)):
        return "Your rounds are almost up: finish your canvas report soon."
    if rounds_left <= budget // 2:
        return (
            "Converge: start no new lines of inquiry; fill the gaps that matter and make "
            "sure every section has a draft in your canvas."
        )
    return (
        "Explore: search broadly; note key facts and source URLs in your scratchpad as "
        "you find them."
    )


def _iteration_notice(
    iteration_index: int,
    max_iterations: int,
    agent_kind: str,
) -> Message | None:
    """The user message appended after an iteration's tool results, or None.

    Sub-agents get a round-budget countdown after every round but the last (the
    forced final call carries its own notice), plus one line of pacing advice
    for the stage they are in (``_round_stage_line``), so they pace themselves
    across the whole budget instead of hitting the cliff. Main-chat
    behavior is unchanged: a warning only at <=2 remaining. Cache-safe either
    way — the message is appended after the tool results, never edited into
    earlier history.
    """
    finished = iteration_index + 1
    remaining = max_iterations - finished

    if agent_kind == "subagent":
        if remaining <= 0:
            # Budget used up: the loop's next call is the forced, tool-less
            # final one and _final_request says so. A countdown here would
            # promise a "final round with tools" that cannot happen.
            return None
        return Message(role="user", content=(
            f"[Round budget: {remaining} {'round' if remaining == 1 else 'rounds'} left "
            f"(round {finished}/{max_iterations} done). Complete your task and report your "
            f"findings within round {max_iterations}/{max_iterations}.] "
            + _round_stage_line(remaining, max_iterations)
        ))

    if remaining > 2:
        return None
    warning = (
        "This is your last tool iteration." if remaining <= 1
        else "You have 2 tool iterations remaining."
    )
    return Message(role="user", content=(
        f"IMPORTANT: {warning} Wrap up your research and prepare to deliver your "
        "comprehensive final response as plain text."
    ))


# A round is "truncated" when the provider cut it at the output cap or the
# stream guard aborted it for degenerate repetition (end_data/metadata
# ``output_truncated``). Its tool calls are never run: a cut-off call's JSON was
# repaired by LangChain's partial parser, so the arguments may be silently
# truncated. After this many in a row the loop gives up on tools and streams the
# final answer, like the round-budget path.
_MAX_CONSECUTIVE_TRUNCATED_ROUNDS = 2
_TRUNCATED_TOOL_LABEL = "Not run — output was cut off"


def _truncated_call_result(tool_name: str, stop_reason: str) -> str:
    """The synthetic tool result for a call the model never finished emitting."""
    return json.dumps({
        "status": "error",
        "error": "output_truncated",
        "message": (
            f"Your output hit the output limit or started repeating ({stop_reason}) "
            f"before this {tool_name} call was complete, so it was NOT run. "
            "Re-issue it with a much shorter payload; split large content across "
            "several smaller calls."
        ),
    })


def _truncated_round_notice(stop_reason: str, progress: Message | None) -> Message:
    """The user message appended after a truncated round. Absorbs the iteration
    progress notice so the round ends in a single user turn (appended after the
    tool results, never edited into earlier history)."""
    text = (
        f"[Your previous output was cut off ({stop_reason}) and discarded. Nothing "
        "ran. Continue with a shorter output. Write a long document in parts: one "
        "canvas_write for the first part, then canvas_edit calls that append the "
        "next parts — each call well under your output limit.]"
    )
    if progress is not None and progress.content:
        text = f"{text} {progress.content}"
    return Message(role="user", content=text)


# Sub-agents only: a tool call without its own ``timeout_seconds`` may run this
# long before the round stops waiting for it (see _run_round_with_timeouts).
SUBAGENT_TOOL_TIMEOUT_DEFAULT = 60.0
_TOOL_TIMEOUT_ERROR = "tool_timeout"


def _tool_timeout(tool) -> float:
    value = getattr(tool, "timeout_seconds", None) if tool is not None else None
    return float(value) if value else SUBAGENT_TOOL_TIMEOUT_DEFAULT


def _timeout_result(tool_name: str, timeout_s: float) -> str:
    return json.dumps({
        "status": "error",
        "error": _TOOL_TIMEOUT_ERROR,
        "message": (
            f"This {tool_name} call timed out after {timeout_s:.0f} s and was not "
            "completed. If it returns later, its result will be delivered to you. "
            "You may retry, narrow, or move on."
        ),
        "timeout_s": round(timeout_s),
    })


def _not_started_result(tool_name: str) -> str:
    return json.dumps({
        "status": "error",
        "error": _TOOL_TIMEOUT_ERROR,
        "message": (
            f"This {tool_name} call was not run: other calls in this round timed out "
            "and blocked it. Call it again if you still need it."
        ),
    })


def _timed_out_label(result_str: str) -> str | None:
    """The tool_end label for a timed-out call, else None."""
    if f'"error": "{_TOOL_TIMEOUT_ERROR}"' not in result_str[:80]:
        return None
    try:
        timeout_s = json.loads(result_str).get("timeout_s")
    except (ValueError, AttributeError):
        timeout_s = None
    return f"Timed out after {timeout_s} s" if timeout_s else "Not run — round timed out"


def _run_round_with_timeouts(tool_calls, tool_by_name, run_one, context) -> list:
    """Run a sub-agent round's calls in parallel, each under its tool's timeout.

    The round waits while any started call is still within its own timeout
    (or a queued call can still get a worker), so a fast tool's timeout never
    cuts a round short while a slower sibling is legitimately working. Then
    every unfinished call gets a timeout result: a started one is abandoned
    (Python cannot kill its thread — it runs on and, if it returns, the result
    is delivered on a later round via ``context.pending_late_tool_results``),
    a never-started one is cancelled. Each timeout is logged at WARNING and
    recorded for calibration (``context.record_tool_timeout``).
    """
    max_workers = min(len(tool_calls), 4)
    pool = ThreadPoolExecutor(max_workers=max_workers)
    started_at: dict[int, float] = {}
    timeouts = [_tool_timeout(tool_by_name.get(tc.name)) for tc in tool_calls]

    def _wrapped(index, tc):
        started_at[index] = time.monotonic()
        # The call's retry backoffs and rate-window waits stop at its timeout
        # (llm.tools._throttle.wait_budget), so an abandoned thread winds down
        # instead of sleeping on and firing requests for a round that has
        # moved on.
        set_call_deadline(timeouts[index])
        try:
            return run_one(tc)
        finally:
            set_call_deadline(None)
            context.note_activity()

    futures = [pool.submit(_wrapped, i, tc) for i, tc in enumerate(tool_calls)]
    try:
        while True:
            pending = [i for i, f in enumerate(futures) if not f.done()]
            if not pending:
                break
            now = time.monotonic()
            running = [i for i in pending if i in started_at]
            within = [i for i in running if now - started_at[i] < timeouts[i]]
            queued = [i for i in pending if i not in started_at]
            # A queued call still gets a worker unless every worker is held by
            # an overdue call.
            can_start = bool(queued) and len(running) - len(within) < max_workers
            if not within and not can_start:
                break
            wake = min((timeouts[i] - (now - started_at[i]) for i in within), default=1.0)
            wait([futures[i] for i in pending], timeout=max(0.05, min(wake, 1.0)),
                 return_when=FIRST_COMPLETED)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    results = []
    for i, (tc, future) in enumerate(zip(tool_calls, futures)):
        if future.done() and not future.cancelled():
            try:
                results.append(future.result())
            except Exception as exc:  # run_one never raises; belt and braces
                results.append((tc, json.dumps({"error": str(exc)})))
            continue
        if i not in started_at:
            results.append((tc, _not_started_result(tc.name)))
            continue
        timeout_s = timeouts[i]
        logger.warning(
            "Sub-agent tool %s timed out after %.0fs (run %s); abandoning the call",
            tc.name, timeout_s, getattr(context, "run_id", ""),
        )
        record = {
            "tool": tc.name,
            "timeout_s": round(timeout_s),
            "round": context.observability.get("rounds", 0),
            "late_after_s": None,
        }
        context.record_tool_timeout(record)
        entry = {
            "future": future, "tool": tc.name, "round": record["round"],
            "started": started_at[i], "timeout_s": timeout_s, "record": record,
        }

        def _on_late_done(_f, entry=entry):
            entry["done_after_s"] = time.monotonic() - entry["started"]
            logger.info(
                "Abandoned sub-agent tool %s returned after %.0fs (timeout %.0fs)",
                entry["tool"], entry["done_after_s"], entry["timeout_s"],
            )

        future.add_done_callback(_on_late_done)
        context.pending_late_tool_results.append(entry)
        results.append((tc, _timeout_result(tc.name, timeout_s)))
    return results


def _drain_late_tool_results(context) -> str:
    """Text for the timed-out calls whose abandoned threads have since returned
    (removed from the pending list), or "". Appended inside the round's single
    notice message, so the loop stays append-only (prompt-cache safe)."""
    pending = getattr(context, "pending_late_tool_results", None) if context else None
    if not pending:
        return ""
    parts = []
    for entry in list(pending):
        future = entry["future"]
        if not future.done():
            continue
        pending.remove(entry)
        if future.cancelled():
            continue
        try:
            _, result_str = future.result()
        except Exception as exc:
            result_str = json.dumps({"error": str(exc)})
        done_after = entry.get("done_after_s") or (time.monotonic() - entry["started"])
        entry["record"]["late_after_s"] = round(done_after)
        parts.append(
            f"[Late tool result] The {entry['tool']} call you made in round {entry['round']} "
            f"timed out after {entry['timeout_s']:.0f} s but returned after {done_after:.0f} s. "
            f"In case you still need it:\n{result_str}"
        )
    return "\n\n".join(parts)


def _with_late_results(notice: Message | None, context) -> Message | None:
    """Fold any late tool results into the round's notice (one user message)."""
    late = _drain_late_tool_results(context)
    if not late:
        return notice
    if notice is None or not isinstance(notice.content, str):
        return Message(role="user", content=late)
    return Message(role="user", content=f"{late}\n\n{notice.content}")


def _count_round(context) -> None:
    """Count one model call of the tool loop (the forced final call included)
    as a round; ``observability["rounds"]`` is what a sub-agent run reports."""
    if context is not None:
        context.bump_stat("rounds")
        context.note_activity()


def _set_loop_stop(context, reason: str) -> None:
    """Record why the tool loop ended: ``answered`` (the model replied without
    tools), ``budget`` (the round budget ran out → forced final call),
    ``truncation`` (consecutive cut-off rounds → forced final call) or
    ``cancelled``. Read by the sub-agent service and logged on LLMCallLog."""
    if context is not None:
        context.observability["loop_stop"] = reason


def _usable_tool_call_dicts(tool_call_dicts) -> list[dict]:
    """Drop calls whose name or id never arrived: a fragment cut before the name
    yields ``name == ""``, which the provider rejects when it is replayed."""
    return [tc for tc in (tool_call_dicts or []) if tc.get("name") and tc.get("id")]


def _keep_replay_blocks(end_data: dict) -> None:
    """Swap the final reply's ``content_blocks`` for ``replay_blocks``: its text
    blocks when the provider signed one (Gemini thought signatures), so the
    caller can store them and send them back on later turns."""
    from llm.core.langchain_utils import signed_text_blocks

    blocks = signed_text_blocks(end_data.pop("content_blocks", None))
    if blocks:
        end_data["replay_blocks"] = blocks


def _assistant_replay_meta(end_data: dict) -> dict:
    """Provider state the assistant message must carry back on the next call."""
    meta: dict = {}
    content_blocks = end_data.get("content_blocks")
    if content_blocks:
        # Carry the provider's native thinking/redacted_thinking blocks (with
        # signatures) so to_langchain_messages can echo them back — Anthropic
        # 400s on a thinking+tool-use turn whose blocks were dropped and the
        # assistant message rebuilt from plain text.
        meta["content_blocks"] = content_blocks
    replay_ak = end_data.get("additional_kwargs")
    if replay_ak:
        # e.g. Gemini function-call thought signatures — must round-trip or
        # Gemini 3 rejects the follow-up with 400 INVALID_ARGUMENT.
        meta["additional_kwargs"] = replay_ak
    return meta


_safe_result_dict = parse_tool_result


def _pipeline_call_meta(messages: List[Message]) -> dict:
    """Map ``tool_call_id -> (tool_name, arguments)`` from assistant messages'
    ``tool_calls`` so an evicted tool result can be stubbed with the call that
    produced it. Handles both ToolCall objects and dicts."""
    out: dict[str, tuple[str, dict]] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if isinstance(tc, dict):
                cid, name, args = tc.get("id"), tc.get("name", ""), tc.get("arguments") or {}
            else:
                cid = getattr(tc, "id", None)
                name = getattr(tc, "name", "")
                args = getattr(tc, "arguments", None) or {}
            if cid:
                out[cid] = (name, args)
    return out


def _prune_tool_messages_midturn(
    messages: List[Message], *, keep_recent: int, protect_call_ids: set
) -> Tuple[List[Message], int]:
    """Collapse old ``tool``-role messages to a re-read stub so a growing tool
    loop stops replaying full results every round (the mid-turn half of context
    pruning). Protects the ``keep_recent`` most recent tool results and every
    result whose ``tool_call_id`` is in ``protect_call_ids`` (the current round —
    identified by id so dedup/reordering can't shift the boundary).
    ``tool_call_id`` is preserved so provider tool-call/result pairing stays
    intact.

    Returns ``(messages, pruned_count)`` — a fresh list only when something was
    pruned, else the original list unchanged.
    """
    from chat.tool_stub import build_tool_result_stub

    tool_indices = [i for i, m in enumerate(messages) if getattr(m, "role", "") == "tool"]
    if not tool_indices:
        return messages, 0
    protected = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
    protected |= {
        i for i in tool_indices
        if getattr(messages[i], "tool_call_id", None) in protect_call_ids
    }
    eligible = [
        i for i in tool_indices
        if i not in protected
        # Already stubbed on a prior round — skip (its body no longer costs much).
        and not (messages[i].content or "").startswith("[Earlier result of ")
    ]
    if not eligible:
        return messages, 0

    call_meta = _pipeline_call_meta(messages)
    out = list(messages)
    for i in eligible:
        name, args = call_meta.get(getattr(out[i], "tool_call_id", None), ("", None))
        out[i] = out[i].model_copy(update={"content": build_tool_result_stub(name, args)})
    return out, len(eligible)


class SimpleChatPipeline(BasePipeline):
    """Single pipeline: LLM-driven tool calling via bind_tools(), else delegate to ChatModel."""

    id = "simple_chat"
    capabilities = {"streaming": True, "tools": True}

    def __init__(self, max_tool_iterations: int = 100) -> None:
        self.max_tool_iterations = max_tool_iterations

    def _get_max_iterations(self, request: ChatRequest) -> int:
        """Return max tool iterations, allowing per-request override via params."""
        override = (request.params or {}).get("max_tool_iterations")
        if override is not None:
            return int(override)
        return self.max_tool_iterations

    def run(self, request: ChatRequest) -> ChatResponse:
        tool_names = request.tools or []
        if not tool_names:
            return self._run_no_tools(request)

        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")
        tools = self._resolve_tools(tool_names, request.context)
        req = request.model_copy(update={"tool_schemas": tools})

        max_iter = self._get_max_iterations(request)
        return self._run_tool_loop(create_chat_model(req.model), req, tools, max_iter)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        tool_names = request.tools or []
        if not tool_names:
            model_name = request.model
            if not model_name:
                raise ValueError("request.model must be set by the service before calling pipeline")
            for event in create_chat_model(model_name).stream(request):
                if event.event_type == "message_end":
                    _keep_replay_blocks(event.data)
                yield event
            return

        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")
        tools = self._resolve_tools(tool_names, request.context)
        req = request.model_copy(update={"tool_schemas": tools})

        model = create_chat_model(req.model)
        run_id = req.context.run_id if req.context else ""
        max_iter = self._get_max_iterations(request)
        yield from self._stream_with_tools(model, req, tools, run_id, max_iter)

    @staticmethod
    def _build_message_end_data(response: ChatResponse) -> dict:
        """Build message_end event data from a ChatResponse.

        Uses key names matching the provider stream format (input_tokens/output_tokens)
        so log_stream() can extract them correctly, and includes response metadata.
        """
        data: dict = {}
        if response.usage:
            usage_map = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cached_tokens": response.usage.cached_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "cost_usd": response.usage.cost_usd,
            }
            data.update({k: v for k, v in usage_map.items() if v is not None})
        if response.metadata:
            for key in (
                "response_metadata", "stop_reason", "provider_model_id", "output_truncated",
            ):
                if key in response.metadata:
                    data[key] = response.metadata[key]
        return data

    def _resolve_tools(self, tool_names: List[str], context: RunContext | None = None) -> List[ContextAwareTool]:
        registry = get_tool_registry()
        agent_kind = getattr(context, "agent_kind", "main") if context else "main"
        tools = []
        for name in tool_names:
            t = registry.get_tool(name)
            if t is None:
                # A skill may carry a tool name that no longer exists (e.g. an
                # imported/older skill). Skip it rather than crashing the turn —
                # the request just proceeds without that tool.
                logger.warning("Skipping unknown tool name in request: %r", name)
                continue
            # Defence in depth: tool selection is gated upstream by audience
            # (preferences / resolve_subagent_tools), but enforce it here too so
            # a bug upstream can never run a tool in the wrong agent kind (e.g. a
            # main-only canvas tool inside a sub-agent).
            audience = getattr(t, "audience", "shared")
            if audience != "shared" and audience != agent_kind:
                logger.warning(
                    "Dropping tool %r (audience=%s) for agent_kind=%s",
                    name, audience, agent_kind,
                )
                continue
            # Copy to avoid shared state across concurrent requests
            copy = t.model_copy()
            if context:
                copy.set_context(context)
            tools.append(copy)
        return tools

    def _run_no_tools(self, request: ChatRequest) -> ChatResponse:
        model_name = request.model
        if not model_name:
            raise ValueError("request.model must be set by the service before calling pipeline")
        chat_model = create_chat_model(model_name)
        return chat_model.generate(request)

    @staticmethod
    def _execute_tool_calls(
        tool_calls: List[ToolCall],
        tool_by_name: dict[str, ContextAwareTool],
        context: RunContext | None = None,
    ) -> List[Tuple[ToolCall, str]]:
        """Execute a batch of tool calls and return (tool_call, result_json) pairs.

        When multiple tool calls are present, they execute concurrently via
        ThreadPoolExecutor (capped at 4 workers).  Single calls skip the pool.
        A sub-agent's calls always go through the pool, each under its tool's
        timeout (_run_round_with_timeouts); the main agent waits as long as a
        tool takes.
        """

        def _run_one(tc: ToolCall) -> Tuple[ToolCall, str]:
            from django.db import close_old_connections

            tool = tool_by_name.get(tc.name)
            if not tool:
                return (tc, json.dumps({"error": f"Unknown tool: {tc.name}"}))
            # A model imitating its own trimmed history can send the "[Omitted
            # from this transcript…]" marker back as content; running that would
            # overwrite a canvas/scratchpad/deck with the placeholder.
            echoed = find_echoed_marker(tc.arguments or {})
            if echoed:
                logger.warning("Refused %s: marker echoed in argument %s", tc.name, echoed)
                return (tc, json.dumps({
                    "status": "error",
                    "message": (
                        f"Not run: argument '{echoed}' contains the '[Omitted from this "
                        "transcript to save space …]' placeholder. That text only marks "
                        "where your earlier calls were shortened in your history — it is "
                        "not content, and those earlier calls ran with the full text. "
                        "Call the tool again with the real content written out in full."
                    ),
                }))
            try:
                result = tool.invoke(tc.arguments)
                result_str = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                # Surface the crash: without this an unexpected tool exception is
                # silently returned to the model as {"error": ...} with no log or
                # Sentry event, so a regression is invisible until someone reads
                # a chat transcript.
                logger.exception("Tool %s raised during execution", tc.name)
                result_str = json.dumps({"error": str(e)})
            finally:
                close_old_connections()
            total_len = len(result_str)
            if total_len > MAX_TOOL_RESULT_CHARS:
                over = total_len - MAX_TOOL_RESULT_CHARS
                logger.warning(
                    "Tool result truncated: tool=%s len=%d (%d over the %d-char limit)",
                    tc.name, total_len, over, MAX_TOOL_RESULT_CHARS,
                )
                result_str = result_str[:MAX_TOOL_RESULT_CHARS] + (
                    f"\n\n[TOOL RESULT TRUNCATED: output was {total_len:,} chars, "
                    f"{over:,} over the {MAX_TOOL_RESULT_CHARS:,}-char limit. "
                    "The content above is cut off; narrow the request or "
                    "paginate to see more.]"
                )
            return (tc, result_str)

        if tool_calls and context is not None and context.agent_kind == "subagent":
            return _run_round_with_timeouts(tool_calls, tool_by_name, _run_one, context)

        if len(tool_calls) <= 1:
            return [_run_one(tc) for tc in tool_calls]

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    @staticmethod
    def _append_pending_native_assets(new_messages: List[Message], req: ChatRequest) -> None:
        """Drain context.pending_native_assets into a user message so the model
        can view the file(s) it asked for on its next turn.

        Each queued item carries a ``kind`` (``"image"`` — the default and the
        common case — or ``"pdf"``). Native content blocks are used when the
        model accepts that modality (image / pdf); otherwise the item degrades to
        its text description (image) or extracted text (pdf). No-op when nothing
        was queued this turn, so blast radius is small.
        """
        ctx = req.context
        pending = list(getattr(ctx, "pending_native_assets", None) or [])
        if not pending:
            return
        ctx.pending_native_assets.clear()

        from llm.display import supports_modality

        model = req.model or ""
        _provider_cache: list = []

        def _provider():
            if not _provider_cache:
                from llm.core.model_factory import detect_provider
                _provider_cache.append(detect_provider(model))
            return _provider_cache[0]

        def _tag_native_block(block: dict, item: dict) -> None:
            # Private markers the send-time enforcer reads to classify a native
            # block's pathway (skill/dataroom/attachment) and size for priority
            # pruning. Stripped before the provider sees the block. _wf_est_tokens
            # lets count_tokens charge the provider's real token cost.
            block["_wf_pathway"] = item.get("_pathway", "dataroom")
            block["_wf_b64len"] = len(item.get("b64") or "")
            try:
                from django.conf import settings as _s

                if item.get("kind") == "pdf":
                    pages = item.get("pages") or 1
                    block["_wf_est_tokens"] = int(pages) * int(getattr(_s, "PDF_PAGE_TOKENS", 2_300))
                else:
                    block["_wf_est_tokens"] = int(getattr(_s, "VISION_IMAGE_TOKENS", 1_600))
            except Exception:
                pass

        blocks: list = [{"type": "text", "text": "Here are the file(s) you requested to view:"}]
        for item in pending:
            kind = item.get("kind", "image")
            if kind == "pdf":
                filename = item.get("filename") or "document.pdf"
                if model and supports_modality(model, "pdf"):
                    from chat.services import build_pdf_content_block

                    block = build_pdf_content_block(item["b64"], filename, _provider())
                    _tag_native_block(block, item)
                    blocks.append(block)
                    desc = item.get("description")
                    if desc:
                        blocks.append({"type": "text", "text": f"(above: {filename} — {desc})"})
                else:
                    # Model can't take PDF natively — fall back to extracted text.
                    from chat.services import build_text_content_block

                    blocks.append(build_text_content_block(item.get("extracted_text") or "", filename))
                continue
            # kind == "image"
            if model and supports_modality(model, "image"):
                from chat.services import build_image_content_block

                block = build_image_content_block(item["b64"], item["media_type"], _provider())
                _tag_native_block(block, item)
                blocks.append(block)
                desc = item.get("description")
                if desc:
                    blocks.append({"type": "text", "text": f"(above: {desc})"})
            else:
                blocks.append({
                    "type": "text",
                    "text": f"[Image not shown — model has no vision. Description: {item.get('description', '')}]",
                })
        new_messages.append(Message(role="user", content=blocks))

    @staticmethod
    def _append_pending_skill_instructions(new_messages: List[Message], req: ChatRequest) -> None:
        """Drain context.pending_skill_instructions into a user message so a skill
        the agent just attached takes effect this turn.

        Mirrors _append_pending_native_assets: chat_skill_attach pushes the rendered
        instruction block(s) of newly-attached skills onto the shared context, and
        the loop injects them as an ephemeral (non-persisted) user message on the
        next iteration. No-op when nothing was queued (the common case).
        """
        ctx = req.context
        pending = list(getattr(ctx, "pending_skill_instructions", None) or [])
        if not pending:
            return
        ctx.pending_skill_instructions.clear()
        new_messages.append(Message(
            role="user",
            content=(
                "The following skill(s) were just attached and are now active for "
                "the rest of this conversation. Follow their instructions:\n\n"
                + "\n\n".join(pending)
            ),
        ))

    def _expand_tools_from_context(
        self,
        tools: List[ContextAwareTool],
        tool_by_name: dict[str, ContextAwareTool],
        req: ChatRequest,
    ) -> List[ContextAwareTool]:
        """Union context.added_tool_names into the live tool set for the next
        iteration, so a skill the agent just attached is callable this turn.

        chat_skill_attach pushes tool names onto the shared context; here we resolve
        them to tool objects (via _resolve_tools, which enforces audience and binds
        context), extend the dispatch map in place, and return the enlarged tools
        list. The caller folds it into req.tool_schemas — the model re-binds tools
        from req.tool_schemas on every call, so the enlarged set reaches the provider
        next iteration. Additive-only and deduped against tools already active.
        No-op when nothing was queued (the common case).
        """
        ctx = req.context
        pending = list(getattr(ctx, "added_tool_names", None) or [])
        if not pending:
            return tools
        ctx.added_tool_names.clear()
        new_names = [n for n in dict.fromkeys(pending) if n not in tool_by_name]
        if not new_names:
            return tools
        new_tools = self._resolve_tools(new_names, ctx)
        if not new_tools:
            return tools
        tool_by_name.update({t.name: t for t in new_tools})
        return tools + new_tools

    @staticmethod
    def _record_tool_round_stats(context, results) -> None:
        """Accumulate per-turn observability counters (tool-call count + total
        tokens of tool results) onto the RunContext; the logger surfaces them on
        LLMCallLog. Best-effort — never let accounting break a turn."""
        if context is None:
            return
        try:
            from core.tokens import count_tokens

            context.bump_stat("tool_calls", len(results))
            context.bump_stat(
                "tool_result_tokens",
                sum(count_tokens(r) for _, r in results),
            )
        except Exception:  # pragma: no cover - metrics must never break a turn
            logger.debug("Failed to record tool-round stats", exc_info=True)

    @staticmethod
    def _record_outbound_estimate(context, req: ChatRequest) -> None:
        """Accumulate our pre-send estimate of this request's input tokens onto
        the RunContext (summed per round), so the logger can record estimate-vs-
        actual drift for the turn. Cheap: pruning keeps req.messages bounded
        (old tool results are already stubs). Best-effort."""
        if context is None:
            return
        try:
            from core.tokens import estimate_chat_request_tokens

            context.bump_stat(
                "estimated_input_tokens",
                estimate_chat_request_tokens(req.messages, req.tool_schemas),
            )
        except Exception:  # pragma: no cover - metrics must never break a turn
            logger.debug("Failed to record outbound estimate", exc_info=True)

    def _midturn_ceiling(self, req: ChatRequest) -> int:
        """The per-request input-token ceiling for this model + org aim + effort
        (``min(aim, window) - output_reservation - margin``)."""
        from llm.context_budget import request_input_ceiling

        params = req.params or {}
        return request_input_ceiling(
            req.model,
            params.get("max_context_tokens"),
            params.get("thinking_level"),
        )

    @staticmethod
    def _final_request(
        req: ChatRequest, edits: EditPointManager, max_iterations: int,
    ) -> ChatRequest:
        """The last, tool-less call after the loop ends (round budget or truncation).

        Tools stay bound and ``tool_choice`` is set to "none" (passed as the
        ``_tool_choice`` param; each provider adapter maps it) instead of dropping
        ``tool_schemas`` — removing the tool set would change the request prefix,
        breaking the prompt cache and invalidating Anthropic's replayed thinking.
        A sub-agent is told its canvas is returned as-is, so the reply is a short
        summary plus what it did not get to (the orchestrator can continue it).
        """
        agent_kind = getattr(req.context, "agent_kind", "main") if req.context else "main"
        if agent_kind == "subagent":
            stop = (getattr(req.context, "observability", None) or {}).get("loop_stop")
            if stop == "truncation":
                # Rounds may well be left; what ran out is the loop's patience
                # with cut-off output. Saying "budget used up" here made the
                # sub-agent report a budget problem it did not have.
                text = (
                    "[Your last replies were cut off before they were complete, so tools "
                    "are off.] Write your final reply now, short and plain: a summary of "
                    'your findings and a "Not covered" list of what you did not get to. '
                    "Your canvas is returned as it is — do not repeat it in your reply."
                )
            else:
                text = (
                    f"[Round {max_iterations + 1}/{max_iterations}: your round budget is used "
                    "up. Tools are off.] Write your final reply now: a short summary of your "
                    'findings and a "Not covered" list of what you did not get to. Your canvas '
                    "is returned as it is — do not repeat it in your reply."
                )
        else:
            text = (
                "You have reached the tool-use limit. Do NOT call any more tools. "
                "Using the information you have gathered so far, provide your "
                "comprehensive final response now as plain text."
            )
        late = _drain_late_tool_results(req.context)
        if late:
            text = f"{late}\n\n{text}"
        final_messages = list(req.messages) + [Message(role="user", content=text)]
        final_messages = edits.finish_final(final_messages, req)
        return req.model_copy(update={
            "messages": final_messages,
            "params": {**(req.params or {}), "_tool_choice": "none"},
        })

    def _run_tool_loop(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[ContextAwareTool],
        max_iterations: int | None = None,
    ) -> ChatResponse:
        tool_by_name = {t.name: t for t in tools}
        req = request
        cancel_check = (request.params or {}).get("_cancel_check")
        edits = EditPointManager(self, request, tool_by_name)

        ctx = request.context

        agg_input = 0
        agg_output = 0
        agg_total = 0
        agg_cached = 0
        agg_cache_write = 0
        agg_reasoning = 0
        agg_cost: float | None = None

        def _accumulate(usage: Usage | None) -> None:
            nonlocal agg_input, agg_output, agg_total, agg_cached, \
                agg_cache_write, agg_reasoning, agg_cost
            if not usage:
                return
            agg_input += usage.prompt_tokens or 0
            agg_output += usage.completion_tokens or 0
            agg_total += usage.total_tokens or 0
            agg_cached += usage.cached_tokens or 0
            agg_cache_write += usage.cache_write_tokens or 0
            agg_reasoning += usage.reasoning_tokens or 0
            if usage.cost_usd is not None:
                agg_cost = (agg_cost or 0.0) + usage.cost_usd

        def _with_aggregate(response: ChatResponse) -> ChatResponse:
            return response.model_copy(update={
                "usage": Usage(
                    prompt_tokens=agg_input,
                    completion_tokens=agg_output,
                    total_tokens=agg_total,
                    cached_tokens=agg_cached or None,
                    cache_write_tokens=agg_cache_write or None,
                    reasoning_tokens=agg_reasoning or None,
                    cost_usd=agg_cost,
                ),
            })

        effective_max = max_iterations if max_iterations is not None else self.max_tool_iterations
        truncated_streak = 0
        for i in range(effective_max):
            if cancel_check and cancel_check():
                _set_loop_stop(ctx, "cancelled")
                return ChatResponse(
                    message=Message(role="assistant", content="[Cancelled]"),
                    metadata={"stop_reason": "cancelled"},
                )
            _count_round(ctx)

            self._record_outbound_estimate(req.context, req)
            response = chat_model.generate(req)
            _accumulate(response.usage)
            record_thinking_drops(
                req.context, (response.metadata or {}).get("response_metadata"),
            )
            msg = response.message
            agent_kind = getattr(req.context, "agent_kind", "main") if req.context else "main"

            if (response.metadata or {}).get("output_truncated"):
                truncated_streak += 1
                stop_reason = str((response.metadata or {}).get("stop_reason") or "output cap")
                usable_calls = [tc for tc in (msg.tool_calls or []) if tc.name and tc.id]
                if req.context:
                    req.context.bump_stat("output_truncated_rounds")
                logger.warning(
                    "Tool-loop round truncated (stop_reason=%s) at iteration %d; "
                    "skipping %d tool call(s) model=%s",
                    stop_reason, i, len(usable_calls), req.model,
                )
                visible = bool(str(msg.content or "").strip())
                if usable_calls or agent_kind != "main" or not visible:
                    # Never run a cut-off call (its arguments may be silently
                    # truncated); pair each with an error result and tell the
                    # model to re-issue a shorter one. Appends only.
                    prior_len = len(req.messages)
                    new_messages = list(req.messages)
                    results: list = []
                    if usable_calls:
                        new_messages.append(msg.model_copy(update={"tool_calls": usable_calls}))
                        results = [
                            (tc, _truncated_call_result(tc.name, stop_reason))
                            for tc in usable_calls
                        ]
                        for tc, result_str in results:
                            new_messages.append(
                                Message(role="tool", content=result_str, tool_call_id=tc.id)
                            )
                        self._record_tool_round_stats(req.context, results)
                    notice = _with_late_results(_truncated_round_notice(
                        stop_reason,
                        _iteration_notice(i, effective_max, agent_kind),
                    ), req.context)
                    new_messages, tools = edits.finish_round(
                        new_messages, req, tools,
                        prior_len=prior_len,
                        real_input_tokens=(response.usage.prompt_tokens if response.usage else 0),
                        round_call_ids={tc.id for tc in usable_calls},
                        notice=notice,
                    )
                    req = req.model_copy(update={"messages": new_messages, "tool_schemas": tools})
                    if truncated_streak >= _MAX_CONSECUTIVE_TRUNCATED_ROUNDS:
                        logger.warning(
                            "Tool loop hit %d consecutive truncated rounds at iteration %d; "
                            "stripping tools for final generate",
                            truncated_streak, i,
                        )
                        _set_loop_stop(ctx, "truncation")
                        break
                    continue
                # A cut plain-text answer for the main agent ends the turn as
                # usual (the flag rides on the response metadata).
                msg = msg.model_copy(update={"tool_calls": None})
                response = response.model_copy(update={"message": msg})
            else:
                truncated_streak = 0

            if not msg.tool_calls:
                _set_loop_stop(ctx, "answered")
                return _with_aggregate(response)

            prior_len = len(req.messages)
            new_messages = list(req.messages) + [msg]
            results = self._execute_tool_calls(msg.tool_calls, tool_by_name, req.context)
            for tc, result_str in results:
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )
            self._record_tool_round_stats(req.context, results)

            notice = _with_late_results(_iteration_notice(
                i, effective_max,
                getattr(req.context, "agent_kind", "main") if req.context else "main",
            ), req.context)
            new_messages, tools = edits.finish_round(
                new_messages, req, tools,
                prior_len=prior_len,
                real_input_tokens=(response.usage.prompt_tokens if response.usage else 0),
                round_call_ids={tc.id for tc in msg.tool_calls},
                notice=notice,
            )
            req = req.model_copy(update={"messages": new_messages, "tool_schemas": tools})
        else:
            logger.warning(
                "Tool loop exhausted %d iterations; disabling tools for final generate",
                effective_max,
            )
            _set_loop_stop(ctx, "budget")

        _count_round(ctx)
        final_req = self._final_request(req, edits, effective_max)
        self._record_outbound_estimate(final_req.context, final_req)
        response = chat_model.generate(final_req)
        _accumulate(response.usage)
        record_thinking_drops(
            final_req.context, (response.metadata or {}).get("response_metadata"),
        )
        if response.message.tool_calls:
            logger.warning(
                "Model returned %d tool call(s) on the final tool-less call; ignoring them",
                len(response.message.tool_calls),
            )
            response = response.model_copy(update={
                "message": response.message.model_copy(update={"tool_calls": None}),
            })
        return _with_aggregate(response)

    def _stream_with_tools(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[ContextAwareTool],
        run_id: str,
        max_iterations: int,
    ) -> Iterator[StreamEvent]:
        """Stream tokens in real-time on every iteration, executing tools between rounds.

        Instead of generate() + fake stream events, this uses chat_model.stream()
        so tokens arrive in real-time. After each stream completes, the accumulated
        message_end data is checked for tool_calls. If present, tools execute
        (in parallel when multiple) and a new iteration begins.
        """
        tool_by_name = {t.name: t for t in tools}
        req = request
        sequence = 1
        params = request.params or {}
        edits = EditPointManager(self, request, tool_by_name)
        cancel_event = params.get("_cancel_event")
        # Sub-agents cancel cooperatively via a DB-poll callable (_cancel_check)
        # rather than the consumer's threading.Event; honor both. Polled only at
        # per-iteration boundaries below (a DB query per token would be wasteful);
        # the per-chunk check stays event-only.
        cancel_check = params.get("_cancel_check")

        def _is_cancelled() -> bool:
            return bool(
                (cancel_event is not None and cancel_event.is_set())
                or (cancel_check is not None and cancel_check())
            )

        ctx = request.context

        # Aggregate usage across all iterations.
        # Cost stays None until at least one iteration reports it, so unknown-pricing
        # models log NULL (like the non-streaming path) instead of 0.0.
        agg_input_tokens = 0
        agg_output_tokens = 0
        agg_total_tokens = 0
        agg_cached_tokens = 0
        agg_cache_write_tokens = 0
        agg_reasoning_tokens = 0
        agg_cost_usd: float | None = None

        def _do_stream_iteration(req, sequence):
            """Stream one LLM call, forwarding events; the final sentinel is
            ``(end_data, sequence, saw_error)``."""
            end_data = {}
            saw_error = False
            # Every streamed chunk is progress for the sub-agent heartbeat (a
            # throttled DB stamp; a no-op for the main agent), so a long but
            # live generation is never mistaken for a hung run.
            note_activity = getattr(ctx, "note_activity", None)
            for event in chat_model.stream(req):
                if note_activity is not None:
                    note_activity()
                if event.event_type == "message_end":
                    end_data = dict(event.data)
                else:
                    if event.event_type == "error":
                        saw_error = True
                    # Re-sequence with pipeline's sequence/run_id
                    yield StreamEvent(
                        event_type=event.event_type,
                        data=event.data,
                        sequence=sequence,
                        run_id=run_id,
                    )
                    sequence += 1
                # Per-token break honors only the in-memory cancel_event (the
                # main-chat Stop). Sub-agents cancel via _cancel_check, a DB poll
                # that must NOT run per token — they break at the iteration
                # boundaries below (_is_cancelled) instead.
                if cancel_event and cancel_event.is_set():
                    break
            # Yield a sentinel to pass end_data back
            yield (end_data, sequence, saw_error)

        def _usage_message_end() -> StreamEvent:
            """A message_end carrying only the accumulated usage (no content),
            emitted on error paths so the ERROR row logs real tokens/cost from
            earlier tool-loop iterations instead of NULL."""
            return StreamEvent(
                event_type="message_end",
                data={
                    "input_tokens": agg_input_tokens,
                    "output_tokens": agg_output_tokens,
                    "total_tokens": agg_total_tokens,
                    "cached_tokens": agg_cached_tokens or None,
                    "cache_write_tokens": agg_cache_write_tokens or None,
                    "reasoning_tokens": agg_reasoning_tokens or None,
                    "cost_usd": agg_cost_usd,
                },
                sequence=sequence,
                run_id=run_id,
            )

        truncated_streak = 0
        for i in range(max_iterations):
            if _is_cancelled():
                _set_loop_stop(ctx, "cancelled")
                return
            _count_round(ctx)
            # Stream from the model, forwarding all events except message_end
            end_data = {}
            saw_error = False
            self._record_outbound_estimate(req.context, req)
            for item in _do_stream_iteration(req, sequence):
                if isinstance(item, StreamEvent):
                    yield item
                else:
                    # Sentinel: (end_data, updated_sequence, saw_error)
                    end_data, sequence, saw_error = item

            if saw_error:
                # The provider already yielded an error event and stopped. Emit a
                # message_end carrying the usage accumulated from earlier
                # iterations so the ERROR row logs real tokens/cost instead of
                # NULL: log_stream keys status off the error event (ERROR) and
                # reads usage from this message_end. Safe for the UI — the error
                # handler already finalized the stream element, so a trailing
                # message_end is a no-op there.
                yield _usage_message_end()
                return

            if _is_cancelled():
                _set_loop_stop(ctx, "cancelled")
                # Cancelled mid-stream: return WITHOUT a message_end so log_stream
                # sees an interrupted stream (-> CANCELLED with token estimation),
                # not a normally-completed final iteration. Closes the race where
                # an empty end_data looked like "no tool calls = final".
                return

            # Accumulate usage from this iteration
            agg_input_tokens += end_data.get("input_tokens") or 0
            agg_output_tokens += end_data.get("output_tokens") or 0
            agg_total_tokens += end_data.get("total_tokens") or 0
            agg_cached_tokens += end_data.get("cached_tokens") or 0
            agg_cache_write_tokens += end_data.get("cache_write_tokens") or 0
            agg_reasoning_tokens += end_data.get("reasoning_tokens") or 0
            iter_cost = end_data.get("cost_usd")
            if iter_cost is not None:
                agg_cost_usd = (agg_cost_usd or 0.0) + iter_cost

            record_thinking_drops(req.context, end_data.get("response_metadata"))
            agent_kind = getattr(req.context, "agent_kind", "main") if req.context else "main"

            # Check for tool calls
            tool_call_dicts = end_data.get("tool_calls")

            if end_data.get("output_truncated"):
                truncated_streak += 1
                stop_reason = str(end_data.get("stop_reason") or "output cap")
                usable = _usable_tool_call_dicts(tool_call_dicts)
                if ctx:
                    ctx.bump_stat("output_truncated_rounds")
                logger.warning(
                    "Streaming tool-loop round truncated (stop_reason=%s) at iteration %d; "
                    "skipping %d tool call(s) model=%s run_id=%s",
                    stop_reason, i, len(usable), req.model, run_id,
                )
                visible = bool(str(end_data.get("content") or "").strip())
                if usable or agent_kind != "main" or not visible:
                    # Never run a cut-off call (its arguments may be silently
                    # truncated); pair each with an error result and tell the
                    # model to re-issue a shorter one. Appends only.
                    skipped_calls = [
                        ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
                        for tc in usable
                    ]
                    prior_len = len(req.messages)
                    new_messages = list(req.messages)
                    if skipped_calls:
                        new_messages.append(
                            Message(
                                role="assistant",
                                content=end_data.get("content", ""),
                                tool_calls=skipped_calls,
                                metadata=_assistant_replay_meta(end_data),
                            )
                        )
                        for tc in skipped_calls:
                            tool = tool_by_name.get(tc.name)
                            yield StreamEvent(
                                event_type="tool_start",
                                data={
                                    "tool_name": tc.name,
                                    "tool_call_id": tc.id,
                                    "arguments": tc.arguments,
                                    "display_label": tool.start_label if tool else "Working...",
                                },
                                sequence=sequence,
                                run_id=run_id,
                            )
                            sequence += 1
                        results = [
                            (tc, _truncated_call_result(tc.name, stop_reason))
                            for tc in skipped_calls
                        ]
                        self._record_tool_round_stats(req.context, results)
                        for tc, result_str in results:
                            yield StreamEvent(
                                event_type="tool_end",
                                data={
                                    "tool_name": tc.name,
                                    "tool_call_id": tc.id,
                                    "result": result_str,
                                    "display_label": _TRUNCATED_TOOL_LABEL,
                                    "is_error": True,
                                },
                                sequence=sequence,
                                run_id=run_id,
                            )
                            sequence += 1
                            new_messages.append(
                                Message(role="tool", content=result_str, tool_call_id=tc.id)
                            )
                    notice = _with_late_results(_truncated_round_notice(
                        stop_reason,
                        _iteration_notice(i, max_iterations, agent_kind),
                    ), req.context)
                    new_messages, tools = edits.finish_round(
                        new_messages, req, tools,
                        prior_len=prior_len,
                        real_input_tokens=(end_data.get("input_tokens") or 0),
                        round_call_ids={tc.id for tc in skipped_calls},
                        notice=notice,
                    )
                    req = req.model_copy(update={"messages": new_messages, "tool_schemas": tools})
                    if truncated_streak >= _MAX_CONSECUTIVE_TRUNCATED_ROUNDS:
                        logger.warning(
                            "Streaming tool loop hit %d consecutive truncated rounds at "
                            "iteration %d; stripping tools for final stream",
                            truncated_streak, i,
                        )
                        _set_loop_stop(ctx, "truncation")
                        break
                    continue
                # A cut plain-text answer for the main agent: the user already
                # saw the text, so end the turn as usual (the flag rides on the
                # final message_end) instead of retrying into the same bubble.
                tool_call_dicts = None
            else:
                truncated_streak = 0

            if not tool_call_dicts:
                _set_loop_stop(ctx, "answered")
                # Final iteration — response already streamed. Emit message_end with aggregates.
                end_data["input_tokens"] = agg_input_tokens
                end_data["output_tokens"] = agg_output_tokens
                end_data["total_tokens"] = agg_total_tokens
                end_data["cached_tokens"] = agg_cached_tokens or None
                end_data["cache_write_tokens"] = agg_cache_write_tokens or None
                end_data["reasoning_tokens"] = agg_reasoning_tokens or None
                end_data["cost_usd"] = agg_cost_usd
                # Clean internal fields from end_data
                end_data.pop("content", None)
                _keep_replay_blocks(end_data)
                end_data.pop("tool_calls", None)
                yield StreamEvent(
                    event_type="message_end",
                    data=end_data,
                    sequence=sequence,
                    run_id=run_id,
                )
                return

            # Reconstruct tool calls and assistant message for history
            parsed_tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
                for tc in tool_call_dicts
            ]
            prior_len = len(req.messages)
            new_messages = list(req.messages) + [
                Message(
                    role="assistant",
                    content=end_data.get("content", ""),
                    tool_calls=parsed_tool_calls,
                    metadata=_assistant_replay_meta(end_data),
                )
            ]

            # Emit tool_start for all tool calls (show spinners simultaneously)
            for tc in parsed_tool_calls:
                tool = tool_by_name.get(tc.name)
                yield StreamEvent(
                    event_type="tool_start",
                    data={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "arguments": tc.arguments,
                        "display_label": tool.start_label if tool else "Working...",
                    },
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1

            if _is_cancelled():
                _set_loop_stop(ctx, "cancelled")
                return

            # Execute tools (in parallel when multiple)
            results = self._execute_tool_calls(parsed_tool_calls, tool_by_name, req.context)
            self._record_tool_round_stats(req.context, results)

            # Emit tool_end for all results and append to history
            for tc, result_str in results:
                tool = tool_by_name.get(tc.name)
                # Tools report failure as {"status": "error", ...}; the client
                # keys the card's icon on this flag so a refusal never renders
                # as a green check. Shared with the history view (reload).
                display_label, is_error = tool_end_display(tool, result_str)
                timed_out = _timed_out_label(result_str)
                if timed_out:
                    display_label, is_error = timed_out, True
                yield StreamEvent(
                    event_type="tool_end",
                    data={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "result": result_str,
                        "display_label": display_label,
                        "is_error": is_error,
                    },
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            notice = _with_late_results(_iteration_notice(
                i, max_iterations,
                getattr(req.context, "agent_kind", "main") if req.context else "main",
            ), req.context)
            new_messages, tools = edits.finish_round(
                new_messages, req, tools,
                prior_len=prior_len,
                real_input_tokens=(end_data.get("input_tokens") or 0),
                round_call_ids={tc.id for tc in parsed_tool_calls},
                notice=notice,
            )
            req = req.model_copy(update={"messages": new_messages, "tool_schemas": tools})
        else:
            logger.warning(
                "Streaming tool loop exhausted %d iterations; disabling tools for final stream",
                max_iterations,
            )
            _set_loop_stop(ctx, "budget")

        _count_round(ctx)
        final_req = self._final_request(req, edits, max_iterations)
        self._record_outbound_estimate(final_req.context, final_req)
        saw_error = False
        for item in _do_stream_iteration(final_req, sequence):
            if isinstance(item, StreamEvent):
                yield item
            else:
                end_data, sequence, saw_error = item
        if saw_error:
            # Error event already forwarded. Emit a usage-only message_end so the
            # ERROR row logs the tokens/cost accumulated across the tool loop
            # (log_stream still logs ERROR from the error event).
            yield _usage_message_end()
            return
        record_thinking_drops(final_req.context, end_data.get("response_metadata"))
        # Add aggregate usage to final message_end
        agg_input_tokens += end_data.get("input_tokens") or 0
        agg_output_tokens += end_data.get("output_tokens") or 0
        agg_total_tokens += end_data.get("total_tokens") or 0
        agg_cached_tokens += end_data.get("cached_tokens") or 0
        agg_cache_write_tokens += end_data.get("cache_write_tokens") or 0
        agg_reasoning_tokens += end_data.get("reasoning_tokens") or 0
        iter_cost = end_data.get("cost_usd")
        if iter_cost is not None:
            agg_cost_usd = (agg_cost_usd or 0.0) + iter_cost
        end_data["input_tokens"] = agg_input_tokens
        end_data["output_tokens"] = agg_output_tokens
        end_data["total_tokens"] = agg_total_tokens
        end_data["cached_tokens"] = agg_cached_tokens or None
        end_data["cache_write_tokens"] = agg_cache_write_tokens or None
        end_data["reasoning_tokens"] = agg_reasoning_tokens or None
        end_data["cost_usd"] = agg_cost_usd
        end_data.pop("content", None)
        _keep_replay_blocks(end_data)
        if end_data.pop("tool_calls", None):
            logger.warning("Model returned tool call(s) on the final tool-less stream; ignoring them")
        yield StreamEvent(
            event_type="message_end",
            data=end_data,
            sequence=sequence,
            run_id=run_id,
        )


# Register so LLMService can resolve "simple_chat"
_registry = get_pipeline_registry()
_registry.register_pipeline(SimpleChatPipeline())


__all__ = ["SimpleChatPipeline"]
