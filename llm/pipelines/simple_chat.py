"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Tuple

from llm.core.interfaces import ChatModel
from llm.core.model_factory import create_chat_model
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import get_pipeline_registry
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent
from llm.types.context import RunContext
from llm.tools.interfaces import ContextAwareTool
from llm.tools.registry import get_tool_registry

logger = logging.getLogger(__name__)

# Hard cap on a single tool result entering the conversation history.
# ~50k tokens — fits comfortably inside the smallest registry context window
# (128k tokens) with room left for history. Enforced centrally in
# _execute_tool_calls so every tool (web, canvas, doc search, skills) is
# covered without per-tool logic.
MAX_TOOL_RESULT_CHARS = 200_000


def _safe_result_dict(result_str: str) -> dict | None:
    """Best-effort parse of a tool result into a dict for label computation.

    Returns None for non-JSON or non-dict results (e.g. tools that return
    markdown), in which case the caller falls back to the tool's static
    ``end_label``.
    """
    try:
        value = json.loads(result_str)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class SimpleChatPipeline(BasePipeline):
    """Single pipeline: LLM-driven tool calling via bind_tools(), else delegate to ChatModel."""

    id = "simple_chat"
    capabilities = {"streaming": True, "tools": True}

    def __init__(self, max_tool_iterations: int = 50) -> None:
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
            yield from create_chat_model(model_name).stream(request)
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
            for key in ("response_metadata", "stop_reason", "provider_model_id"):
                if key in response.metadata:
                    data[key] = response.metadata[key]
        return data

    def _resolve_tools(self, tool_names: List[str], context: RunContext | None = None) -> List[ContextAwareTool]:
        registry = get_tool_registry()
        tools = []
        for name in tool_names:
            t = registry.get_tool(name)
            if t is None:
                # A skill may carry a tool name that no longer exists (e.g. an
                # imported/older skill). Skip it rather than crashing the turn —
                # the request just proceeds without that tool.
                logger.warning("Skipping unknown tool name in request: %r", name)
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
    ) -> List[Tuple[ToolCall, str]]:
        """Execute a batch of tool calls and return (tool_call, result_json) pairs.

        When multiple tool calls are present, they execute concurrently via
        ThreadPoolExecutor (capped at 4 workers).  Single calls skip the pool.
        """

        def _run_one(tc: ToolCall) -> Tuple[ToolCall, str]:
            from django.db import close_old_connections

            tool = tool_by_name.get(tc.name)
            if not tool:
                return (tc, json.dumps({"error": f"Unknown tool: {tc.name}"}))
            try:
                result = tool.invoke(tc.arguments)
                result_str = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
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

        if len(tool_calls) <= 1:
            return [_run_one(tc) for tc in tool_calls]

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    @staticmethod
    def _append_pending_images(new_messages: List[Message], req: ChatRequest) -> None:
        """Drain context.pending_image_assets into a user message so the model
        can view them on its next turn.

        Native image blocks are used when the model accepts image input; a
        non-vision model gets the text descriptions instead. No-op when no tool
        queued any images this turn (the common case), so blast radius is small.
        """
        ctx = req.context
        pending = list(getattr(ctx, "pending_image_assets", None) or [])
        if not pending:
            return
        ctx.pending_image_assets.clear()

        from llm.display import supports_modality

        blocks: list = [{"type": "text", "text": "Here are the image(s) you requested to view:"}]
        if req.model and supports_modality(req.model, "image"):
            from chat.services import build_image_content_block
            from llm.core.model_factory import detect_provider

            provider = detect_provider(req.model)
            for item in pending:
                blocks.append(build_image_content_block(item["b64"], item["media_type"], provider))
                desc = item.get("description")
                if desc:
                    blocks.append({"type": "text", "text": f"(above: {desc})"})
        else:
            for item in pending:
                blocks.append({
                    "type": "text",
                    "text": f"[Image not shown — model has no vision. Description: {item.get('description', '')}]",
                })
        new_messages.append(Message(role="user", content=blocks))

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

        ctx = request.context
        deadline_dt = None
        if ctx and ctx.deadline_seconds:
            # Reserve 60s for the final tool-stripped generate + result storage,
            # so we don't race with the Celery soft time limit.
            margin = min(60, ctx.deadline_seconds // 4)
            deadline_dt = ctx.started_at + timedelta(seconds=ctx.deadline_seconds - margin)

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
        for i in range(effective_max):
            if cancel_check and cancel_check():
                return ChatResponse(
                    message=Message(role="assistant", content="[Cancelled]"),
                    metadata={"stop_reason": "cancelled"},
                )

            if deadline_dt and datetime.now(timezone.utc) >= deadline_dt:
                logger.warning(
                    "Tool loop deadline (%ds) reached at iteration %d; "
                    "stripping tools for final generate",
                    ctx.deadline_seconds, i,
                )
                break

            response = chat_model.generate(req)
            _accumulate(response.usage)
            msg = response.message
            if not msg.tool_calls:
                return _with_aggregate(response)

            new_messages = list(req.messages) + [msg]
            results = self._execute_tool_calls(msg.tool_calls, tool_by_name)
            for tc, result_str in results:
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            from chat.dedup import deduplicate_tool_results
            new_messages = deduplicate_tool_results(new_messages)

            remaining_iters = effective_max - i - 1
            time_running_low = (
                deadline_dt
                and (deadline_dt - datetime.now(timezone.utc)).total_seconds() <= 90
            )
            warnings = []
            if remaining_iters <= 2:
                if remaining_iters <= 1:
                    warnings.append("This is your last tool iteration.")
                else:
                    warnings.append("You have 2 tool iterations remaining.")
            if time_running_low:
                warnings.append("You are running low on time.")
            if warnings:
                new_messages.append(Message(
                    role="user",
                    content=(
                        f"IMPORTANT: {' '.join(warnings)} Wrap up your research "
                        "and prepare to deliver your comprehensive final response "
                        "as plain text."
                    ),
                ))

            self._append_pending_images(new_messages, req)
            req = req.model_copy(update={"messages": new_messages})
        else:
            logger.warning(
                "Tool loop exhausted %d iterations; stripping tools for final generate",
                effective_max,
            )

        final_messages = list(req.messages) + [
            Message(
                role="user",
                content=(
                    "You have reached the tool-use limit. Do NOT call any more tools. "
                    "Using the information you have gathered so far, provide your "
                    "comprehensive final response now as plain text."
                ),
            )
        ]
        final_req = req.model_copy(update={
            "tool_schemas": None,
            "messages": final_messages,
        })
        response = chat_model.generate(final_req)
        _accumulate(response.usage)
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
        deadline_dt = None
        if ctx and ctx.deadline_seconds:
            margin = min(60, ctx.deadline_seconds // 4)
            deadline_dt = ctx.started_at + timedelta(seconds=ctx.deadline_seconds - margin)

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
            for event in chat_model.stream(req):
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
                if cancel_event and cancel_event.is_set():
                    break
            # Yield a sentinel to pass end_data back
            yield (end_data, sequence, saw_error)

        for i in range(max_iterations):
            if _is_cancelled():
                return

            if deadline_dt and datetime.now(timezone.utc) >= deadline_dt:
                logger.warning(
                    "Streaming tool loop deadline (%ds) reached at iteration %d; "
                    "stripping tools for final stream",
                    ctx.deadline_seconds, i,
                )
                break
            # Stream from the model, forwarding all events except message_end
            end_data = {}
            saw_error = False
            for item in _do_stream_iteration(req, sequence):
                if isinstance(item, StreamEvent):
                    yield item
                else:
                    # Sentinel: (end_data, updated_sequence, saw_error)
                    end_data, sequence, saw_error = item

            if saw_error:
                # The provider already yielded an error event and stopped.
                # Don't fabricate a message_end — log_stream would record
                # this failed turn as SUCCESS.
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

            # Check for tool calls
            tool_call_dicts = end_data.get("tool_calls")
            if not tool_call_dicts:
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
                end_data.pop("content_blocks", None)
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
            assistant_content = end_data.get("content", "")
            new_messages = list(req.messages) + [
                Message(role="assistant", content=assistant_content, tool_calls=parsed_tool_calls)
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
                return

            # Execute tools (in parallel when multiple)
            results = self._execute_tool_calls(parsed_tool_calls, tool_by_name)

            # Emit tool_end for all results and append to history
            for tc, result_str in results:
                tool = tool_by_name.get(tc.name)
                display_label = "Done"
                if tool:
                    parsed = _safe_result_dict(result_str)
                    dynamic = tool.end_label_for_result(parsed) if parsed is not None else None
                    display_label = dynamic or tool.end_label
                yield StreamEvent(
                    event_type="tool_end",
                    data={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "result": result_str,
                        "display_label": display_label,
                    },
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            from chat.dedup import deduplicate_tool_results
            new_messages = deduplicate_tool_results(new_messages)

            remaining_iters = max_iterations - i - 1
            time_running_low = (
                deadline_dt
                and (deadline_dt - datetime.now(timezone.utc)).total_seconds() <= 90
            )
            warnings = []
            if remaining_iters <= 2:
                if remaining_iters <= 1:
                    warnings.append("This is your last tool iteration.")
                else:
                    warnings.append("You have 2 tool iterations remaining.")
            if time_running_low:
                warnings.append("You are running low on time.")
            if warnings:
                new_messages.append(Message(
                    role="user",
                    content=(
                        f"IMPORTANT: {' '.join(warnings)} Wrap up your research "
                        "and prepare to deliver your comprehensive final response "
                        "as plain text."
                    ),
                ))

            self._append_pending_images(new_messages, req)
            req = req.model_copy(update={"messages": new_messages})
        else:
            logger.warning(
                "Streaming tool loop exhausted %d iterations; stripping tools for final stream",
                max_iterations,
            )

        final_messages = list(req.messages) + [
            Message(
                role="user",
                content=(
                    "You have reached the tool-use limit. Do NOT call any more tools. "
                    "Using the information you have gathered so far, provide your "
                    "comprehensive final response now as plain text."
                ),
            )
        ]
        final_req = req.model_copy(update={
            "tool_schemas": None,
            "messages": final_messages,
        })
        saw_error = False
        for item in _do_stream_iteration(final_req, sequence):
            if isinstance(item, StreamEvent):
                yield item
            else:
                end_data, sequence, saw_error = item
        if saw_error:
            # Error event already forwarded; no synthetic message_end.
            return
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
        end_data.pop("content_blocks", None)
        end_data.pop("tool_calls", None)
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
