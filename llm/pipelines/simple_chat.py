"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
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

try:  # pragma: no cover
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:
    BaseCallbackHandler = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """Aggregates token usage across multiple LLM calls within a pipeline run.

    Collects usage_metadata from each on_llm_end callback and sums them,
    giving accurate totals for multi-turn tool-calling conversations.
    """

    def __init__(self) -> None:
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        """Accumulate usage from each LLM call completion."""
        for generation_list in (response.generations or []):
            for generation in generation_list:
                msg = getattr(generation, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if isinstance(usage, dict):
                    self.total_input_tokens += usage.get("input_tokens", 0)
                    self.total_output_tokens += usage.get("output_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)

    def get_aggregate_usage(self) -> dict:
        """Return the aggregated usage across all calls."""
        if self.total_tokens == 0 and self.total_input_tokens == 0:
            return {}
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }


class SimpleChatPipeline(BasePipeline):
    """Single pipeline: LLM-driven tool calling via bind_tools(), else delegate to ChatModel."""

    id = "simple_chat"
    capabilities = {"streaming": True, "tools": True}

    def __init__(self, max_tool_iterations: int = 25) -> None:
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

        # Create usage callback to aggregate across tool loop rounds
        usage_callback = UsageMetadataCallbackHandler()
        params = dict(req.params or {})
        params["_usage_callback"] = usage_callback
        req = req.model_copy(update={"params": params})

        max_iter = self._get_max_iterations(request)
        response = self._run_tool_loop(create_chat_model(req.model), req, tools, max_iter)

        # Attach aggregate usage if the response doesn't already have it
        aggregate = usage_callback.get_aggregate_usage()
        if aggregate and response.usage is None:
            response = response.model_copy(update={
                "usage": Usage(
                    prompt_tokens=aggregate.get("input_tokens"),
                    completion_tokens=aggregate.get("output_tokens"),
                    total_tokens=aggregate.get("total_tokens"),
                ),
            })

        return response

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
                raise ValueError(f"Unknown tool name: {name!r}")
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
            tool = tool_by_name.get(tc.name)
            if not tool:
                return (tc, json.dumps({"error": f"Unknown tool: {tc.name}"}))
            try:
                result = tool.invoke(tc.arguments)
                result_str = result if isinstance(result, str) else json.dumps(result)
            except Exception as e:
                result_str = json.dumps({"error": str(e)})
            return (tc, result_str)

        if len(tool_calls) <= 1:
            return [_run_one(tc) for tc in tool_calls]

        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
            futures = [pool.submit(_run_one, tc) for tc in tool_calls]
            return [f.result() for f in futures]

    def _run_tool_loop(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[ContextAwareTool],
        max_iterations: int | None = None,
    ) -> ChatResponse:
        tool_by_name = {t.name: t for t in tools}
        req = request

        for _ in range(max_iterations if max_iterations is not None else self.max_tool_iterations):
            response = chat_model.generate(req)
            msg = response.message
            if not msg.tool_calls:
                return response

            new_messages = list(req.messages) + [msg]
            results = self._execute_tool_calls(msg.tool_calls, tool_by_name)
            for tc, result_str in results:
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            req = req.model_copy(update={"messages": new_messages})

        # Max iterations reached: one final generate WITHOUT tools,
        # forcing the model to synthesise a text response.
        logger.warning(
            "Tool loop exhausted %d iterations; stripping tools for final generate",
            max_iterations if max_iterations is not None else self.max_tool_iterations,
        )
        final_req = req.model_copy(update={"tool_schemas": None})
        return chat_model.generate(final_req)

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

        # Aggregate usage across all iterations
        agg_input_tokens = 0
        agg_output_tokens = 0
        agg_total_tokens = 0
        agg_cost_usd = 0.0

        def _do_stream_iteration(req, sequence):
            """Stream one LLM call, forwarding events and returning (end_data, sequence)."""
            end_data = {}
            for event in chat_model.stream(req):
                if event.event_type == "message_end":
                    end_data = dict(event.data)
                else:
                    # Re-sequence with pipeline's sequence/run_id
                    yield StreamEvent(
                        event_type=event.event_type,
                        data=event.data,
                        sequence=sequence,
                        run_id=run_id,
                    )
                    sequence += 1
            # Yield a sentinel to pass end_data back
            yield (end_data, sequence)

        for _ in range(max_iterations):
            # Stream from the model, forwarding all events except message_end
            end_data = {}
            for item in _do_stream_iteration(req, sequence):
                if isinstance(item, StreamEvent):
                    yield item
                else:
                    # Sentinel: (end_data, updated_sequence)
                    end_data, sequence = item

            # Accumulate usage from this iteration
            agg_input_tokens += end_data.get("input_tokens") or 0
            agg_output_tokens += end_data.get("output_tokens") or 0
            agg_total_tokens += end_data.get("total_tokens") or 0
            agg_cost_usd += end_data.get("cost_usd") or 0.0

            # Check for tool calls
            tool_call_dicts = end_data.get("tool_calls")
            if not tool_call_dicts:
                # Final iteration — response already streamed. Emit message_end with aggregates.
                end_data["input_tokens"] = agg_input_tokens
                end_data["output_tokens"] = agg_output_tokens
                end_data["total_tokens"] = agg_total_tokens
                end_data["cost_usd"] = agg_cost_usd
                # Clean internal fields from end_data
                end_data.pop("content", None)
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
                yield StreamEvent(
                    event_type="tool_start",
                    data={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "arguments": tc.arguments,
                    },
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1

            # Execute tools (in parallel when multiple)
            results = self._execute_tool_calls(parsed_tool_calls, tool_by_name)

            # Emit tool_end for all results and append to history
            for tc, result_str in results:
                yield StreamEvent(
                    event_type="tool_end",
                    data={
                        "tool_name": tc.name,
                        "tool_call_id": tc.id,
                        "result": result_str,
                    },
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            req = req.model_copy(update={"messages": new_messages})

        # Max iterations reached: one final streaming call WITHOUT tools
        logger.warning(
            "Streaming tool loop exhausted %d iterations; stripping tools for final stream",
            max_iterations,
        )
        final_req = req.model_copy(update={"tool_schemas": None})
        for item in _do_stream_iteration(final_req, sequence):
            if isinstance(item, StreamEvent):
                yield item
            else:
                end_data, sequence = item
        # Add aggregate usage to final message_end
        agg_input_tokens += end_data.get("input_tokens") or 0
        agg_output_tokens += end_data.get("output_tokens") or 0
        agg_total_tokens += end_data.get("total_tokens") or 0
        agg_cost_usd += end_data.get("cost_usd") or 0.0
        end_data["input_tokens"] = agg_input_tokens
        end_data["output_tokens"] = agg_output_tokens
        end_data["total_tokens"] = agg_total_tokens
        end_data["cost_usd"] = agg_cost_usd
        end_data.pop("content", None)
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


__all__ = ["SimpleChatPipeline", "UsageMetadataCallbackHandler"]
