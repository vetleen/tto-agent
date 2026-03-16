"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import json
import logging
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

    def __init__(self, max_tool_iterations: int = 10) -> None:
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

        # Create usage callback to aggregate across tool loop rounds
        usage_callback = UsageMetadataCallbackHandler()
        params = dict(req.params or {})
        params["_usage_callback"] = usage_callback
        req = req.model_copy(update={"params": params})

        model = create_chat_model(req.model)
        run_id = req.context.run_id if req.context else ""
        max_iter = self._get_max_iterations(request)
        yield from self._stream_with_tools(model, req, tools, run_id, max_iter)

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
        """Execute a batch of tool calls and return (tool_call, result_json) pairs."""
        results: List[Tuple[ToolCall, str]] = []
        for tc in tool_calls:
            tool = tool_by_name.get(tc.name)
            if not tool:
                result_str = json.dumps({"error": f"Unknown tool: {tc.name}"})
            else:
                try:
                    result = tool.invoke(tc.arguments)
                    # BaseTool._run returns str; ensure it's a string
                    result_str = result if isinstance(result, str) else json.dumps(result)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
            results.append((tc, result_str))
        return results

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

        # Max iterations reached: do one final generate.
        return chat_model.generate(req)

    def _stream_with_tools(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[ContextAwareTool],
        run_id: str,
        max_iterations: int,
    ) -> Iterator[StreamEvent]:
        tool_by_name = {t.name: t for t in tools}
        req = request
        sequence = 1

        for _ in range(max_iterations):
            response = chat_model.generate(req)
            msg = response.message
            if not msg.tool_calls:
                # Emit the already-fetched response as stream events
                # instead of calling the LLM a second time via stream().
                yield StreamEvent(
                    event_type="message_start",
                    data={"model": req.model},
                    sequence=sequence,
                    run_id=run_id,
                )
                sequence += 1
                if msg.content:
                    yield StreamEvent(
                        event_type="token",
                        data={"text": msg.content},
                        sequence=sequence,
                        run_id=run_id,
                    )
                    sequence += 1
                usage_data = {}
                if response.usage:
                    usage_data = response.usage.model_dump(exclude_none=True)
                yield StreamEvent(
                    event_type="message_end",
                    data=usage_data,
                    sequence=sequence,
                    run_id=run_id,
                )
                return

            new_messages = list(req.messages) + [msg]

            # Execute tools one-by-one, yielding tool_start BEFORE and
            # tool_end AFTER each execution so the client gets real-time
            # feedback.
            for tc in msg.tool_calls:
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

                tool = tool_by_name.get(tc.name)
                if not tool:
                    result_str = json.dumps({"error": f"Unknown tool: {tc.name}"})
                else:
                    try:
                        result = tool.invoke(tc.arguments)
                        result_str = result if isinstance(result, str) else json.dumps(result)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})

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

        # Max iterations reached: one final generate, emitted as stream events.
        response = chat_model.generate(req)
        msg = response.message
        yield StreamEvent(
            event_type="message_start",
            data={"model": req.model},
            sequence=sequence,
            run_id=run_id,
        )
        sequence += 1
        if msg.content:
            yield StreamEvent(
                event_type="token",
                data={"text": msg.content},
                sequence=sequence,
                run_id=run_id,
            )
            sequence += 1
        usage_data = {}
        if response.usage:
            usage_data = response.usage.model_dump(exclude_none=True)
        yield StreamEvent(
            event_type="message_end",
            data=usage_data,
            sequence=sequence,
            run_id=run_id,
        )


# Register so LLMService can resolve "simple_chat"
_registry = get_pipeline_registry()
_registry.register_pipeline(SimpleChatPipeline())


__all__ = ["SimpleChatPipeline", "UsageMetadataCallbackHandler"]
