"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import json
from typing import Iterator, List, Tuple

from llm.core.interfaces import ChatModel
from llm.core.registry import get_model_registry
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import get_pipeline_registry
from llm.types.messages import Message, ToolCall
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent
from llm.types.context import RunContext
from llm.tools import tools_to_langchain_schemas
from llm.tools.interfaces import Tool
from llm.tools.registry import get_tool_registry

# Ensure built-in tools (e.g. add_number) are registered when this pipeline is used.
import llm.tools.builtins  # noqa: F401


class SimpleChatPipeline(BasePipeline):
    """Single pipeline: LLM-driven tool calling via bind_tools(), else delegate to ChatModel."""

    id = "simple_chat"
    capabilities = {"streaming": True, "tools": True}

    def __init__(self, max_tool_iterations: int = 10) -> None:
        self.max_tool_iterations = max_tool_iterations

    def run(self, request: ChatRequest) -> ChatResponse:
        tool_names = request.tools or []
        if not tool_names:
            return self._run_no_tools(request)

        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")
        tools = self._resolve_tools(tool_names)
        schemas = tools_to_langchain_schemas(tools)
        req = request.model_copy(update={"tool_schemas": schemas})
        return self._run_tool_loop(get_model_registry().get_model(req.model), req, tools)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        tool_names = request.tools or []
        if not tool_names:
            model_name = request.model
            if not model_name:
                raise ValueError("request.model must be set by the service before calling pipeline")
            yield from get_model_registry().get_model(model_name).stream(request)
            return

        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")
        tools = self._resolve_tools(tool_names)
        schemas = tools_to_langchain_schemas(tools)
        req = request.model_copy(update={"tool_schemas": schemas})
        model = get_model_registry().get_model(req.model)
        run_id = req.context.run_id if req.context else ""
        yield from self._stream_with_tools(model, req, tools, run_id, req.context)

    def _resolve_tools(self, tool_names: List[str]) -> List[Tool]:
        registry = get_tool_registry()
        tools = []
        for name in tool_names:
            t = registry.get_tool(name)
            if t is None:
                raise ValueError(f"Unknown tool name: {name!r}")
            tools.append(t)
        return tools

    def _run_no_tools(self, request: ChatRequest) -> ChatResponse:
        model_name = request.model
        if not model_name:
            raise ValueError("request.model must be set by the service before calling pipeline")
        chat_model = get_model_registry().get_model(model_name)
        return chat_model.generate(request)

    @staticmethod
    def _execute_tool_calls(
        tool_calls: List[ToolCall],
        tool_by_name: dict[str, Tool],
        context: RunContext | None,
    ) -> List[Tuple[ToolCall, str]]:
        """Execute a batch of tool calls and return (tool_call, result_json) pairs."""
        results: List[Tuple[ToolCall, str]] = []
        for tc in tool_calls:
            tool = tool_by_name.get(tc.name)
            if not tool:
                result_str = json.dumps({"error": f"Unknown tool: {tc.name}"})
            else:
                try:
                    result = tool.run(tc.arguments, context or RunContext.create())
                    result_str = json.dumps(result)
                except Exception as e:
                    result_str = json.dumps({"error": str(e)})
            results.append((tc, result_str))
        return results

    def _run_tool_loop(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[Tool],
    ) -> ChatResponse:
        tool_by_name = {t.name: t for t in tools}
        req = request

        for _ in range(self.max_tool_iterations):
            response = chat_model.generate(req)
            msg = response.message
            if not msg.tool_calls:
                return response

            new_messages = list(req.messages) + [msg]
            results = self._execute_tool_calls(msg.tool_calls, tool_by_name, req.context)
            for tc, result_str in results:
                new_messages.append(
                    Message(role="tool", content=result_str, tool_call_id=tc.id)
                )

            req = req.model_copy(update={"messages": new_messages})

        # Max iterations reached: strip tools and do one final generate
        req = req.model_copy(update={"tool_schemas": None})
        return chat_model.generate(req)

    def _stream_with_tools(
        self,
        chat_model: ChatModel,
        request: ChatRequest,
        tools: List[Tool],
        run_id: str,
        context,
    ) -> Iterator[StreamEvent]:
        tool_by_name = {t.name: t for t in tools}
        req = request
        sequence = 1

        for _ in range(self.max_tool_iterations):
            response = chat_model.generate(req)
            msg = response.message
            if not msg.tool_calls:
                # Final round: stream the real response token-by-token.
                # Strip tool_schemas so the provider doesn't bind tools.
                final_req = req.model_copy(update={"tool_schemas": None})
                yield from chat_model.stream(final_req)
                return

            new_messages = list(req.messages) + [msg]
            results = self._execute_tool_calls(msg.tool_calls, tool_by_name, context)

            for tc, result_str in results:
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

        # Max iterations: strip tools and stream the final response.
        req = req.model_copy(update={"tool_schemas": None})
        yield from chat_model.stream(req)


# Register so LLMService can resolve "simple_chat"
_registry = get_pipeline_registry()
_registry.register_pipeline(SimpleChatPipeline())


__all__ = ["SimpleChatPipeline"]
