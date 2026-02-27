"""Simple chat pipeline: optional tool shortcut, then model generate/stream."""

from __future__ import annotations

import re
from typing import Iterator

from llm.core.interfaces import ChatModel
from llm.core.registry import get_model_registry
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import get_pipeline_registry
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent
from llm.tools.registry import get_tool_registry

# Ensure built-in tools (e.g. add_number) are registered when this pipeline is used.
import llm.tools.builtins  # noqa: F401


# Pattern for the testing tool shortcut: "tool:add_number a=<num> b=<num>"
_ADD_NUMBER_PATTERN = re.compile(
    r"^tool:add_number\s+a=([0-9]+(?:\.[0-9]+)?)\s+b=([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)


class SimpleChatPipeline(BasePipeline):
    """Single pipeline: optional add_number tool shortcut, else delegate to ChatModel."""

    id = "simple_chat"
    capabilities = {"streaming": True, "tools": True}

    def run(self, request: ChatRequest) -> ChatResponse:
        last_user_content = self._last_user_content(request)
        if last_user_content is not None:
            tool_result = self._try_add_number_tool(last_user_content, request)
            if tool_result is not None:
                return tool_result

        model_name = request.model
        if not model_name:
            raise ValueError("request.model must be set by the service before calling pipeline")
        chat_model = get_model_registry().get_model(model_name)
        return chat_model.generate(request)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        last_user_content = self._last_user_content(request)
        if last_user_content is not None:
            tool_response = self._try_add_number_tool(last_user_content, request)
            if tool_response is not None:
                run_id = request.context.run_id if request.context else ""
                yield StreamEvent(
                    event_type="message_start",
                    data={"model": ""},
                    sequence=1,
                    run_id=run_id,
                )
                yield StreamEvent(
                    event_type="token",
                    data={"text": tool_response.message.content},
                    sequence=2,
                    run_id=run_id,
                )
                yield StreamEvent(
                    event_type="message_end",
                    data={"model": ""},
                    sequence=3,
                    run_id=run_id,
                )
                return

        model_name = request.model
        if not model_name:
            raise ValueError("request.model must be set by the service before calling pipeline")
        chat_model = get_model_registry().get_model(model_name)
        yield from chat_model.stream(request)

    @staticmethod
    def _last_user_content(request: ChatRequest) -> str | None:
        for m in reversed(request.messages):
            if m.role == "user":
                return (m.content or "").strip()
        return None

    @staticmethod
    def _try_add_number_tool(last_content: str, request: ChatRequest) -> ChatResponse | None:
        m = _ADD_NUMBER_PATTERN.match(last_content.strip())
        if not m:
            return None
        a_str, b_str = m.group(1), m.group(2)
        try:
            a_num = float(a_str)
            b_num = float(b_str)
        except ValueError:
            return None
        tool = get_tool_registry().get_tool("add_number")
        if not tool:
            return None
        context = request.context
        if context is None:
            from llm.types.context import RunContext
            context = RunContext.create()
        result = tool.run({"a": a_num, "b": b_num}, context)
        result_val = result.get("result")
        return ChatResponse(
            message=Message(role="assistant", content=f"Result: {result_val}"),
            model="",
            usage=None,
            metadata={},
        )


# Register so LLMService can resolve "simple_chat"
_registry = get_pipeline_registry()
_registry.register_pipeline(SimpleChatPipeline())


__all__ = ["SimpleChatPipeline"]
