"""
LLM call logging helpers.

These functions write to LLMCallLog without ever raising — a logging failure must
never surface to the caller.

raw_output stores the entire response as a single coherent JSON blob:
- Non-streaming: full ChatResponse (message, model, usage, metadata).
- Streaming: one assembled response built from all events after the stream
  finishes (message content from token events, tool_calls from tool_start/tool_end).
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from llm.types.requests import ChatRequest
    from llm.types.responses import ChatResponse
    from llm.types.streaming import StreamEvent

logger = logging.getLogger(__name__)


def _serialize_messages(request: "ChatRequest") -> list:
    """Convert request messages to a plain list of dicts."""
    return [
        {"role": m.role, "content": m.content}
        for m in request.messages
    ]


def _resolve_user(user_id: str | None):
    """Look up the User FK from a string user_id. Returns None on any failure."""
    if not user_id:
        return None
    try:
        from django.contrib.auth import get_user_model
        User = get_user_model()
        return User.objects.filter(pk=user_id).first()
    except Exception:
        return None


def log_call(request: "ChatRequest", response: "ChatResponse", duration_ms: int) -> None:
    """Write a SUCCESS log entry for a non-streaming call. raw_output = full response JSON."""
    try:
        from llm.models import LLMCallLog

        from llm.service.pricing import calculate_cost

        usage = response.usage
        cost = None
        if usage and usage.cost_usd is not None:
            cost = Decimal(str(usage.cost_usd))
        elif usage and cost is None:
            # Defense-in-depth: calculate cost if provider didn't supply it
            computed = calculate_cost(
                request.model or "",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_tokens,
            )
            if computed is not None:
                cost = computed

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            model=request.model or "",
            is_stream=False,
            prompt=_serialize_messages(request),
            raw_prompt=response.metadata.get("raw_prompt"),
            raw_output=response.model_dump_json(),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
        )
    except Exception:
        logger.exception("Failed to write LLM call log (non-streaming)")


def _assemble_stream_response(events: "List[StreamEvent]") -> str:
    """Build a single coherent response JSON from stream events (after stream finished)."""
    content = "".join(
        e.data.get("text", "") for e in events if e.event_type == "token"
    )
    # Pair tool_start with tool_end by tool_call_id
    tool_by_id: dict = {}
    for e in events:
        if e.event_type == "tool_start":
            tid = e.data.get("tool_call_id") or ""
            tool_by_id[tid] = {
                "tool_call_id": tid,
                "tool_name": e.data.get("tool_name", ""),
                "arguments": e.data.get("arguments", {}),
                "result": None,
            }
        elif e.event_type == "tool_end":
            tid = e.data.get("tool_call_id") or ""
            if tid in tool_by_id:
                tool_by_id[tid]["result"] = e.data.get("result")
            else:
                tool_by_id[tid] = {
                    "tool_call_id": tid,
                    "tool_name": e.data.get("tool_name", ""),
                    "arguments": {},
                    "result": e.data.get("result"),
                }
    tool_calls = list(tool_by_id.values())
    payload = {
        "message": {"role": "assistant", "content": content},
        "tool_calls": tool_calls,
    }
    return json.dumps(payload)


def log_stream(
    request: "ChatRequest",
    events: "List[StreamEvent]",
    duration_ms: int,
) -> None:
    """Write a SUCCESS log entry after a streaming call completes. raw_output = assembled response JSON."""
    try:
        from llm.models import LLMCallLog

        raw_output = _assemble_stream_response(events)
        raw_prompt = next(
            (e.data.get("raw_prompt") for e in events if e.event_type == "raw_prompt"),
            None,
        )

        # Extract usage from the message_end event (populated by provider)
        end_event = next(
            (e for e in events if e.event_type == "message_end"),
            None,
        )
        end_data = end_event.data if end_event else {}
        input_tokens = end_data.get("input_tokens")
        output_tokens = end_data.get("output_tokens")
        total_tokens = end_data.get("total_tokens")
        cost_raw = end_data.get("cost_usd")
        cost = Decimal(str(cost_raw)) if cost_raw is not None else None

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            model=request.model or "",
            is_stream=True,
            prompt=_serialize_messages(request),
            raw_prompt=raw_prompt,
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
        )
    except Exception:
        logger.exception("Failed to write LLM call log (streaming)")


def log_error(
    request: "ChatRequest",
    exc: BaseException,
    duration_ms: int,
    *,
    is_stream: bool = False,
) -> None:
    """Write an ERROR log entry."""
    try:
        from llm.models import LLMCallLog

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            model=request.model or "",
            is_stream=is_stream,
            prompt=_serialize_messages(request),
            raw_output="",
            duration_ms=duration_ms,
            status=LLMCallLog.Status.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception:
        logger.exception("Failed to write LLM error log")


__all__ = ["log_call", "log_stream", "log_error"]
