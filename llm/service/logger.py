"""
LLM call logging helpers.

These functions write to LLMCallLog without ever raising — a logging failure must
never surface to the caller.
"""

from __future__ import annotations

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
    """Write a SUCCESS log entry for a non-streaming call."""
    try:
        from llm.models import LLMCallLog

        usage = response.usage
        cost = None
        if usage and usage.cost_usd is not None:
            cost = Decimal(str(usage.cost_usd))

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            model=request.model or "",
            is_stream=False,
            prompt=_serialize_messages(request),
            raw_output=response.message.content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
        )
    except Exception:
        logger.exception("Failed to write LLM call log (non-streaming)")


def log_stream(
    request: "ChatRequest",
    events: "List[StreamEvent]",
    duration_ms: int,
) -> None:
    """Write a SUCCESS log entry after a streaming call completes."""
    try:
        from llm.models import LLMCallLog

        raw_output = "".join(
            e.data.get("text", "") for e in events if e.event_type == "token"
        )

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            model=request.model or "",
            is_stream=True,
            prompt=_serialize_messages(request),
            raw_output=raw_output,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
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
