"""
LLM call logging helpers.

These functions write to LLMCallLog without ever raising — a logging failure must
never surface to the caller.

raw_output stores the entire response as a single coherent JSON blob:
- Non-streaming: full ChatResponse (message, model, usage, metadata).
- Streaming: one assembled response built from all events after the stream
  finishes (message content from token events, tool_calls from tool_start/tool_end).

tools stores a simplified list of tool schemas from the request:
- [{"name": "...", "description": "..."}, ...] when tools are bound
- None when no tools are present
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


def _truncate_base64_in_content(content):
    """Replace large base64 strings with placeholders in logged content."""
    if not isinstance(content, list):
        return content
    truncated = []
    for block in content:
        if isinstance(block, dict) and "base64" in block:
            truncated.append({**block, "base64": f"[{len(block['base64'])} chars]"})
        elif isinstance(block, dict) and block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:") and len(url) > 200:
                truncated.append({**block, "image_url": {"url": f"[data URI, {len(url)} chars]"}})
            else:
                truncated.append(block)
        elif isinstance(block, dict) and block.get("type") == "document":
            # Anthropic PDF block — truncate source.data
            src = block.get("source", {})
            data = src.get("data", "")
            if data and len(data) > 200:
                truncated.append({**block, "source": {**src, "data": f"[{len(data)} chars]"}})
            else:
                truncated.append(block)
        elif isinstance(block, dict) and block.get("type") == "file":
            # OpenAI PDF block — truncate file.file_data
            f = block.get("file", {})
            fd = f.get("file_data", "")
            if fd and len(fd) > 200:
                truncated.append({**block, "file": {**f, "file_data": f"[{len(fd)} chars]"}})
            else:
                truncated.append(block)
        else:
            truncated.append(block)
    return truncated


def _serialize_messages(request: "ChatRequest") -> list:
    """Convert request messages to a plain list of dicts."""
    result = []
    for m in request.messages:
        d = {"role": m.role, "content": _truncate_base64_in_content(m.content)}
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in m.tool_calls
            ]
        result.append(d)
    return result


def _serialize_tool_schemas(tool_schemas: list | None, tool_names: list | None = None) -> list | None:
    """Serialize tool schemas from the request to a compact list of dicts.

    Falls back to tool_names (string list) when tool_schemas hasn't been
    resolved — e.g. when the service logs the original request object while
    the pipeline only populated tool_schemas on a local copy.
    """
    if not tool_schemas:
        if not tool_names:
            return None
        return [{"name": n} for n in tool_names]
    result = []
    for tool in tool_schemas:
        try:
            result.append({
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
            })
        except Exception:
            result.append({"name": str(tool)})
    return result


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
        metadata = response.metadata or {}
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=request.model or "",
            is_stream=False,
            prompt=_serialize_messages(request),
            tools=_serialize_tool_schemas(request.tool_schemas, request.tools),
            raw_output=response.model_dump_json(),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cached_tokens=usage.cached_tokens if usage else None,
            reasoning_tokens=usage.reasoning_tokens if usage else None,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
            response_metadata=metadata.get("response_metadata"),
            stop_reason=metadata.get("stop_reason", ""),
            provider_model_id=metadata.get("provider_model_id", ""),
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
    """Write a SUCCESS or ERROR log entry after a streaming call completes.

    Providers catch exceptions in ``BaseLangChainChatModel.stream`` and yield
    an ``error`` ``StreamEvent`` instead of re-raising, so ``LLMService.stream``
    never sees the underlying exception and can't call ``log_error`` itself.
    Detect that case here: when the stream ended without a ``message_end`` and
    an ``error`` event is present, write an ERROR row so failed calls stay
    visible in ``LLMCallLog``.
    """
    try:
        from llm.models import LLMCallLog

        raw_output = _assemble_stream_response(events)

        # Extract usage from the message_end event (populated by provider)
        end_event = next(
            (e for e in events if e.event_type == "message_end"),
            None,
        )
        error_event = next(
            (e for e in events if e.event_type == "error"),
            None,
        )

        if end_event is None and error_event is not None:
            err_data = error_event.data or {}
            context = request.context
            LLMCallLog.objects.create(
                user=_resolve_user(context.user_id if context else None),
                run_id=context.run_id if context else "",
                trace_id=(context.trace_id if context else "") or "",
                conversation_id=(context.conversation_id if context else "") or "",
                model=request.model or "",
                is_stream=True,
                prompt=_serialize_messages(request),
                tools=_serialize_tool_schemas(request.tool_schemas, request.tools),
                raw_output=raw_output,
                duration_ms=duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=(err_data.get("error_code") or "stream_error")[:255],
                error_message=(err_data.get("details") or err_data.get("message") or "")[:2000],
            )
            return

        end_data = end_event.data if end_event else {}
        input_tokens = end_data.get("input_tokens")
        output_tokens = end_data.get("output_tokens")
        total_tokens = end_data.get("total_tokens")
        cached_tokens = end_data.get("cached_tokens")
        reasoning_tokens = end_data.get("reasoning_tokens")
        cost_raw = end_data.get("cost_usd")
        cost = Decimal(str(cost_raw)) if cost_raw is not None else None
        if cost is None and (input_tokens or output_tokens):
            from llm.service.pricing import calculate_cost

            computed = calculate_cost(
                request.model or "",
                input_tokens,
                output_tokens,
                cached_tokens,
            )
            if computed is not None:
                cost = computed

        # Response metadata from message_end
        resp_metadata = end_data.get("response_metadata")
        stop_reason = end_data.get("stop_reason", "")
        provider_model_id = end_data.get("provider_model_id", "")

        context = request.context
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=request.model or "",
            is_stream=True,
            prompt=_serialize_messages(request),
            tools=_serialize_tool_schemas(request.tool_schemas, request.tools),
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
            response_metadata=resp_metadata,
            stop_reason=stop_reason,
            provider_model_id=provider_model_id,
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
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=request.model or "",
            is_stream=is_stream,
            prompt=_serialize_messages(request),
            tools=_serialize_tool_schemas(request.tool_schemas, request.tools),
            raw_output="",
            duration_ms=duration_ms,
            status=LLMCallLog.Status.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    except Exception:
        logger.exception("Failed to write LLM error log")


def log_transcription(
    model: str,
    context: "RunContext | None",
    audio_duration_seconds: float,
    transcript_len: int,
    cost_usd: "Decimal | None",
    duration_ms: int,
    file_size: int,
    segments: int = 1,
    input_tokens: "int | None" = None,
    output_tokens: "int | None" = None,
    total_tokens: "int | None" = None,
    audio_tokens: "int | None" = None,
) -> None:
    """Write a SUCCESS log entry for a transcription call. Never raises.

    When the OpenAI transcription API returns ``response.usage`` (which it
    does for ``gpt-4o-transcribe`` / ``gpt-4o-mini-transcribe``), callers
    should pass ``input_tokens`` / ``output_tokens`` / ``total_tokens`` so
    transcription rows populate the same ``LLMCallLog`` columns chat rows do
    and token-based analytics stays consistent across call types.
    ``audio_tokens`` is stashed inside ``response_metadata`` as a nice-to-have
    observability signal but is not a first-class column.
    """
    try:
        from llm.models import LLMCallLog

        if TYPE_CHECKING:
            from llm.types.context import RunContext

        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=model,
            is_stream=False,
            prompt=[{
                "role": "user",
                "content": (
                    f"[audio transcription: {file_size:,} bytes, "
                    f"{audio_duration_seconds:.1f}s, {segments} segment(s)]"
                ),
            }],
            raw_output=f"[transcript: {transcript_len:,} chars]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
            response_metadata={
                "audio_duration_seconds": audio_duration_seconds,
                "file_size": file_size,
                "segments": segments,
                "transcript_len": transcript_len,
                "audio_tokens": audio_tokens,
            },
        )
    except Exception:
        logger.exception("Failed to write LLM call log (transcription)")


def log_transcription_error(
    model: str,
    context: "RunContext | None",
    exc: BaseException,
    duration_ms: int,
    file_size: int = 0,
) -> None:
    """Write an ERROR log entry for a transcription call. Never raises."""
    try:
        from llm.models import LLMCallLog

        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=model,
            is_stream=False,
            prompt=[{
                "role": "user",
                "content": f"[audio transcription: {file_size:,} bytes]",
            }],
            raw_output="",
            duration_ms=duration_ms,
            status=LLMCallLog.Status.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
    except Exception:
        logger.exception("Failed to write LLM error log (transcription)")


def log_transcription_streaming(
    model: str,
    context: "RunContext | None",
    *,
    kind: str,                # "realtime_utterance" or "realtime_session"
    session_id: str | None = None,
    item_id: str | None = None,
    audio_duration_seconds: float = 0.0,
    transcript_len: int = 0,
    cost_usd: "Decimal | None" = None,
    duration_ms: int = 0,
    input_tokens: "int | None" = None,
    output_tokens: "int | None" = None,
    total_tokens: "int | None" = None,
    audio_tokens: "int | None" = None,
    extra_metadata: dict | None = None,
) -> None:
    """Write a SUCCESS log entry for a realtime transcription event. Never raises.

    Used for both per-utterance events (one row per completed utterance) and
    session summaries (one row on session close with aggregate counters). The
    ``kind`` column in ``response_metadata`` distinguishes them so analytics
    can GROUP BY mode without breaking existing transcription cost dashboards
    that read ``LLMCallLog.cost_usd`` directly.
    """
    try:
        from llm.models import LLMCallLog

        metadata = {
            "kind": kind,
            "session_id": session_id or "",
            "item_id": item_id or "",
            "audio_duration_seconds": audio_duration_seconds,
            "transcript_len": transcript_len,
            "audio_tokens": audio_tokens,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        label = "realtime utterance" if kind == "realtime_utterance" else "realtime session"
        LLMCallLog.objects.create(
            user=_resolve_user(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=model,
            is_stream=True,
            prompt=[{
                "role": "user",
                "content": (
                    f"[{label}: {audio_duration_seconds:.1f}s audio, "
                    f"transcript_len={transcript_len}]"
                ),
            }],
            raw_output=f"[transcript: {transcript_len:,} chars]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
            response_metadata=metadata,
        )
    except Exception:
        logger.exception("Failed to write LLM call log (realtime transcription)")


__all__ = [
    "log_call",
    "log_stream",
    "log_error",
    "log_transcription",
    "log_transcription_error",
    "log_transcription_streaming",
]
