"""
LLM call logging helpers.

These functions write to LLMCallLog without ever raising — a logging failure must
never surface to the caller.

Payloads are stored SLIM by design (memory + DB footprint): ``prompt`` keeps the
message-list shape but caps text content (500 chars/message) and tool-call
arguments (1KB); ``raw_output`` is a compact JSON summary
``{final_text_preview, tool_calls: [{tool_name, args_size, result_size,
result_preview}], counts}`` — never the full transcript. Set
``LLM_LOG_FULL_PAYLOADS=true`` (env) for larger, still-capped payloads when
debugging locally. See ``llm/service/stream_log.py``.

tools stores a simplified list of tool schemas from the request:
- [{"name": "...", "description": "..."}, ...] when tools are bound
- None when no tools are present
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import TYPE_CHECKING

from llm.service.stream_log import (
    LOG_FULL_PAYLOAD_CAP_CHARS,
    LOG_TEXT_PREVIEW_CHARS,
    LOG_TOOL_ARGS_CHARS,
    cap_text,
    full_payloads_enabled,
    slim_response_raw_output,
)

if TYPE_CHECKING:
    from llm.service.stream_log import StreamLogAccumulator
    from llm.types.requests import ChatRequest
    from llm.types.responses import ChatResponse

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


def _cap_content(content):
    """Cap logged message content: whole string, or each text block in a list.

    Base64/data-URI blocks were already replaced by ``_truncate_base64_in_content``;
    this bounds the remaining text so a 24k-char extracted-text block or a giant
    system prompt never lands in the log row.
    """
    cap = LOG_FULL_PAYLOAD_CAP_CHARS if full_payloads_enabled() else LOG_TEXT_PREVIEW_CHARS
    if isinstance(content, str):
        return cap_text(content, cap)
    if isinstance(content, list):
        capped = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                capped.append({**block, "text": cap_text(block["text"], cap)})
            else:
                capped.append(block)
        return capped
    return content


def _cap_tool_arguments(arguments):
    """Bound logged tool-call arguments; keep the dict when it's small."""
    cap = LOG_FULL_PAYLOAD_CAP_CHARS if full_payloads_enabled() else LOG_TOOL_ARGS_CHARS
    try:
        serialized = json.dumps(arguments, default=str)
    except Exception:
        return arguments
    if len(serialized) <= cap:
        return arguments
    return f"[tool args truncated: {len(serialized):,} chars] {serialized[:cap]}"


def _serialize_messages(request: "ChatRequest") -> list:
    """Convert request messages to a plain list of dicts (content capped)."""
    result = []
    for m in request.messages:
        d = {"role": m.role, "content": _cap_content(_truncate_base64_in_content(m.content))}
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": _cap_tool_arguments(tc.arguments)}
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


def _user_fk_id(user_id: str | None) -> int | None:
    """Coerce a context user_id string to the FK value, without a DB lookup.

    Passing the id straight to ``user_id=`` on create avoids a per-log ``User``
    SELECT on every call. The FK is nullable + ``on_delete=SET_NULL``; the rare
    case where the id references an already-deleted user raises IntegrityError on
    insert, which the caller's best-effort try/except swallows — losing one cost
    log for a mid-call user deletion is acceptable versus a SELECT on every write.
    """
    try:
        return int(user_id) if user_id else None
    except (TypeError, ValueError):
        return None


def log_call(request: "ChatRequest", response: "ChatResponse", duration_ms: int) -> None:
    """Write a SUCCESS log entry for a non-streaming call. raw_output = slim summary JSON."""
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
                usage.cache_write_tokens,
            )
            if computed is not None:
                cost = computed

        context = request.context
        metadata = response.metadata or {}
        LLMCallLog.objects.create(
            user_id=_user_fk_id(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=request.model or "",
            is_stream=False,
            prompt=_serialize_messages(request),
            tools=_serialize_tool_schemas(request.tool_schemas, request.tools),
            raw_output=slim_response_raw_output(response),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            cached_tokens=usage.cached_tokens if usage else None,
            cache_write_tokens=usage.cache_write_tokens if usage else None,
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


def log_stream(
    request: "ChatRequest",
    acc: "StreamLogAccumulator",
    duration_ms: int,
) -> None:
    """Write a SUCCESS or ERROR log entry after a streaming call completes.

    Takes a ``StreamLogAccumulator`` that folded the events as they streamed —
    the full event list is never retained (see ``llm/service/stream_log.py``).

    Providers catch exceptions in ``BaseLangChainChatModel.stream`` and yield
    an ``error`` ``StreamEvent`` instead of re-raising, so ``LLMService.stream``
    never sees the underlying exception and can't call ``log_error`` itself.
    Detect that case here: whenever an ``error`` event is present, write an
    ERROR row so failed calls stay visible in ``LLMCallLog`` — even if a
    ``message_end`` also slipped through (defensive; its usage fields are
    kept on the ERROR row).
    """
    try:
        from llm.models import LLMCallLog

        raw_output = acc.build_raw_output()

        # Usage comes from the message_end data (populated by provider)
        end_data = acc.end_data
        input_tokens = end_data.get("input_tokens")
        output_tokens = end_data.get("output_tokens")
        total_tokens = end_data.get("total_tokens")
        cached_tokens = end_data.get("cached_tokens")
        cache_write_tokens = end_data.get("cache_write_tokens")
        reasoning_tokens = end_data.get("reasoning_tokens")

        # Interrupted stream (user Stop, repetition-guard abort, or disconnect):
        # the provider never sent a terminal message_end, so usage is absent.
        # Estimate output tokens from the text we did stream so cost stays
        # visible instead of NULL, and flag explicit cancellations distinctly.
        # Keys off has_end_event, not end_data truthiness — an empty-data
        # message_end is still terminal.
        interrupted = not acc.has_end_event and acc.error_data is None
        if interrupted and output_tokens is None:
            from core.tokens import count_tokens

            streamed_text = acc.streamed_text
            if streamed_text:
                output_tokens = count_tokens(streamed_text)
        params = request.params or {}
        cancel_event = params.get("_cancel_event")
        cancel_check = params.get("_cancel_check")
        # Honor BOTH cancel signals: the consumer path sets a threading.Event
        # (_cancel_event); sub-agent runs cancel via a _cancel_check callable
        # (DB poll — True once the run is marked FAILED). Without the second
        # check, an aborted sub-agent stream was logged SUCCESS. cancel_check()
        # only runs when interrupted (no message_end/error), so it's bounded.
        was_cancelled = bool(
            interrupted and (
                (cancel_event is not None and cancel_event.is_set())
                or (cancel_check is not None and cancel_check())
            )
        )

        cost_raw = end_data.get("cost_usd")
        cost = Decimal(str(cost_raw)) if cost_raw is not None else None
        if cost is None and (input_tokens or output_tokens):
            from llm.service.pricing import calculate_cost

            computed = calculate_cost(
                request.model or "",
                input_tokens,
                output_tokens,
                cached_tokens,
                cache_write_tokens,
            )
            if computed is not None:
                cost = computed

        # Response metadata from message_end
        resp_metadata = end_data.get("response_metadata")
        stop_reason = end_data.get("stop_reason", "")
        provider_model_id = end_data.get("provider_model_id", "")

        context = request.context
        if acc.error_data is not None:
            err_data = acc.error_data
            LLMCallLog.objects.create(
                user_id=_user_fk_id(context.user_id if context else None),
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
                cache_write_tokens=cache_write_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=cost,
                duration_ms=duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=(err_data.get("error_code") or "stream_error")[:255],
                error_message=(err_data.get("details") or err_data.get("message") or "")[:2000],
                response_metadata=resp_metadata,
                stop_reason=stop_reason,
                provider_model_id=provider_model_id,
            )
            return

        LLMCallLog.objects.create(
            user_id=_user_fk_id(context.user_id if context else None),
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
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            status=(
                LLMCallLog.Status.CANCELLED if was_cancelled
                else LLMCallLog.Status.SUCCESS
            ),
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
            user_id=_user_fk_id(context.user_id if context else None),
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
            user_id=_user_fk_id(context.user_id if context else None),
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
            user_id=_user_fk_id(context.user_id if context else None),
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


def log_image_generation(
    model: str,
    context: "RunContext | None",
    *,
    prompt: str,
    cost_usd: "Decimal | None",
    duration_ms: int,
    width: "int | None" = None,
    height: "int | None" = None,
    n_images: int = 1,
    is_edit: bool = False,
    input_tokens: "int | None" = None,
    output_tokens: "int | None" = None,
    total_tokens: "int | None" = None,
) -> None:
    """Write a SUCCESS log entry for an image-generation call. Never raises.

    Image generation runs outside the chat pipeline (like transcription), so it
    writes its own LLMCallLog row for unified cost tracking. ``cost_usd`` may be
    None when the model has no pricing configured; the row is still written so
    usage is visible. Token counts are passed through when the provider returns
    a usage object (e.g. OpenAI gpt-image-*); flat-priced Gemini models leave
    them None.
    """
    try:
        from llm.models import LLMCallLog

        if TYPE_CHECKING:
            from llm.types.context import RunContext

        action = "image edit" if is_edit else "image generation"
        size = f"{width}x{height}" if width and height else "?"
        LLMCallLog.objects.create(
            user_id=_user_fk_id(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=model,
            is_stream=False,
            prompt=[{"role": "user", "content": f"[{action}: {prompt[:500]}]"}],
            raw_output=f"[image: {size}, {n_images} image(s)]",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            status=LLMCallLog.Status.SUCCESS,
            response_metadata={
                "width": width,
                "height": height,
                "n_images": n_images,
                "is_edit": is_edit,
            },
        )
    except Exception:
        logger.exception("Failed to write LLM call log (image generation)")


def log_image_generation_error(
    model: str,
    context: "RunContext | None",
    exc: BaseException,
    duration_ms: int,
    *,
    prompt: str = "",
    is_edit: bool = False,
) -> None:
    """Write an ERROR log entry for an image-generation call. Never raises."""
    try:
        from llm.models import LLMCallLog

        action = "image edit" if is_edit else "image generation"
        LLMCallLog.objects.create(
            user_id=_user_fk_id(context.user_id if context else None),
            run_id=context.run_id if context else "",
            trace_id=(context.trace_id if context else "") or "",
            conversation_id=(context.conversation_id if context else "") or "",
            model=model,
            is_stream=False,
            prompt=[{"role": "user", "content": f"[{action}: {prompt[:500]}]"}],
            raw_output="",
            duration_ms=duration_ms,
            status=LLMCallLog.Status.ERROR,
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
        )
    except Exception:
        logger.exception("Failed to write LLM error log (image generation)")


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
            user_id=_user_fk_id(context.user_id if context else None),
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
