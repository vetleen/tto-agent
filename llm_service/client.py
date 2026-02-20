"""
Entry points: completion(**kwargs), acompletion(**kwargs), get_client().
Applies policy (timeout, retry, guardrails), logs to LLMCallLog, proxies to BaseLLMClient.
"""
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, AsyncIterator, Iterator

from django.db import connection

from llm_service.base import BaseLLMClient
from llm_service.conf import (
    get_default_model,
    get_log_write_timeout,
    get_max_retries,
    get_post_call_hooks,
    get_pre_call_hooks,
    is_model_allowed,
)
from llm_service.litellm_client import LiteLLMClient
from llm_service.models import LLMCallLog
from llm_service.pricing import get_fallback_cost_usd
from llm_service.request_result import LLMRequest, LLMResult

logger = logging.getLogger(__name__)

# Module-level client instance (lazy)
_client: BaseLLMClient | None = None


def get_client() -> BaseLLMClient:
    """Return the configured LLM client (LiteLLM by default)."""
    global _client
    if _client is None:
        _client = LiteLLMClient()
    return _client


def _kwargs_to_request(**kwargs: Any) -> LLMRequest:
    """Build LLMRequest from completion(**kwargs)."""
    model = kwargs.pop("model", None) or get_default_model()
    messages = kwargs.get("messages", [])
    stream = kwargs.get("stream", False)
    metadata = kwargs.pop("metadata", None) or {}
    user = kwargs.pop("user", None)
    request_id = kwargs.pop("request_id", None)
    if request_id is not None and "request_id" not in metadata:
        metadata = {**metadata, "request_id": str(request_id)}
    return LLMRequest(
        model=model,
        messages=messages,
        stream=stream,
        metadata=metadata,
        raw_kwargs=kwargs,
        user=user,
    )


def _response_to_result(response: Any, model: str) -> LLMResult:
    """Build LLMResult from LiteLLM completion response."""
    usage = {}
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }
    cost = None
    try:
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
    except Exception:
        pass
    if cost is None and usage:
        fallback = get_fallback_cost_usd(
            model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        if fallback is not None:
            cost = float(fallback)
    text = None
    if getattr(response, "choices", None) and len(response.choices) > 0:
        c = response.choices[0]
        if getattr(c, "message", None) and getattr(c.message, "content", None):
            text = c.message.content
    return LLMResult(
        text=text,
        usage=usage or None,
        cost=cost,
        raw_response=response,
        provider_response_id=getattr(response, "id", None),
        response_model=getattr(response, "model", None),
    )


def _truncate(s: str, max_len: int = 2000) -> str:
    if not s or len(s) <= max_len:
        return s or ""
    return s[:max_len] + "..."


def _user_message_preview(messages: list) -> str:
    """Last user message content, truncated to 300 chars for list display."""
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return _truncate(content, 300)
            if content is not None:
                return _truncate(str(content), 300)
            return ""
    return ""


def _object_to_json_serializable(obj: Any) -> Any:
    """Convert a LiteLLM-style response object to JSON-serializable dict/list."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_object_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _object_to_json_serializable(v) for k, v in obj.items()}
    # Fallback: common response attributes
    out = {}
    for key in ("id", "object", "created", "model", "system_fingerprint", "usage", "choices", "_hidden_params"):
        if hasattr(obj, key):
            try:
                v = getattr(obj, key)
                out[key] = _object_to_json_serializable(v)
            except Exception:
                out[key] = str(v) if v is not None else None
    return out if out else str(obj)


def _serialize_raw_response_payload(raw_response: Any = None, raw_chunks: list[Any] | None = None) -> str:
    """Serialize raw LLM response for storage. Streaming: metadata once + aggregated deltas (no repeated chunks)."""
    if raw_chunks is not None:
        return _stream_chunks_to_condensed_payload(raw_chunks)
    if raw_response is not None:
        payload = _object_to_json_serializable(raw_response)
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            return str(payload)[:100000]
    return ""


def _tiktoken_encoding_name_for_model(model_name: str | None) -> str:
    """Return tiktoken encoding name. OpenAI models use their encoding; others use cl100k_base."""
    if not model_name:
        return "cl100k_base"
    m = model_name.lower()
    if "openai" in m or m.startswith("gpt-"):
        if "gpt-4o" in m or "4o-mini" in m or m.startswith("gpt-5"):
            return "o200k_base"
        if "gpt-4" in m or "gpt-3.5" in m:
            return "cl100k_base"
    return "cl100k_base"


def _count_tokens_tiktoken(text: str, model_name: str | None) -> int:
    """Count tokens with tiktoken. Returns 0 if tiktoken unavailable or on error."""
    if not text:
        return 0
    try:
        import tiktoken
    except Exception:
        return 0
    try:
        # For OpenAI-style model names (e.g. gpt-4o) use encoding_for_model; else use encoding name
        if model_name and ("gpt-" in model_name.lower() or "openai" in model_name.lower()):
            try:
                enc = tiktoken.encoding_for_model(model_name.split("/")[-1] if "/" in model_name else model_name)
            except Exception:
                enc = tiktoken.get_encoding(_tiktoken_encoding_name_for_model(model_name))
        else:
            enc = tiktoken.get_encoding(_tiktoken_encoding_name_for_model(model_name))
        return len(enc.encode(text))
    except Exception:
        return 0


def _messages_to_text_for_counting(messages: list) -> str:
    """Serialize messages to a single string for input token counting."""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", str(p)) for p in content if isinstance(p, dict) and "text" in p
            ) or str(content)
        elif not isinstance(content, str):
            content = str(content) if content is not None else ""
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _stream_chunks_to_output_text(chunks: list[Any]) -> str:
    """Concatenate delta.content from stream chunks into full output text."""
    reasoning, content = _stream_chunks_to_aggregated_deltas(chunks)
    return content or reasoning


def _stream_chunk_metadata(chunk: Any) -> dict:
    """Extract invariant metadata from a single stream chunk (id, model, etc.)."""
    out = {}
    for key in ("id", "created", "model", "object", "system_fingerprint"):
        if hasattr(chunk, key):
            try:
                out[key] = getattr(chunk, key)
            except Exception:
                pass
    return out


def _stream_chunks_to_aggregated_deltas(chunks: list[Any]) -> tuple[str, str]:
    """Return (reasoning_text, content_text) from all chunks. Kimi uses reasoning_content; many use content."""
    reasoning: list[str] = []
    content: list[str] = []
    for chunk in chunks:
        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
            continue
        delta = getattr(chunk.choices[0], "delta", None)
        if not delta:
            continue
        rc = getattr(delta, "reasoning_content", None)
        if isinstance(rc, str):
            reasoning.append(rc)
        c = getattr(delta, "content", None)
        if isinstance(c, str):
            content.append(c)
    return "".join(reasoning), "".join(content)


def _stream_chunks_to_condensed_payload(chunks: list[Any]) -> str:
    """Build a single JSON object: metadata (once) + aggregated reasoning and content. No repeated chunk blobs."""
    if not chunks:
        return json.dumps({"stream_metadata": {}, "reasoning": "", "content": ""}, ensure_ascii=False, default=str)
    metadata = _stream_chunk_metadata(chunks[0])
    reasoning, content = _stream_chunks_to_aggregated_deltas(chunks)
    payload = {
        "stream_metadata": metadata,
        "reasoning": reasoning,
        "content": content,
        "chunk_count": len(chunks),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _usage_from_tiktoken(messages: list, model: str, output_text: str) -> dict:
    """Compute input/output/total token counts using tiktoken when provider usage is missing."""
    input_text = _messages_to_text_for_counting(messages)
    input_tokens = _count_tokens_tiktoken(input_text, model)
    output_tokens = _count_tokens_tiktoken(output_text, model)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _hash_preview(s: str, max_len: int = 64) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:max_len]


def _sanitize_kwargs(kwargs: dict) -> dict:
    """Remove or redact sensitive keys; truncate large values."""
    out = {}
    skip = {"api_key", "api_key_id", "credentials"}
    for k, v in kwargs.items():
        if k.lower() in skip:
            continue
        if isinstance(v, str) and len(v) > 500:
            out[k] = _truncate(v, 500)
        elif isinstance(v, (list, dict)) and len(str(v)) > 1000:
            out[k] = "<truncated>"
        else:
            out[k] = v
    return out


def _write_log(
    request: LLMRequest,
    result: LLMResult,
    duration_ms: int | None,
    status: str = LLMCallLog.Status.SUCCESS,
    error_type: str | None = None,
    error_message: str = "",
    http_status: int | None = None,
    retry_count: int = 0,
) -> LLMCallLog | None:
    """Write full LLMCallLog. Returns None on failure (caller may try minimal)."""
    try:
        prompt_preview = ""
        if request.messages:
            parts = []
            for m in request.messages[:5]:
                content = m.get("content") if isinstance(m.get("content"), str) else str(m.get("content", ""))[:500]
                parts.append(content)
            prompt_preview = _truncate("\n".join(parts), 2000)
        user_message_preview = _user_message_preview(request.messages)
        raw_payload = _serialize_raw_response_payload(
            raw_response=result.raw_response,
            raw_chunks=getattr(result, "raw_response_chunks", None),
        )
        request_id = (request.metadata or {}).get("request_id", "")
        from decimal import Decimal
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        if (input_tokens == 0 and output_tokens == 0) and request.messages:
            tiktoken_usage = _usage_from_tiktoken(
                request.messages, request.model, result.text or ""
            )
            input_tokens = tiktoken_usage.get("input_tokens", 0)
            output_tokens = tiktoken_usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        cost_usd = None
        cost_source = None
        if result.cost is not None:
            cost_usd = Decimal(str(result.cost))
            cost_source = "litellm"
        else:
            fb = get_fallback_cost_usd(request.model, input_tokens, output_tokens)
            if fb is not None:
                cost_usd = fb
                cost_source = "fallback"
        log = LLMCallLog.objects.create(
            model=request.model,
            is_stream=request.stream,
            user=request.user,
            metadata=request.metadata or {},
            request_id=request_id,
            duration_ms=duration_ms,
            request_kwargs=_sanitize_kwargs(request.raw_kwargs),
            prompt_hash=_hash_preview(prompt_preview),
            prompt_preview=prompt_preview,
            user_message_preview=user_message_preview,
            provider_response_id=result.provider_response_id,
            response_model=result.response_model,
            response_preview=_truncate(result.text or "", 2000),
            response_hash=_hash_preview(result.text or ""),
            raw_response_payload=raw_payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
            status=status,
            error_type=error_type,
            error_message=_truncate(error_message, 2000),
            http_status=http_status,
            retry_count=retry_count,
        )
        return log
    except Exception as e:
        logger.warning("LLMCallLog full write failed: %s", e)
        return None


def _write_minimal_log(request: LLMRequest, note: str) -> None:
    """Fallback: minimal row with primitives only when full log fails."""
    try:
        LLMCallLog.objects.create(
            model=request.model or "unknown",
            is_stream=request.stream,
            status=LLMCallLog.Status.LOGGING_FAILED,
            error_message=_truncate(note, 500),
        )
    except Exception as e:
        logger.warning("LLMCallLog minimal write failed: %s", e)


def _run_pre_hooks(request: LLMRequest) -> None:
    for hook in get_pre_call_hooks():
        try:
            hook(request)
        except Exception as e:
            logger.info("Pre-call hook blocked: %s", e)
            raise


def _run_post_hooks(result: LLMResult) -> None:
    for hook in get_post_call_hooks():
        try:
            hook(result)
        except Exception as e:
            logger.info("Post-call hook blocked: %s", e)
            raise


def _save_log_with_timeout(request: LLMRequest, result: LLMResult, duration_ms: int | None, status: str = LLMCallLog.Status.SUCCESS, error_type: str | None = None, error_message: str = "", http_status: int | None = None, retry_count: int = 0) -> None:
    """Write log with DB timeout; on failure write minimal row."""
    timeout = get_log_write_timeout()
    try:
        connection.set_parameter("statement_timeout", int(timeout * 1000))
    except Exception:
        pass
    try:
        log = _write_log(request, result, duration_ms, status=status, error_type=error_type, error_message=error_message, http_status=http_status, retry_count=retry_count)
        if log is None:
            _write_minimal_log(request, "log write failed")
    except Exception as e:
        _write_minimal_log(request, f"log write failed: {e}")
    finally:
        try:
            connection.set_parameter("statement_timeout", 0)
        except Exception:
            pass


def _is_retryable(e: Exception) -> bool:
    s = str(e).lower()
    if "429" in s or "rate limit" in s or "timeout" in s:
        return True
    if "503" in s or "502" in s or "500" in s:
        return True
    return False


def completion(**kwargs: Any) -> Any:
    """
    Sync completion. Validates model, runs pre/post hooks, retries, logs.
    With stream=True returns an iterator that proxies provider chunks and writes LLMCallLog on exit.
    """
    request = _kwargs_to_request(**kwargs)
    if not is_model_allowed(request.model):
        from llm_service.conf import get_allowed_models
        raise ValueError(f"Model not allowed: {request.model}. Allowed: {get_allowed_models()}")
    _run_pre_hooks(request)
    client = get_client()
    start = time.perf_counter()
    last_error = None
    retry_count = 0
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        try:
            if request.stream:
                return _stream_sync(request, client, start)
            resp = client.completion(**request.to_completion_kwargs())
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = _response_to_result(resp, request.model)
            _run_post_hooks(result)
            _save_log_with_timeout(request, result, duration_ms)
            return resp
        except Exception as e:
            last_error = e
            retry_count = attempt
            if attempt < max_retries and _is_retryable(e):
                time.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = LLMResult(error=e, usage={}, text=None)
            _save_log_with_timeout(
                request, result, duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_count=retry_count,
            )
            raise
    raise last_error


def _stream_sync(request: LLMRequest, client: BaseLLMClient, start: float) -> Iterator[Any]:
    """Wrap stream so we finalize and write LLMCallLog when iterator ends or is closed. Retries whole stream on 503/502/500/429 when no chunks yielded yet."""
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        usage_holder: list[dict] = []
        final_response_holder: list[Any] = []
        try:
            stream = client.completion(**request.to_completion_kwargs())
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_holder.append({
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or getattr(chunk.usage, "input_tokens", 0),
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or getattr(chunk.usage, "output_tokens", 0),
                        "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                    })
                if getattr(chunk, "choices", None) and len(chunk.choices) > 0 and getattr(chunk.choices[0], "delta", None):
                    pass
                final_response_holder.append(chunk)
                yield chunk
            duration_ms = int((time.perf_counter() - start) * 1000)
            u = usage_holder[-1] if usage_holder else {}
            if not u or (u.get("input_tokens", 0) == 0 and u.get("output_tokens", 0) == 0):
                output_text = _stream_chunks_to_output_text(final_response_holder)
                u = _usage_from_tiktoken(request.messages, request.model, output_text)
            cost = None
            try:
                last = final_response_holder[-1] if final_response_holder else None
                if last and getattr(last, "_hidden_params", None):
                    cost = last._hidden_params.get("response_cost")
            except Exception:
                pass
            if cost is None and u:
                fb = get_fallback_cost_usd(request.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
                cost = float(fb) if fb is not None else None
            result = LLMResult(usage=u, cost=cost, text=None, raw_response_chunks=final_response_holder)
            _save_log_with_timeout(request, result, duration_ms)
            return
        except Exception as e:
            # Retry only if no chunks yielded yet (avoids duplicating content for the user)
            if attempt < max_retries and _is_retryable(e) and len(final_response_holder) == 0:
                logger.info("Stream retryable error before first chunk (attempt %s/%s): %s", attempt + 1, max_retries + 1, e)
                time.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            u = usage_holder[-1] if usage_holder else {}
            result = LLMResult(error=e, usage=u, text=None)
            _save_log_with_timeout(
                request, result, duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_count=attempt,
            )
            raise


async def acompletion(**kwargs: Any) -> Any:
    """
    Async completion. Same policy and logging as completion().
    With stream=True returns an async iterator.
    """
    request = _kwargs_to_request(**kwargs)
    if not is_model_allowed(request.model):
        from llm_service.conf import get_allowed_models
        raise ValueError(f"Model not allowed: {request.model}. Allowed: {get_allowed_models()}")
    _run_pre_hooks(request)
    client = get_client()
    start = time.perf_counter()
    last_error = None
    retry_count = 0
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        try:
            if request.stream:
                return _stream_async(request, client, start)
            resp = await client.acompletion(**request.to_completion_kwargs())
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = _response_to_result(resp, request.model)
            _run_post_hooks(result)
            _save_log_with_timeout(request, result, duration_ms)
            return resp
        except Exception as e:
            last_error = e
            retry_count = attempt
            if attempt < max_retries and _is_retryable(e):
                await asyncio.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = LLMResult(error=e, usage={}, text=None)
            _save_log_with_timeout(request, result, duration_ms, status=LLMCallLog.Status.ERROR, error_type=type(e).__name__, error_message=str(e), retry_count=retry_count)
            raise
    raise last_error


async def _stream_async(request: LLMRequest, client: BaseLLMClient, start: float) -> AsyncIterator[Any]:
    """Retries whole stream on 503/502/500/429 when no chunks yielded yet."""
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        usage_holder: list[dict] = []
        final_response_holder: list[Any] = []
        try:
            stream = await client.acompletion(**request.to_completion_kwargs())
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage_holder.append({
                        "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or getattr(chunk.usage, "input_tokens", 0),
                        "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or getattr(chunk.usage, "output_tokens", 0),
                        "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                    })
                final_response_holder.append(chunk)
                yield chunk
            duration_ms = int((time.perf_counter() - start) * 1000)
            u = usage_holder[-1] if usage_holder else {}
            if not u or (u.get("input_tokens", 0) == 0 and u.get("output_tokens", 0) == 0):
                output_text = _stream_chunks_to_output_text(final_response_holder)
                u = _usage_from_tiktoken(request.messages, request.model, output_text)
            cost = None
            try:
                last = final_response_holder[-1] if final_response_holder else None
                if last and getattr(last, "_hidden_params", None):
                    cost = last._hidden_params.get("response_cost")
            except Exception:
                pass
            if cost is None and u:
                fb = get_fallback_cost_usd(request.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
                cost = float(fb) if fb is not None else None
            result = LLMResult(usage=u, cost=cost, text=None, raw_response_chunks=final_response_holder)
            _save_log_with_timeout(request, result, duration_ms)
            return
        except Exception as e:
            if attempt < max_retries and _is_retryable(e) and len(final_response_holder) == 0:
                logger.info("Stream retryable error before first chunk (attempt %s/%s): %s", attempt + 1, max_retries + 1, e)
                await asyncio.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            u = usage_holder[-1] if usage_holder else {}
            result = LLMResult(error=e, usage=u, text=None)
            _save_log_with_timeout(
                request, result, duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_count=attempt,
            )
            raise