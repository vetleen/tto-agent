"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

from dataclasses import dataclass

from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.service.errors import (
    LLMAuthError,
    LLMConnectionError,
    LLMOverloadedError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestTooLargeError,
    LLMTimeoutError,
)
from llm.service.pricing import calculate_cost
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

logger = logging.getLogger(__name__)

# Rate-limit retry settings
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_INITIAL_WAIT = 30  # seconds
_RATE_LIMIT_BACKOFF_FACTOR = 2
_RATE_LIMIT_MAX_WAIT = 120  # seconds

# Token-related keywords that indicate a request-too-large error on 400
_TOKEN_KEYWORDS = ("token", "too long", "too large", "context length", "max_tokens")


@dataclass(frozen=True)
class ClassifiedError:
    """Result of classifying an API error."""

    error_code: str
    user_message: str
    log_level: str  # "warning" or "error"


# Mapping from error_code to the appropriate exception subclass
_ERROR_CODE_TO_EXCEPTION: dict[str, type[LLMProviderError]] = {
    "rate_limited": LLMRateLimitError,
    "overloaded": LLMOverloadedError,
    "auth_error": LLMAuthError,
    "request_too_large": LLMRequestTooLargeError,
    "timeout": LLMTimeoutError,
    "connection_error": LLMConnectionError,
    "server_error": LLMProviderError,
    "unknown": LLMProviderError,
}


def _extract_body_error_type(exc: Exception) -> str | None:
    """Extract ``body["error"]["type"]`` from a provider SDK exception, if present.

    Mid-stream errors from Anthropic (and similar SSE-based APIs) arrive over a
    200 OK response, so ``exc.status_code`` reflects the stream open, not the
    real failure. The structured error type is carried in the SSE body.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            t = err.get("type")
            if isinstance(t, str):
                return t
    return None


def classify_api_error(exc: Exception, provider_label: str) -> ClassifiedError:
    """Inspect an exception and return a classified error with user-facing message."""
    status = getattr(exc, "status_code", None)
    msg_lower = str(exc).lower()
    body_error_type = _extract_body_error_type(exc)

    if status == 429 or body_error_type == "rate_limit_error":
        return ClassifiedError(
            error_code="rate_limited",
            user_message=f"{provider_label} is rate limiting requests. Please wait a moment and try again.",
            log_level="warning",
        )
    if status in (503, 529) or body_error_type == "overloaded_error":
        return ClassifiedError(
            error_code="overloaded",
            user_message=(
                f"{provider_label} is currently overloaded. This is a temporary issue on their end "
                "— try again in a few minutes or switch to a different model."
            ),
            log_level="warning",
        )
    if status in (401, 403):
        return ClassifiedError(
            error_code="auth_error",
            user_message=f"Authentication failed with {provider_label}. Please contact support.",
            log_level="error",
        )
    if status == 400 and any(kw in msg_lower for kw in _TOKEN_KEYWORDS):
        return ClassifiedError(
            error_code="request_too_large",
            user_message=(
                "The request was too large for the model to process. "
                "Try removing some attachments or shortening the conversation."
            ),
            log_level="warning",
        )
    if status == 500:
        return ClassifiedError(
            error_code="server_error",
            user_message=f"{provider_label} experienced an internal error. Please try again.",
            log_level="error",
        )
    if status == 408 or isinstance(exc, TimeoutError):
        return ClassifiedError(
            error_code="timeout",
            user_message=f"The request to {provider_label} timed out. Please try again.",
            log_level="warning",
        )
    if isinstance(exc, (ConnectionError, OSError)):
        return ClassifiedError(
            error_code="connection_error",
            user_message=f"Unable to reach {provider_label}. Please check your connection and try again.",
            log_level="warning",
        )
    return ClassifiedError(
        error_code="unknown",
        user_message=f"{provider_label} encountered an unexpected error. Please try again.",
        log_level="error",
    )


def _highlight_if_unmapped(classified: ClassifiedError, exc: Exception, model: str, provider_label: str, run_id: str) -> None:
    """Emit a distinct, Sentry-actionable line when an error hits the ``unknown``
    catch-all in :func:`classify_api_error`.

    These are failures we have no curated user-facing message for yet. The
    dedicated message (grouped in Sentry by ``exc_type``) flags each new failure
    mode so we can add a branch to ``classify_api_error`` with a friendly message
    and resolve the Sentry issue — incrementally growing the mapping.
    """
    if classified.error_code != "unknown":
        return
    logger.error(
        "Unmapped LLM provider error — add a branch to classify_api_error so the "
        "user gets a specific message. exc_type=%s model=%s provider=%s run_id=%s",
        type(exc).__name__, model, provider_label, run_id,
        exc_info=True,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check whether an exception is a 429 rate-limit error.

    Covers both request-level errors (``status_code=429``) and mid-stream
    errors where the SSE body carries ``error.type == "rate_limit_error"``.
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    return _extract_body_error_type(exc) == "rate_limit_error"


def _is_overloaded_error(exc: Exception) -> bool:
    """Check whether an exception is a 503/529 overloaded error.

    Covers both request-level errors (``status_code`` in 503/529) and
    mid-stream errors where the SSE body carries ``error.type == "overloaded_error"``.
    """
    if getattr(exc, "status_code", None) in (503, 529):
        return True
    return _extract_body_error_type(exc) == "overloaded_error"


def _is_retryable_transient_error(exc: Exception) -> bool:
    """Transient provider errors safe to retry with backoff."""
    return _is_rate_limit_error(exc) or _is_overloaded_error(exc)


def _get_retry_controls(request: ChatRequest):
    """Extract (cancel_event, cancel_check, deadline_dt) from a request.

    ``_cancel_event`` (threading.Event) is set by the chat consumer on the
    streaming path; ``_cancel_check`` (callable) by the sync/subagent path.
    ``deadline_dt`` is the absolute run deadline when the context carries one.
    """
    params = request.params or {}
    cancel_event = params.get("_cancel_event")
    cancel_check = params.get("_cancel_check")
    deadline_dt = None
    ctx = request.context
    if ctx is not None and getattr(ctx, "deadline_seconds", None):
        deadline_dt = ctx.started_at + timedelta(seconds=ctx.deadline_seconds)
    return cancel_event, cancel_check, deadline_dt


def _is_cancelled(cancel_event, cancel_check) -> bool:
    if cancel_event is not None and cancel_event.is_set():
        return True
    if cancel_check is not None and cancel_check():
        return True
    return False


def _wait_before_retry(wait: float, *, cancel_event=None, cancel_check=None, deadline_dt=None) -> bool:
    """Sleep *wait* seconds before a retry, honouring cancellation and deadlines.

    Returns False when the retry should be abandoned instead: the run was
    cancelled, or the deadline would pass before the wait completes (in which
    case it doesn't sleep at all). With no cancel mechanism and no deadline
    this is a single ``time.sleep(wait)``.
    """
    if deadline_dt is not None and datetime.now(timezone.utc) + timedelta(seconds=wait) >= deadline_dt:
        return False
    if cancel_event is None and cancel_check is None:
        time.sleep(wait)
        return True
    # Sleep in ~1s slices so a user cancel takes effect promptly instead of
    # blocking the worker thread for up to 120s.
    remaining = float(wait)
    while remaining > 0:
        if _is_cancelled(cancel_event, cancel_check):
            return False
        step = min(1.0, remaining)
        time.sleep(step)
        remaining -= step
    return not _is_cancelled(cancel_event, cancel_check)


class BaseLangChainChatModel(ChatModel):
    """Shared generate/stream logic for all LangChain-backed providers.

    Constructed by the model factory with a pre-built LangChain client.
    Subclasses may override ``_parse_chunk`` and ``_get_streaming_client``
    for provider-specific behavior (thinking/reasoning).
    """

    name: str
    _client: object  # LangChain BaseChatModel instance
    _provider_label: str = "LLM"
    _provider_id: str | None = None  # e.g. "openai", "anthropic", "google_genai"

    def __init__(self, model_name: str, client: object) -> None:
        self.name = model_name
        self._client = client

    @staticmethod
    def _extract_usage_dict(result: object) -> dict | None:
        """Extract a usage dict with input/output/total tokens from a LangChain result.

        Checks ``usage_metadata`` first (standard path), then falls back to
        ``response_metadata["usage"]`` (OpenAI Responses API edge case).
        """
        # Primary: usage_metadata (populated by all LangChain providers with stream_usage=True)
        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta.get("output_tokens"):
            return usage_meta

        # Fallback: response_metadata.usage (OpenAI Responses API / raw provider data)
        resp_meta = getattr(result, "response_metadata", None)
        if isinstance(resp_meta, dict):
            usage = resp_meta.get("usage")
            if isinstance(usage, dict) and usage.get("output_tokens"):
                return usage

        return None

    @staticmethod
    def _extract_response_metadata(result: object) -> dict:
        """Extract stop_reason, provider_model_id, and full response_metadata."""
        resp_meta = getattr(result, "response_metadata", None) or {}
        stop = (
            resp_meta.get("stop_reason")       # Anthropic
            or resp_meta.get("finish_reason")   # OpenAI
            or ""
        )
        model_id = (
            resp_meta.get("model_id")           # Anthropic
            or resp_meta.get("model_name")      # OpenAI
            or resp_meta.get("model")           # generic
            or ""
        )
        return {
            "response_metadata": resp_meta,
            "stop_reason": stop,
            "provider_model_id": model_id,
        }

    @staticmethod
    def _extract_reasoning_tokens(usage_meta: dict | None) -> int | None:
        """Extract reasoning/thinking output tokens from usage metadata."""
        if not usage_meta:
            return None
        details = usage_meta.get("output_token_details") or {}
        if isinstance(details, dict):
            return details.get("reasoning") or details.get("reasoning_tokens")
        return None

    @staticmethod
    def _extract_cache_write_1h_tokens(result: object) -> int | None:
        """Extract the 1h-TTL portion of cache-write tokens (Anthropic only).

        Anthropic's raw usage carries a ``cache_creation`` breakdown
        (``ephemeral_1h_input_tokens`` / ``ephemeral_5m_input_tokens``) inside
        ``response_metadata["usage"]``. Returns None when absent, in which
        case cost falls back to billing all cache writes at the 5m rate.
        """
        resp_meta = getattr(result, "response_metadata", None)
        if not isinstance(resp_meta, dict):
            return None
        usage = resp_meta.get("usage")
        if not isinstance(usage, dict):
            return None
        cache_creation = usage.get("cache_creation")
        if not isinstance(cache_creation, dict):
            return None
        tokens = cache_creation.get("ephemeral_1h_input_tokens")
        return tokens if isinstance(tokens, int) and tokens > 0 else None

    def _build_config(self, request: ChatRequest, callbacks: list) -> dict:
        """Build LangChain RunnableConfig with tracing metadata."""
        config: dict = {"callbacks": callbacks}
        ctx = request.context
        if ctx:
            config["metadata"] = {
                "run_id": ctx.run_id,
                "trace_id": ctx.trace_id,
                "user_id": ctx.user_id,
                "conversation_id": ctx.conversation_id,
            }
            config["tags"] = [
                f"user:{ctx.user_id or 'anon'}",
                f"model:{self.name}",
            ]
            config["run_name"] = f"{self._provider_label}/{self.name}"
        return config

    def _get_streaming_client(self, request: ChatRequest):
        """Return the LangChain client for streaming.

        Subclasses may override to return a differently-configured client
        (e.g. with thinking/reasoning enabled) based on *request.params*.
        """
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        return client

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        """Extract ``(event_type, data)`` pairs from a single LangChain chunk.

        Override in provider subclasses to handle thinking/reasoning content.
        """
        text = getattr(chunk, "content", "") or ""
        if not text:
            return []
        return [("token", {"text": str(text)})]

    def _extract_stream_usage(self, last_chunk: object | None, output_text: str) -> dict:
        """Build usage dict from the final streaming chunk.

        With ``stream_usage=True`` set consistently via the factory, usage_metadata
        is populated reliably on the final chunk for all providers.
        """
        data: dict = {}
        usage_meta = self._extract_usage_dict(last_chunk) if last_chunk else None
        if usage_meta and usage_meta.get("output_tokens"):
            input_tokens = usage_meta.get("input_tokens")
            output_tokens = usage_meta.get("output_tokens")
            total_tokens = usage_meta.get("total_tokens")
            details = usage_meta.get("input_token_details") or {}
            cached_tokens = details.get("cache_read") if isinstance(details, dict) else None
            cache_write_tokens = details.get("cache_creation") if isinstance(details, dict) else None
            reasoning_tokens = self._extract_reasoning_tokens(usage_meta)
            cost = calculate_cost(
                self.name, input_tokens, output_tokens, cached_tokens, cache_write_tokens,
                cache_write_1h_tokens=self._extract_cache_write_1h_tokens(last_chunk),
            )
            data["input_tokens"] = input_tokens
            data["output_tokens"] = output_tokens
            data["total_tokens"] = total_tokens
            data["cached_tokens"] = cached_tokens
            data["cache_write_tokens"] = cache_write_tokens
            data["reasoning_tokens"] = reasoning_tokens
            data["cost_usd"] = float(cost) if cost is not None else None
        elif output_text:
            logger.warning(
                "No usage_metadata on final stream chunk for model=%s; "
                "usage data will be incomplete.",
                self.name,
            )
        return data

    def _get_callbacks(self, request: ChatRequest) -> list:
        """Build the callbacks list for a generate/stream call."""
        callbacks = []
        usage_cb = (request.params or {}).get("_usage_callback")
        if usage_cb is not None:
            callbacks.append(usage_cb)
        return callbacks

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages, provider=self._provider_id)
        client = self._get_streaming_client(request)
        callbacks = self._get_callbacks(request)
        config = self._build_config(request, callbacks)
        run_id = request.context.run_id if request.context else "n/a"
        logger.info(
            "LLM generate start model=%s provider=%s messages=%d run_id=%s",
            self.name, self._provider_label, len(request.messages), run_id,
        )

        result = None
        last_exc: Exception | None = None
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                result = client.invoke(lc_messages, config=config)
                break
            except Exception as exc:
                if not _is_retryable_transient_error(exc):
                    classified = classify_api_error(exc, self._provider_label)
                    log_fn = getattr(logger, classified.log_level, logger.error)
                    log_fn(
                        "LLM generate failed model=%s provider=%s error_code=%s run_id=%s",
                        self.name, self._provider_label, classified.error_code, run_id,
                        exc_info=True,
                    )
                    _highlight_if_unmapped(classified, exc, self.name, self._provider_label, run_id)
                    exc_cls = _ERROR_CODE_TO_EXCEPTION.get(classified.error_code, LLMProviderError)
                    raise exc_cls(
                        classified.user_message,
                        error_code=classified.error_code,
                    ) from exc
                last_exc = exc
                if attempt < _RATE_LIMIT_MAX_RETRIES:
                    wait = min(
                        _RATE_LIMIT_INITIAL_WAIT * (_RATE_LIMIT_BACKOFF_FACTOR ** attempt),
                        _RATE_LIMIT_MAX_WAIT,
                    )
                    err_code = classify_api_error(exc, self._provider_label).error_code
                    # Mid-retry: only the "retries exhausted" path below is alert-worthy.
                    logger.info(
                        "LLM generate transient error model=%s provider=%s error_code=%s "
                        "attempt=%d/%d waiting=%.0fs run_id=%s",
                        self.name, self._provider_label, err_code,
                        attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1,
                        wait, run_id,
                    )
                    cancel_event, cancel_check, deadline_dt = _get_retry_controls(request)
                    if not _wait_before_retry(
                        wait,
                        cancel_event=cancel_event,
                        cancel_check=cancel_check,
                        deadline_dt=deadline_dt,
                    ):
                        # Cancelled or deadline too close to wait out the
                        # backoff — treat as retries exhausted.
                        logger.info(
                            "LLM generate retry wait aborted (cancelled or deadline) "
                            "model=%s provider=%s run_id=%s",
                            self.name, self._provider_label, run_id,
                        )
                        break

        if result is None:
            classified = classify_api_error(last_exc, self._provider_label)
            logger.error(
                "LLM generate transient retries exhausted model=%s provider=%s error_code=%s run_id=%s",
                self.name, self._provider_label, classified.error_code, run_id,
            )
            exc_cls = _ERROR_CODE_TO_EXCEPTION.get(classified.error_code, LLMProviderError)
            raise exc_cls(
                classified.user_message,
                error_code=classified.error_code,
            ) from last_exc

        raw_content = getattr(result, "content", "") or ""
        content_blocks = None
        if isinstance(raw_content, list):
            content = "".join(
                block.get("text", "") for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            full_blocks = [
                block for block in raw_content
                if isinstance(block, dict) and block.get("type") in ("thinking", "text")
            ]
            if any(b.get("type") == "thinking" for b in full_blocks):
                content_blocks = full_blocks
        else:
            content = str(raw_content)
        message_tool_calls = parse_tool_calls_from_ai_message(result)

        msg_meta = {}
        if content_blocks:
            msg_meta["content_blocks"] = content_blocks
        message = Message(
            role="assistant",
            content=content,
            tool_calls=message_tool_calls,
            metadata=msg_meta,
        )

        usage = None
        usage_meta = self._extract_usage_dict(result)
        if usage_meta:
            input_tokens = usage_meta.get("input_tokens")
            output_tokens = usage_meta.get("output_tokens")
            # Extract cached token count from input_token_details
            details = usage_meta.get("input_token_details") or {}
            cached_tokens = details.get("cache_read") if isinstance(details, dict) else None
            cache_write_tokens = details.get("cache_creation") if isinstance(details, dict) else None
            reasoning_tokens = self._extract_reasoning_tokens(usage_meta)
            cost = calculate_cost(
                self.name, input_tokens, output_tokens, cached_tokens, cache_write_tokens,
                cache_write_1h_tokens=self._extract_cache_write_1h_tokens(result),
            )
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=usage_meta.get("total_tokens"),
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=float(cost) if cost is not None else None,
            )

        resp_meta = self._extract_response_metadata(result)
        logger.info(
            "LLM generate complete model=%s provider=%s "
            "input_tokens=%s output_tokens=%s cost_usd=%s run_id=%s",
            self.name, self._provider_label,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            usage.cost_usd if usage else None,
            run_id,
        )
        metadata = {}
        metadata.update(resp_meta)
        return ChatResponse(message=message, model=self.name, usage=usage, metadata=metadata)

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        lc_messages = to_langchain_messages(request.messages, provider=self._provider_id)
        client = self._get_streaming_client(request)
        run_id = request.context.run_id if request.context else ""
        logger.info(
            "LLM stream start model=%s provider=%s messages=%d run_id=%s",
            self.name, self._provider_label, len(request.messages), run_id,
        )
        sequence = 1

        yield StreamEvent(
            event_type="message_start",
            data={"model": self.name},
            sequence=sequence,
            run_id=run_id,
        )
        sequence += 1

        callbacks = self._get_callbacks(request)
        config = self._build_config(request, callbacks)
        last_chunk = None
        accumulated = None  # AIMessageChunk accumulator for tool_calls aggregation
        output_text_parts: list[str] = []
        # True once any token/thinking event has been yielded in the current
        # attempt. A retry after that point would replay already-delivered
        # content (duplicate thinking/text on the client), so it becomes a
        # terminal error instead.
        events_yielded = False

        stream_succeeded = False
        last_exc: Exception | None = None
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                for chunk in client.stream(lc_messages, config=config):
                    last_chunk = chunk
                    accumulated = chunk if accumulated is None else accumulated + chunk

                    for event_type, event_data in self._parse_chunk(chunk):
                        if event_type == "token":
                            output_text_parts.append(event_data.get("text", ""))
                        events_yielded = True
                        yield StreamEvent(
                            event_type=event_type,
                            data=event_data,
                            sequence=sequence,
                            run_id=run_id,
                        )
                        sequence += 1
                stream_succeeded = True
                break
            except Exception as exc:
                last_exc = exc
                if _is_retryable_transient_error(exc) and attempt < _RATE_LIMIT_MAX_RETRIES:
                    # Only retry if nothing (token or thinking) has streamed yet
                    if events_yielded:
                        mid_classified = classify_api_error(exc, self._provider_label)
                        logger.error(
                            "LLM stream transient error mid-stream model=%s provider=%s error_code=%s run_id=%s",
                            self.name, self._provider_label, mid_classified.error_code, run_id,
                        )
                        yield StreamEvent(
                            event_type="error",
                            data={
                                "message": mid_classified.user_message,
                                "error_code": mid_classified.error_code,
                                "details": str(exc),
                            },
                            sequence=sequence,
                            run_id=run_id,
                        )
                        return
                    wait = min(
                        _RATE_LIMIT_INITIAL_WAIT * (_RATE_LIMIT_BACKOFF_FACTOR ** attempt),
                        _RATE_LIMIT_MAX_WAIT,
                    )
                    err_code = classify_api_error(exc, self._provider_label).error_code
                    # Mid-retry: only the "retries exhausted" path below is alert-worthy.
                    logger.info(
                        "LLM stream transient error model=%s provider=%s error_code=%s "
                        "attempt=%d/%d waiting=%.0fs run_id=%s",
                        self.name, self._provider_label, err_code,
                        attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1,
                        wait, run_id,
                    )
                    cancel_event, cancel_check, deadline_dt = _get_retry_controls(request)
                    if not _wait_before_retry(
                        wait,
                        cancel_event=cancel_event,
                        cancel_check=cancel_check,
                        deadline_dt=deadline_dt,
                    ):
                        if _is_cancelled(cancel_event, cancel_check):
                            # User cancelled — end the stream quietly; the
                            # consumer already stopped listening.
                            logger.info(
                                "LLM stream retry wait cancelled model=%s provider=%s run_id=%s",
                                self.name, self._provider_label, run_id,
                            )
                            return
                        # Deadline too close to wait out the backoff — fall
                        # through to the retries-exhausted error event.
                        break
                    # Reset state for retry
                    last_chunk = None
                    accumulated = None
                    output_text_parts = []
                else:
                    classified = classify_api_error(exc, self._provider_label)
                    log_fn = getattr(logger, classified.log_level, logger.error)
                    log_fn(
                        "LLM stream error model=%s provider=%s error_code=%s run_id=%s",
                        self.name, self._provider_label, classified.error_code, run_id,
                        exc_info=True,
                    )
                    _highlight_if_unmapped(classified, exc, self.name, self._provider_label, run_id)
                    yield StreamEvent(
                        event_type="error",
                        data={
                            "message": classified.user_message,
                            "error_code": classified.error_code,
                            "details": str(exc),
                        },
                        sequence=sequence,
                        run_id=run_id,
                    )
                    return

        if not stream_succeeded:
            classified = classify_api_error(last_exc, self._provider_label) if last_exc else ClassifiedError(
                error_code="rate_limited",
                user_message=f"{self._provider_label} is rate limiting requests. Please wait a moment and try again.",
                log_level="warning",
            )
            logger.error(
                "LLM stream transient retries exhausted model=%s provider=%s error_code=%s run_id=%s",
                self.name, self._provider_label, classified.error_code, run_id,
            )
            yield StreamEvent(
                event_type="error",
                data={
                    "message": classified.user_message,
                    "error_code": classified.error_code,
                },
                sequence=sequence,
                run_id=run_id,
            )
            return

        # Use accumulated (not last_chunk) for usage and metadata extraction.
        # LangChain may append a trailing empty chunk with chunk_position="last"
        # that lacks usage_metadata; the accumulated message retains it.
        usage_source = accumulated if accumulated is not None else last_chunk
        end_data = self._extract_stream_usage(usage_source, "".join(output_text_parts))
        end_data["model"] = self.name
        if usage_source:
            resp_meta = self._extract_response_metadata(usage_source)
            end_data.update(resp_meta)

        # Include accumulated content and tool_calls for pipeline consumption
        if accumulated is not None:
            raw_content = getattr(accumulated, "content", "") or ""
            if isinstance(raw_content, list):
                end_data["content"] = "".join(
                    block.get("text", "") for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                # Preserve thinking+text blocks for conversation history
                full_blocks = [
                    block for block in raw_content
                    if isinstance(block, dict) and block.get("type") in ("thinking", "text")
                ]
                if any(b.get("type") == "thinking" for b in full_blocks):
                    end_data["content_blocks"] = full_blocks
            else:
                end_data["content"] = str(raw_content)
            tool_calls = parse_tool_calls_from_ai_message(accumulated)
            if tool_calls:
                end_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        logger.info(
            "LLM stream complete model=%s provider=%s "
            "output_tokens=%s cost_usd=%s run_id=%s",
            self.name, self._provider_label,
            end_data.get("output_tokens"), end_data.get("cost_usd"), run_id,
        )
        yield StreamEvent(
            event_type="message_end",
            data=end_data,
            sequence=sequence,
            run_id=run_id,
        )


__all__ = ["BaseLangChainChatModel"]
