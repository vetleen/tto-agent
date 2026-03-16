"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.service.errors import LLMProviderError, LLMRateLimitError
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


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check whether an exception is a 429 rate-limit error."""
    return getattr(exc, "status_code", None) == 429


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
            reasoning_tokens = self._extract_reasoning_tokens(usage_meta)
            cost = calculate_cost(self.name, input_tokens, output_tokens, cached_tokens)
            data["input_tokens"] = input_tokens
            data["output_tokens"] = output_tokens
            data["total_tokens"] = total_tokens
            data["cached_tokens"] = cached_tokens
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
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
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
                if not _is_rate_limit_error(exc):
                    logger.exception(
                        "LLM generate failed model=%s provider=%s run_id=%s",
                        self.name, self._provider_label, run_id,
                    )
                    raise LLMProviderError(
                        f"{self._provider_label} generate failed for model={self.name}"
                    ) from exc
                last_exc = exc
                if attempt < _RATE_LIMIT_MAX_RETRIES:
                    wait = min(
                        _RATE_LIMIT_INITIAL_WAIT * (_RATE_LIMIT_BACKOFF_FACTOR ** attempt),
                        _RATE_LIMIT_MAX_WAIT,
                    )
                    logger.warning(
                        "LLM generate rate limited model=%s provider=%s "
                        "attempt=%d/%d waiting=%.0fs run_id=%s",
                        self.name, self._provider_label,
                        attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1,
                        wait, run_id,
                    )
                    time.sleep(wait)

        if result is None:
            logger.error(
                "LLM generate rate limited (retries exhausted) model=%s provider=%s run_id=%s",
                self.name, self._provider_label, run_id,
            )
            raise LLMRateLimitError(
                f"{self._provider_label} rate limited for model={self.name}"
            ) from last_exc

        raw_content = getattr(result, "content", "") or ""
        # Responses API may return content as a list of typed blocks;
        # normalise to a plain string.
        if isinstance(raw_content, list):
            content = "".join(
                block.get("text", "") for block in raw_content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            content = str(raw_content)
        message_tool_calls = parse_tool_calls_from_ai_message(result)

        message = Message(
            role="assistant",
            content=content,
            tool_calls=message_tool_calls,
        )

        usage = None
        usage_meta = self._extract_usage_dict(result)
        if usage_meta:
            input_tokens = usage_meta.get("input_tokens")
            output_tokens = usage_meta.get("output_tokens")
            # Extract cached token count from input_token_details
            details = usage_meta.get("input_token_details") or {}
            cached_tokens = details.get("cache_read") if isinstance(details, dict) else None
            reasoning_tokens = self._extract_reasoning_tokens(usage_meta)
            cost = calculate_cost(self.name, input_tokens, output_tokens, cached_tokens)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=usage_meta.get("total_tokens"),
                cached_tokens=cached_tokens,
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

        stream_succeeded = False
        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                for chunk in client.stream(lc_messages, config=config):
                    last_chunk = chunk
                    accumulated = chunk if accumulated is None else accumulated + chunk

                    for event_type, event_data in self._parse_chunk(chunk):
                        if event_type == "token":
                            output_text_parts.append(event_data.get("text", ""))
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
                if _is_rate_limit_error(exc) and attempt < _RATE_LIMIT_MAX_RETRIES:
                    # Only retry if no tokens have been streamed yet
                    if output_text_parts:
                        logger.error(
                            "LLM stream rate limited mid-stream model=%s provider=%s run_id=%s",
                            self.name, self._provider_label, run_id,
                        )
                        yield StreamEvent(
                            event_type="error",
                            data={"message": f"{self._provider_label} streaming failure", "details": str(exc)},
                            sequence=sequence,
                            run_id=run_id,
                        )
                        return
                    wait = min(
                        _RATE_LIMIT_INITIAL_WAIT * (_RATE_LIMIT_BACKOFF_FACTOR ** attempt),
                        _RATE_LIMIT_MAX_WAIT,
                    )
                    logger.warning(
                        "LLM stream rate limited model=%s provider=%s "
                        "attempt=%d/%d waiting=%.0fs run_id=%s",
                        self.name, self._provider_label,
                        attempt + 1, _RATE_LIMIT_MAX_RETRIES + 1,
                        wait, run_id,
                    )
                    time.sleep(wait)
                    # Reset state for retry
                    last_chunk = None
                    accumulated = None
                    output_text_parts = []
                else:
                    logger.exception(
                        "LLM stream error model=%s provider=%s run_id=%s",
                        self.name, self._provider_label, run_id,
                    )
                    yield StreamEvent(
                        event_type="error",
                        data={"message": f"{self._provider_label} streaming failure", "details": str(exc)},
                        sequence=sequence,
                        run_id=run_id,
                    )
                    return

        if not stream_succeeded:
            yield StreamEvent(
                event_type="error",
                data={"message": f"{self._provider_label} rate limited (retries exhausted)"},
                sequence=sequence,
                run_id=run_id,
            )
            return

        end_data = self._extract_stream_usage(last_chunk, "".join(output_text_parts))
        end_data["model"] = self.name
        # Include response metadata from last chunk
        if last_chunk:
            resp_meta = self._extract_response_metadata(last_chunk)
            end_data.update(resp_meta)

        # Include accumulated content and tool_calls for pipeline consumption
        if accumulated is not None:
            raw_content = getattr(accumulated, "content", "") or ""
            if isinstance(raw_content, list):
                end_data["content"] = "".join(
                    block.get("text", "") for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
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
