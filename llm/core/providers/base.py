"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from llm.core.callbacks import PromptCaptureCallback
from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.service.errors import LLMProviderError, LLMRateLimitError
from llm.service.pricing import calculate_cost
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

logger = logging.getLogger(__name__)


class BaseLangChainChatModel(ChatModel):
    """Shared generate/stream logic for all LangChain-backed providers.

    Subclasses must set ``self.name`` and ``self._client`` in their
    ``__init__`` (the LangChain chat model instance, e.g. ``ChatOpenAI``).
    They may override ``_provider_label`` for error messages.
    """

    name: str
    _client: object  # LangChain BaseChatModel instance
    _provider_label: str = "LLM"

    _RATE_LIMIT_MAX_RETRIES = 2  # 3 total attempts
    _RATE_LIMIT_BACKOFF_BASE = 1.0  # delays: 1s, 2s

    @staticmethod
    def _is_rate_limit_error(exc: BaseException) -> bool:
        """Check if an exception is a 429 rate-limit error."""
        return getattr(exc, "status_code", None) == 429

    @staticmethod
    def _extract_usage_dict(result: object) -> dict | None:
        """Extract a usage dict with input/output/total tokens from a LangChain result.

        Checks ``usage_metadata`` first (standard path), then falls back to
        ``response_metadata`` which may contain raw provider data (e.g. OpenAI
        Responses API puts usage in ``response_metadata["usage"]``).
        """
        # Primary: usage_metadata (populated by most LangChain providers)
        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta.get("output_tokens"):
            return usage_meta

        # Fallback: response_metadata.usage (OpenAI Responses API / raw provider data)
        resp_meta = getattr(result, "response_metadata", None)
        if isinstance(resp_meta, dict):
            # OpenAI Responses API: response_metadata["usage"]
            usage = resp_meta.get("usage")
            if isinstance(usage, dict) and usage.get("output_tokens"):
                return usage
            # Chat Completions API fallback: response_metadata["token_usage"]
            token_usage = resp_meta.get("token_usage")
            if isinstance(token_usage, dict):
                # Normalize prompt_tokens/completion_tokens to input/output
                return {
                    "input_tokens": token_usage.get("prompt_tokens") or token_usage.get("input_tokens"),
                    "output_tokens": token_usage.get("completion_tokens") or token_usage.get("output_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                }

        return None

    @staticmethod
    def _build_raw_prompt(callback: PromptCaptureCallback, tool_schemas: list | None) -> dict | None:
        """Assemble the raw_prompt dict from callback messages + request tool schemas."""
        if callback.captured_messages is None:
            return None
        serialized_tools = []
        for tool in (tool_schemas or []):
            try:
                serialized_tools.append({
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                })
            except Exception:
                serialized_tools.append(str(tool))
        return {
            "messages": callback.captured_messages,
            "tools": serialized_tools,
        }

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

        Checks ``last_chunk.usage_metadata`` for provider-reported token counts
        (OpenAI with ``stream_usage=True``, Anthropic, Gemini), then falls back
        to ``response_metadata`` (OpenAI Responses API).  Last resort: estimate
        output tokens via tiktoken.
        """
        data: dict = {}
        usage_meta = self._extract_usage_dict(last_chunk) if last_chunk else None
        if usage_meta and usage_meta.get("output_tokens"):
            input_tokens = usage_meta.get("input_tokens")
            output_tokens = usage_meta.get("output_tokens")
            total_tokens = usage_meta.get("total_tokens")
            details = usage_meta.get("input_token_details") or {}
            cached_tokens = details.get("cache_read") if isinstance(details, dict) else None
            cost = calculate_cost(self.name, input_tokens, output_tokens, cached_tokens)
            data["input_tokens"] = input_tokens
            data["output_tokens"] = output_tokens
            data["total_tokens"] = total_tokens
            data["cached_tokens"] = cached_tokens
            data["cost_usd"] = float(cost) if cost is not None else None
        elif output_text:
            # Fallback: estimate output tokens only
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                output_tokens = len(enc.encode(output_text))
            except Exception:
                output_tokens = len(output_text) // 4  # rough estimate
            cost = calculate_cost(self.name, None, output_tokens)
            data["output_tokens"] = output_tokens
            data["cost_usd"] = float(cost) if cost is not None else None
        return data

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        callback = PromptCaptureCallback()
        last_exc: Exception | None = None
        for attempt in range(1 + self._RATE_LIMIT_MAX_RETRIES):
            try:
                result = client.invoke(lc_messages, config={"callbacks": [callback]})
                break
            except Exception as exc:
                last_exc = exc
                if self._is_rate_limit_error(exc) and attempt < self._RATE_LIMIT_MAX_RETRIES:
                    wait = self._RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "%s rate limited (attempt %d/%d), retrying in %.1fs",
                        self._provider_label, attempt + 1,
                        1 + self._RATE_LIMIT_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                if self._is_rate_limit_error(exc):
                    raise LLMRateLimitError(
                        f"{self._provider_label} rate limited after "
                        f"{1 + self._RATE_LIMIT_MAX_RETRIES} attempts for model={self.name}"
                    ) from exc
                raise LLMProviderError(
                    f"{self._provider_label} generate failed for model={self.name}"
                ) from exc
        else:
            raise LLMRateLimitError(
                f"{self._provider_label} rate limited after "
                f"{1 + self._RATE_LIMIT_MAX_RETRIES} attempts for model={self.name}"
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
            cost = calculate_cost(self.name, input_tokens, output_tokens, cached_tokens)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=usage_meta.get("total_tokens"),
                cached_tokens=cached_tokens,
                cost_usd=float(cost) if cost is not None else None,
            )

        raw_prompt = self._build_raw_prompt(callback, request.tool_schemas)
        return ChatResponse(message=message, model=self.name, usage=usage, metadata={"raw_prompt": raw_prompt})

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        lc_messages = to_langchain_messages(request.messages)
        client = self._get_streaming_client(request)
        run_id = request.context.run_id if request.context else ""
        sequence = 1

        yield StreamEvent(
            event_type="message_start",
            data={"model": self.name},
            sequence=sequence,
            run_id=run_id,
        )
        sequence += 1

        callback = PromptCaptureCallback()
        raw_prompt_yielded = False
        last_chunk = None
        output_text_parts: list[str] = []

        for attempt in range(1 + self._RATE_LIMIT_MAX_RETRIES):
            try:
                for chunk in client.stream(lc_messages, config={"callbacks": [callback]}):
                    last_chunk = chunk
                    if not raw_prompt_yielded:
                        raw_prompt = self._build_raw_prompt(callback, request.tool_schemas)
                        yield StreamEvent(
                            event_type="raw_prompt",
                            data={"raw_prompt": raw_prompt},
                            sequence=sequence,
                            run_id=run_id,
                        )
                        sequence += 1
                        raw_prompt_yielded = True

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
                break  # success
            except Exception as exc:
                if (
                    self._is_rate_limit_error(exc)
                    and not raw_prompt_yielded
                    and attempt < self._RATE_LIMIT_MAX_RETRIES
                ):
                    wait = self._RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "%s stream rate limited (attempt %d/%d), retrying in %.1fs",
                        self._provider_label, attempt + 1,
                        1 + self._RATE_LIMIT_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                yield StreamEvent(
                    event_type="error",
                    data={"message": f"{self._provider_label} streaming failure", "details": str(exc)},
                    sequence=sequence,
                    run_id=run_id,
                )
                return

        end_data = self._extract_stream_usage(last_chunk, "".join(output_text_parts))
        end_data["model"] = self.name
        yield StreamEvent(
            event_type="message_end",
            data=end_data,
            sequence=sequence,
            run_id=run_id,
        )


__all__ = ["BaseLangChainChatModel"]
