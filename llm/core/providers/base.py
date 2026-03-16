"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

import logging
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

    def _get_callbacks(self, request: ChatRequest, callback: PromptCaptureCallback) -> list:
        """Build the callbacks list for a generate/stream call."""
        callbacks = [callback]
        usage_cb = (request.params or {}).get("_usage_callback")
        if usage_cb is not None:
            callbacks.append(usage_cb)
        return callbacks

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages, provider=self._provider_id)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        callback = PromptCaptureCallback()
        callbacks = self._get_callbacks(request, callback)
        config = self._build_config(request, callbacks)
        run_id = request.context.run_id if request.context else "n/a"
        logger.info(
            "LLM generate start model=%s provider=%s messages=%d run_id=%s",
            self.name, self._provider_label, len(request.messages), run_id,
        )
        try:
            result = client.invoke(lc_messages, config=config)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                logger.error(
                    "LLM generate rate limited model=%s provider=%s run_id=%s",
                    self.name, self._provider_label, run_id,
                )
                raise LLMRateLimitError(
                    f"{self._provider_label} rate limited for model={self.name}"
                ) from exc
            logger.exception(
                "LLM generate failed model=%s provider=%s run_id=%s",
                self.name, self._provider_label, run_id,
            )
            raise LLMProviderError(
                f"{self._provider_label} generate failed for model={self.name}"
            ) from exc

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

        raw_prompt = self._build_raw_prompt(callback, request.tool_schemas)
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
        metadata = {"raw_prompt": raw_prompt}
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

        callback = PromptCaptureCallback()
        callbacks = self._get_callbacks(request, callback)
        config = self._build_config(request, callbacks)
        raw_prompt_yielded = False
        last_chunk = None
        output_text_parts: list[str] = []

        try:
            for chunk in client.stream(lc_messages, config=config):
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
        except Exception as exc:
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

        end_data = self._extract_stream_usage(last_chunk, "".join(output_text_parts))
        end_data["model"] = self.name
        # Include response metadata from last chunk
        if last_chunk:
            resp_meta = self._extract_response_metadata(last_chunk)
            end_data.update(resp_meta)
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
