"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

from typing import Iterator

from llm.core.callbacks import PromptCaptureCallback
from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.service.errors import LLMProviderError
from llm.service.pricing import calculate_cost
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent


class BaseLangChainChatModel(ChatModel):
    """Shared generate/stream logic for all LangChain-backed providers.

    Subclasses must set ``self.name`` and ``self._client`` in their
    ``__init__`` (the LangChain chat model instance, e.g. ``ChatOpenAI``).
    They may override ``_provider_label`` for error messages.
    """

    name: str
    _client: object  # LangChain BaseChatModel instance
    _provider_label: str = "LLM"

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
        (OpenAI with ``stream_usage=True``, Anthropic, Gemini). Falls back to
        estimating output tokens via tiktoken if no provider data is available.
        """
        data: dict = {}
        usage_meta = getattr(last_chunk, "usage_metadata", None) if last_chunk else None
        if isinstance(usage_meta, dict) and usage_meta.get("output_tokens"):
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
        try:
            result = client.invoke(lc_messages, config={"callbacks": [callback]})
        except Exception as exc:
            raise LLMProviderError(
                f"{self._provider_label} generate failed for model={self.name}"
            ) from exc

        content = getattr(result, "content", "") or ""
        message_tool_calls = parse_tool_calls_from_ai_message(result)

        message = Message(
            role="assistant",
            content=str(content),
            tool_calls=message_tool_calls,
        )

        usage = None
        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict):
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
        except Exception as exc:
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
