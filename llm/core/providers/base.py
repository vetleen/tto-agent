"""Base class for LangChain-backed ChatModel implementations.

Encapsulates the shared generate/stream logic so provider subclasses only
need to supply a configured LangChain chat model client.
"""

from __future__ import annotations

from typing import Iterator

from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.service.errors import LLMProviderError
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

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        try:
            result = client.invoke(lc_messages)
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
            usage = Usage(
                prompt_tokens=usage_meta.get("input_tokens"),
                completion_tokens=usage_meta.get("output_tokens"),
                total_tokens=usage_meta.get("total_tokens"),
                cost_usd=None,
            )

        return ChatResponse(message=message, model=self.name, usage=usage, metadata={})

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        lc_messages = to_langchain_messages(request.messages)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        run_id = request.context.run_id if request.context else ""
        sequence = 1

        yield StreamEvent(
            event_type="message_start",
            data={"model": self.name},
            sequence=sequence,
            run_id=run_id,
        )
        sequence += 1

        try:
            for chunk in client.stream(lc_messages):
                text = getattr(chunk, "content", "") or ""
                if not text:
                    continue
                yield StreamEvent(
                    event_type="token",
                    data={"text": str(text)},
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

        yield StreamEvent(
            event_type="message_end",
            data={"model": self.name},
            sequence=sequence,
            run_id=run_id,
        )


__all__ = ["BaseLangChainChatModel"]
