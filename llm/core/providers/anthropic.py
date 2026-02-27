from __future__ import annotations

import os
from typing import Iterator, List

from llm.core.interfaces import ChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except Exception as exc:
    ChatAnthropic = None  # type: ignore[assignment]
    AIMessage = HumanMessage = SystemMessage = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


def _to_lc_messages(messages: List[Message]):
    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        else:
            lc_messages.append(HumanMessage(content=m.content))
    return lc_messages


class AnthropicChatModel(ChatModel):
    """ChatModel backed by LangChain's ChatAnthropic."""

    def __init__(self, model_name: str) -> None:
        if _import_error is not None or ChatAnthropic is None:
            raise LLMProviderError(
                "langchain-anthropic is not installed or failed to import. "
                "Install with `pip install langchain-anthropic`."
            ) from _import_error

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not set; cannot initialize AnthropicChatModel."
            )

        self.name = model_name
        self._client = ChatAnthropic(model=model_name)

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = _to_lc_messages(request.messages)
        try:
            result = self._client.invoke(lc_messages)
        except Exception as exc:  # pragma: no cover
            raise LLMProviderError(f"Anthropic generate failed for model={self.name}") from exc

        content = getattr(result, "content", "") or ""
        message = Message(role="assistant", content=str(content))

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
        lc_messages = _to_lc_messages(request.messages)
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
            for chunk in self._client.stream(lc_messages):
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
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(
                event_type="error",
                data={"message": "Anthropic streaming failure", "details": str(exc)},
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


_registry = get_model_registry()
_registry.register_model_prefix("claude-", lambda name: AnthropicChatModel(name))
_registry.register_model_prefix("anthropic/", lambda name: AnthropicChatModel(name))


__all__ = ["AnthropicChatModel"]

