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
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except Exception as exc:  # ImportError or other environment issues
    # Defer failure until provider is actually constructed
    ChatOpenAI = None  # type: ignore[assignment]
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
            # Treat "user" and any other roles as human messages for v1
            lc_messages.append(HumanMessage(content=m.content))
    return lc_messages


class OpenAIChatModel(ChatModel):
    """ChatModel backed by LangChain's ChatOpenAI."""

    def __init__(self, model_name: str) -> None:
        if _import_error is not None or ChatOpenAI is None:
            raise LLMProviderError(
                "langchain-openai is not installed or failed to import. "
                "Install with `pip install langchain-openai`."
            ) from _import_error

        if not os.getenv("OPENAI_API_KEY"):
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not set; cannot initialize OpenAIChatModel."
            )

        self.name = model_name
        # Let ChatOpenAI read configuration from environment; enable streaming usage accounting.
        self._client = ChatOpenAI(model=model_name, stream_usage=True)

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = _to_lc_messages(request.messages)
        try:
            result = self._client.invoke(lc_messages)
        except Exception as exc:  # pragma: no cover - wrapped in higher level tests
            raise LLMProviderError(f"OpenAI generate failed for model={self.name}") from exc

        if hasattr(result, "content"):
            content = result.content  # type: ignore[assignment]
        else:
            content = str(result)

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
        except Exception as exc:  # pragma: no cover - wrapped in higher level tests
            yield StreamEvent(
                event_type="error",
                data={"message": "OpenAI streaming failure", "details": str(exc)},
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


# Register default prefixes for OpenAI models.
_registry = get_model_registry()
_registry.register_model_prefix("gpt-", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("o1", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("openai/", lambda name: OpenAIChatModel(name))


__all__ = ["OpenAIChatModel"]

