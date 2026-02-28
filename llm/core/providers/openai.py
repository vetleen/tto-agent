from __future__ import annotations

import os
from typing import Iterator

from llm.core.interfaces import ChatModel
from llm.core.langchain_utils import to_langchain_messages, parse_tool_calls_from_ai_message
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_openai import ChatOpenAI
except Exception as exc:  # ImportError or other environment issues
    # Defer failure until provider is actually constructed
    ChatOpenAI = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


class OpenAIChatModel(ChatModel):
    """ChatModel backed by LangChain's ChatOpenAI."""

    # Prefix used in LLM_ALLOWED_MODELS; strip before sending to API.
    _API_MODEL_PREFIX = "openai/"

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
        # API expects model id without provider prefix (e.g. gpt-5-mini, not openai/gpt-5-mini).
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX) :]
        # Let ChatOpenAI read configuration from environment; enable streaming usage accounting.
        self._client = ChatOpenAI(model=api_model, stream_usage=True)

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        try:
            result = client.invoke(lc_messages)
        except Exception as exc:  # pragma: no cover - wrapped in higher level tests
            raise LLMProviderError(f"OpenAI generate failed for model={self.name}") from exc

        if hasattr(result, "content"):
            content = result.content  # type: ignore[assignment]
        else:
            content = str(result)

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

