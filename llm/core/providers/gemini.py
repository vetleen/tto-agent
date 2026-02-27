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
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as exc:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


class GeminiChatModel(ChatModel):
    """ChatModel backed by LangChain's ChatGoogleGenerativeAI."""

    def __init__(self, model_name: str) -> None:
        if _import_error is not None or ChatGoogleGenerativeAI is None:
            raise LLMProviderError(
                "langchain-google-genai is not installed or failed to import. "
                "Install with `pip install langchain-google-genai`."
            ) from _import_error

        # New Google GenAI client expects GOOGLE_API_KEY by default.
        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            raise LLMConfigurationError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set to use GeminiChatModel."
            )

        self.name = model_name
        # Let the underlying library pick up GOOGLE_API_KEY / GEMINI_API_KEY from env.
        self._client = ChatGoogleGenerativeAI(model=model_name)

    def generate(self, request: ChatRequest) -> ChatResponse:
        lc_messages = to_langchain_messages(request.messages)
        client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        try:
            result = client.invoke(lc_messages)
        except Exception as exc:  # pragma: no cover
            raise LLMProviderError(f"Gemini generate failed for model={self.name}") from exc

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
            # If true token streaming is available, this will yield chunks.
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
        except Exception as exc:  # pragma: no cover
            yield StreamEvent(
                event_type="error",
                data={"message": "Gemini streaming failure", "details": str(exc)},
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
_registry.register_model_prefix("gemini-", lambda name: GeminiChatModel(name))
_registry.register_model_prefix("gemini/", lambda name: GeminiChatModel(name))


__all__ = ["GeminiChatModel"]

