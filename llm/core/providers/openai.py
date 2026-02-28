from __future__ import annotations

import os

from llm.core.providers.base import BaseLangChainChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_openai import ChatOpenAI
except Exception as exc:  # ImportError or other environment issues
    # Defer failure until provider is actually constructed
    ChatOpenAI = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


class OpenAIChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatOpenAI."""

    _provider_label = "OpenAI"

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


# Register default prefixes for OpenAI models.
_registry = get_model_registry()
_registry.register_model_prefix("gpt-", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("o1", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("openai/", lambda name: OpenAIChatModel(name))


__all__ = ["OpenAIChatModel"]
