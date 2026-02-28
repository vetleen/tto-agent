from __future__ import annotations

import os

from llm.core.providers.base import BaseLangChainChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_anthropic import ChatAnthropic
except Exception as exc:
    ChatAnthropic = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


class AnthropicChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatAnthropic."""

    _provider_label = "Anthropic"

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


_registry = get_model_registry()
_registry.register_model_prefix("claude-", lambda name: AnthropicChatModel(name))
_registry.register_model_prefix("anthropic/", lambda name: AnthropicChatModel(name))


__all__ = ["AnthropicChatModel"]
