from __future__ import annotations

import os

from llm.core.providers.base import BaseLangChainChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception as exc:
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None


class GeminiChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatGoogleGenerativeAI."""

    _API_MODEL_PREFIX = "gemini/"
    _provider_label = "Gemini"

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
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX) :]
        # Let the underlying library pick up GOOGLE_API_KEY / GEMINI_API_KEY from env.
        self._client = ChatGoogleGenerativeAI(model=api_model)


_registry = get_model_registry()
_registry.register_model_prefix("gemini-", lambda name: GeminiChatModel(name))
_registry.register_model_prefix("gemini/", lambda name: GeminiChatModel(name))


__all__ = ["GeminiChatModel"]
