from __future__ import annotations

from llm.core.providers.base import BaseLangChainChatModel


class GeminiChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatGoogleGenerativeAI."""

    _API_MODEL_PREFIX = "gemini/"
    _provider_label = "Gemini"
    _provider_id = "google_genai"


__all__ = ["GeminiChatModel"]
