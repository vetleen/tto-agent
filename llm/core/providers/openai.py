from __future__ import annotations

import logging
import os

from llm.core.providers.base import BaseLangChainChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError
from llm.types.requests import ChatRequest

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_openai import ChatOpenAI
except Exception as exc:  # ImportError or other environment issues
    # Defer failure until provider is actually constructed
    ChatOpenAI = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None

logger = logging.getLogger(__name__)

_REASONING_PREFIXES = ("o1", "o3", "o4")


class OpenAIChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatOpenAI."""

    # Prefix used in LLM_ALLOWED_MODELS; strip before sending to API.
    _API_MODEL_PREFIX = "openai/"
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
        # API expects model id without provider prefix (e.g. gpt-5-mini, not openai/gpt-5-mini).
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX) :]
        self._api_model = api_model
        # Let ChatOpenAI read configuration from environment; enable streaming usage accounting.
        self._client = ChatOpenAI(model=api_model, stream_usage=True)

    # -- Reasoning support for o-series models --

    def _is_reasoning_model(self) -> bool:
        return any(self._api_model.lower().startswith(p) for p in _REASONING_PREFIXES)

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        if request.params.get("thinking") and self._is_reasoning_model():
            try:
                client = ChatOpenAI(
                    model=self._api_model,
                    stream_usage=True,
                    model_kwargs={"reasoning_effort": "medium"},
                )
            except Exception:
                logger.warning(
                    "Failed to create reasoning-enabled OpenAI client; "
                    "falling back to standard client.",
                    exc_info=True,
                )
                client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        return client

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        # Check for reasoning content in additional_kwargs
        additional = getattr(chunk, "additional_kwargs", {}) or {}
        reasoning = additional.get("reasoning_content", "")
        if reasoning:
            events.append(("thinking", {"text": str(reasoning)}))
        # Regular text content
        text = getattr(chunk, "content", "") or ""
        if text:
            events.append(("token", {"text": str(text)}))
        return events


# Register default prefixes for OpenAI models.
_registry = get_model_registry()
_registry.register_model_prefix("gpt-", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("o1", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("openai/", lambda name: OpenAIChatModel(name))


__all__ = ["OpenAIChatModel"]
