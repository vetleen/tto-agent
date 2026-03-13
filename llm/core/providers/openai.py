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

# Models that require OpenAI's Responses API (LangChain may not recognise all of
# them yet via its built-in _model_prefers_responses_api check).
_RESPONSES_API_PREFIXES = ("gpt-5.4", "gpt-5.2-pro")


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

        use_responses = any(api_model.startswith(p) for p in _RESPONSES_API_PREFIXES)
        # Let ChatOpenAI read configuration from environment; enable streaming usage accounting.
        self._client = ChatOpenAI(
            model=api_model,
            stream_usage=True,
            **({"use_responses_api": True} if use_responses else {}),
        )

    # -- Reasoning support for o-series models --

    def _is_reasoning_model(self) -> bool:
        return any(self._api_model.lower().startswith(p) for p in _REASONING_PREFIXES)

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        if request.params.get("thinking") and self._is_reasoning_model():
            try:
                use_responses = any(
                    self._api_model.startswith(p) for p in _RESPONSES_API_PREFIXES
                )
                client = ChatOpenAI(
                    model=self._api_model,
                    stream_usage=True,
                    model_kwargs={"reasoning_effort": "medium"},
                    **({"use_responses_api": True} if use_responses else {}),
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

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from chunk content.

        The Responses API returns content as a list of dicts
        (e.g. ``[{'type': 'text', 'text': 'hello', 'index': 0}]``)
        while the Chat Completions API returns a plain string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content) if content else ""

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        # Check for reasoning content in additional_kwargs
        additional = getattr(chunk, "additional_kwargs", {}) or {}
        reasoning = additional.get("reasoning_content", "")
        if reasoning:
            events.append(("thinking", {"text": str(reasoning)}))
        # Regular text content
        text = self._extract_text(getattr(chunk, "content", None))
        if text:
            events.append(("token", {"text": text}))
        return events


# Register default prefixes for OpenAI models.
_registry = get_model_registry()
_registry.register_model_prefix("gpt-", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("o1", lambda name: OpenAIChatModel(name))
_registry.register_model_prefix("openai/", lambda name: OpenAIChatModel(name))


__all__ = ["OpenAIChatModel"]
