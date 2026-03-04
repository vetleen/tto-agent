from __future__ import annotations

import logging
import os

from llm.core.providers.base import BaseLangChainChatModel
from llm.core.registry import get_model_registry
from llm.service.errors import LLMConfigurationError, LLMProviderError
from llm.types.requests import ChatRequest

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_anthropic import ChatAnthropic
except Exception as exc:
    ChatAnthropic = None  # type: ignore[assignment]
    _import_error: Exception | None = exc
else:
    _import_error = None

logger = logging.getLogger(__name__)


class AnthropicChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatAnthropic."""

    _API_MODEL_PREFIX = "anthropic/"
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
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX) :]
        self._api_model = api_model
        self._client = ChatAnthropic(model=api_model)

    # -- Thinking / extended-thinking support --

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        if request.params.get("thinking"):
            budget = request.params.get("thinking_budget", 10_000)
            try:
                client = ChatAnthropic(
                    model=self._api_model,
                    thinking={"type": "enabled", "budget_tokens": budget},
                    max_tokens=16_384,
                )
            except Exception:
                logger.warning(
                    "Failed to create thinking-enabled Anthropic client; "
                    "falling back to standard client.",
                    exc_info=True,
                )
                client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        return client

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        content = getattr(chunk, "content", None)
        # Anthropic extended thinking: content is a list of typed blocks
        if isinstance(content, list):
            results: list[tuple[str, dict]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking":
                    text = block.get("thinking", "")
                    if text:
                        results.append(("thinking", {"text": text}))
                elif block_type == "text":
                    text = block.get("text", "")
                    if text:
                        results.append(("token", {"text": text}))
            return results
        # Non-thinking response: fall back to base
        return super()._parse_chunk(chunk)


_registry = get_model_registry()
_registry.register_model_prefix("claude-", lambda name: AnthropicChatModel(name))
_registry.register_model_prefix("anthropic/", lambda name: AnthropicChatModel(name))


__all__ = ["AnthropicChatModel"]
