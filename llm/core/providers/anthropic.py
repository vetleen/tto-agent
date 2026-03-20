from __future__ import annotations

import logging

from llm.core.model_factory import create_variant_client
from llm.core.providers.base import BaseLangChainChatModel
from llm.types.requests import ChatRequest

logger = logging.getLogger(__name__)

_ANTHROPIC_THINKING = {
    "low": {"budget": 4_096, "max_tokens": 16_384},
    "medium": {"budget": 10_000, "max_tokens": 16_384},
    "high": {"budget": 32_000, "max_tokens": 40_000},
}


class AnthropicChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatAnthropic."""

    _API_MODEL_PREFIX = "anthropic/"
    _provider_label = "Anthropic"
    _provider_id = "anthropic"

    def __init__(self, model_name: str, client: object) -> None:
        super().__init__(model_name, client)
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX):]
        self._api_model = api_model

    # -- Thinking / extended-thinking support --

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        level = request.params.get("thinking_level", "off")
        if level != "off" and level in _ANTHROPIC_THINKING:
            cfg = _ANTHROPIC_THINKING[level]
            try:
                client = create_variant_client(
                    self._api_model,
                    provider="anthropic",
                    thinking={"type": "enabled", "budget_tokens": cfg["budget"]},
                    max_tokens=cfg["max_tokens"],
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


__all__ = ["AnthropicChatModel"]
