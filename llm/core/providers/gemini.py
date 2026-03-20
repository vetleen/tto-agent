from __future__ import annotations

import logging

from llm.core.model_factory import create_variant_client
from llm.core.providers.base import BaseLangChainChatModel
from llm.model_registry import get_model_info
from llm.types.requests import ChatRequest

logger = logging.getLogger(__name__)

_GEMINI_THINKING_BUDGETS = {
    "low": 1_024,
    "medium": 8_192,
    "high": 24_576,
}


class GeminiChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatGoogleGenerativeAI."""

    _API_MODEL_PREFIX = "gemini/"
    _provider_label = "Gemini"
    _provider_id = "google_genai"

    def __init__(self, model_name: str, client: object) -> None:
        super().__init__(model_name, client)
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX):]
        self._api_model = api_model

    def _supports_thinking(self) -> bool:
        info = get_model_info(self.name)
        if info is not None:
            return info.supports_thinking
        return False

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        level = request.params.get("thinking_level", "off")
        if level != "off" and self._supports_thinking():
            budget = _GEMINI_THINKING_BUDGETS.get(level, 8_192)
            try:
                client = create_variant_client(
                    self._api_model,
                    provider="google_genai",
                    thinking={"thinking_budget": budget},
                    include_thoughts=True,
                )
            except Exception:
                logger.warning(
                    "Failed to create thinking-enabled Gemini client; "
                    "falling back to standard client.",
                    exc_info=True,
                )
                client = self._client
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        return client

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        content = getattr(chunk, "content", None)
        # Gemini thinking: content is a list of typed parts
        if isinstance(content, list):
            results: list[tuple[str, dict]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                # Gemini marks thinking parts with thought=True
                if part.get("thought"):
                    text = part.get("text", "")
                    if text:
                        results.append(("thinking", {"text": text}))
                elif part.get("type") == "text" or "text" in part:
                    text = part.get("text", "")
                    if text and not part.get("thought"):
                        results.append(("token", {"text": text}))
            return results
        # Non-thinking response: fall back to base
        return super()._parse_chunk(chunk)


__all__ = ["GeminiChatModel"]
