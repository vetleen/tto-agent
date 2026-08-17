from __future__ import annotations

from llm.core.model_factory import create_variant_client
from llm.core.providers.base import BaseLangChainChatModel
from llm.model_registry import get_model_info
from llm.types.requests import ChatRequest

# Fallback prefixes for reasoning support of models NOT in the registry
# (registered models report ModelInfo.supports_thinking, preferred below).
_REASONING_PREFIXES = (
    "o1", "o3", "o4", "gpt-5.6", "gpt-5.5", "gpt-5.4", "gpt-5.2-pro",
)


class OpenAIChatModel(BaseLangChainChatModel):
    """ChatModel backed by LangChain's ChatOpenAI."""

    # Prefix used in LLM_ALLOWED_MODELS; strip before sending to API.
    _API_MODEL_PREFIX = "openai/"
    _provider_label = "OpenAI"
    _provider_id = "openai"

    def __init__(self, model_name: str, client: object) -> None:
        super().__init__(model_name, client)
        # API expects model id without provider prefix (e.g. gpt-5-mini, not openai/gpt-5-mini).
        api_model = model_name
        if model_name.startswith(self._API_MODEL_PREFIX):
            api_model = model_name[len(self._API_MODEL_PREFIX):]
        self._api_model = api_model

    # -- Reasoning support --

    def _supports_reasoning(self) -> bool:
        info = get_model_info(self.name)
        if info is not None:
            return info.supports_thinking
        return any(self._api_model.lower().startswith(p) for p in _REASONING_PREFIXES)

    def _get_streaming_client(self, request: ChatRequest):
        client = self._client
        level = request.params.get("thinking_level")
        if level is not None and self._supports_reasoning():
            reasoning = {"effort": level}
            if level != "none":
                reasoning["summary"] = "auto"
            client = create_variant_client(
                self._api_model,
                provider="openai",
                reasoning=reasoning,
            )
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

    @staticmethod
    def _extract_reasoning(additional: dict) -> str:
        """Pull reasoning-summary text out of a chunk's additional_kwargs.

        Chat Completions surfaces it as a plain string under ``reasoning_content``;
        the Responses API (gpt-5.x) uses ``reasoning``, which is a summary string
        or a ``{"summary": [{"type": "summary_text", "text": ...}]}`` dict. Reading
        only ``reasoning_content`` left the thinking UI permanently empty for
        Responses-API models. Best-effort: an unrecognized shape yields "" (the
        prior no-thinking behavior), never an error.
        """
        rc = additional.get("reasoning_content")
        if isinstance(rc, str) and rc:
            return rc
        reasoning = additional.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning
        if isinstance(reasoning, dict):
            summary = reasoning.get("summary")
            if isinstance(summary, list):
                return "".join(
                    part.get("text", "") for part in summary
                    if isinstance(part, dict)
                )
            if isinstance(summary, str):
                return summary
        return ""

    @staticmethod
    def _extract_reasoning_blocks(content) -> str:
        """Extract Responses API summary deltas from normalized content blocks."""
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "reasoning":
                continue
            reasoning = block.get("reasoning")
            if isinstance(reasoning, str):
                parts.append(reasoning)
            summary = block.get("summary")
            if isinstance(summary, list):
                parts.extend(
                    item.get("text", "") for item in summary
                    if isinstance(item, dict)
                )
            elif isinstance(summary, str):
                parts.append(summary)
        return "".join(parts)

    def _extract_replay_metadata(self, lc_message) -> dict:
        """Preserve GPT-5.6 encrypted reasoning items when Responses uses store=False."""
        content = getattr(lc_message, "content", None)
        if not isinstance(content, list):
            return {}
        blocks = [
            dict(block) for block in content
            if isinstance(block, dict)
            and block.get("type") in ("reasoning", "text", "output_text")
        ]
        if any(block.get("type") == "reasoning" for block in blocks):
            return {"content_blocks": blocks}
        return {}

    def _parse_chunk(self, chunk) -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        additional = getattr(chunk, "additional_kwargs", {}) or {}
        content = getattr(chunk, "content", None)
        reasoning = self._extract_reasoning(additional) or self._extract_reasoning_blocks(content)
        if reasoning:
            events.append(("thinking", {"text": reasoning}))
        # Regular text content
        text = self._extract_text(content)
        if text:
            events.append(("token", {"text": text}))
        return events


__all__ = ["OpenAIChatModel"]
