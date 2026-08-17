from __future__ import annotations

from llm.core.model_factory import create_variant_client
from llm.core.providers.base import BaseLangChainChatModel
from llm.model_registry import get_model_info
from llm.types.requests import ChatRequest

# langchain-google-genai stashes each function call's thought signature here
# (keyed by tool_call id) on parse, and reads it back from an AIMessage's
# additional_kwargs on re-send. Gemini 3 rejects a follow-up whose function call
# is missing its signature (400 INVALID_ARGUMENT), so this map must survive our
# Message round-trip.
_GEMINI_FN_CALL_SIGNATURES_KEY = "__gemini_function_call_thought_signatures__"


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
        level = request.params.get("thinking_level")
        if level is not None and self._supports_thinking():
            client = create_variant_client(
                self._api_model,
                provider="google_genai",
                thinking_level=level,
                include_thoughts=True,
            )
        if request.tool_schemas:
            client = client.bind_tools(request.tool_schemas)
        return client

    def _extract_replay_metadata(self, lc_message) -> dict:
        """Carry Gemini function-call thought signatures through the tool loop.

        Our Message abstraction drops additional_kwargs, which is where
        langchain-google-genai keeps the signatures; preserve just that map so
        the follow-up request re-attaches each signature to its function call.
        """
        ak = getattr(lc_message, "additional_kwargs", None) or {}
        sig_map = ak.get(_GEMINI_FN_CALL_SIGNATURES_KEY)
        if sig_map:
            return {"additional_kwargs": {_GEMINI_FN_CALL_SIGNATURES_KEY: sig_map}}
        return {}

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
