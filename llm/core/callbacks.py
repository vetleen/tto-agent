"""LangChain callback for capturing the final prompt sent to the API.

PromptCaptureCallback hooks into on_chat_model_start, which fires after
bind_tools() is applied but before the HTTP request — capturing the exact
messages LangChain sees, including tool schemas.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised via mocks in unit tests
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
except Exception:
    BaseCallbackHandler = object  # type: ignore[assignment,misc]
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore[assignment]


def _truncate_base64_in_content(content):
    """Replace large base64 strings with placeholders in logged content."""
    if not isinstance(content, list):
        return content
    truncated = []
    for block in content:
        if isinstance(block, dict) and "base64" in block:
            truncated.append({**block, "base64": f"[{len(block['base64'])} chars]"})
        elif isinstance(block, dict) and block.get("type") == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:") and len(url) > 200:
                truncated.append({**block, "image_url": {"url": f"[data URI, {len(url)} chars]"}})
            else:
                truncated.append(block)
        else:
            truncated.append(block)
    return truncated


def _serialize_lc_message(msg: object) -> dict:
    """Serialize a LangChain BaseMessage to a plain dict."""
    if SystemMessage is not None and isinstance(msg, SystemMessage):
        return {"role": "system", "content": _truncate_base64_in_content(msg.content)}
    if HumanMessage is not None and isinstance(msg, HumanMessage):
        return {"role": "user", "content": _truncate_base64_in_content(msg.content)}
    if AIMessage is not None and isinstance(msg, AIMessage):
        d: dict = {"role": "assistant", "content": _truncate_base64_in_content(msg.content)}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", ""),
                    "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                    "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}),
                }
                for tc in tool_calls
            ]
        return d
    if ToolMessage is not None and isinstance(msg, ToolMessage):
        return {"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id}
    # Fallback for unknown message types
    return {"role": "unknown", "content": getattr(msg, "content", str(msg))}


class PromptCaptureCallback(BaseCallbackHandler):
    """Captures the serialized prompt from on_chat_model_start.

    After client.invoke() or the first chunk of client.stream() completes,
    self.captured_messages holds a list[dict] — the messages as LangChain
    sees them right before the API call.

    The caller (BaseLangChainChatModel) assembles the final raw_prompt dict
    by combining captured_messages with the tool_schemas from the request.
    """

    def __init__(self) -> None:
        self.captured_messages: list | None = None

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        """Fires synchronously before the API call, after bind_tools() is applied."""
        if messages:
            self.captured_messages = [_serialize_lc_message(m) for m in messages[0]]


__all__ = ["PromptCaptureCallback"]
