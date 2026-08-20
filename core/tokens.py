"""Shared token-counting utility."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Warn-once flag: when tiktoken can't load an encoding (e.g. no network to
# fetch the BPE file), EVERY count falls back — one WARNING per call would
# flood logs/Sentry. First failure warns, the rest log at DEBUG.
_tiktoken_fallback_warned = False


def count_tokens(content, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in *content* using tiktoken.

    *content* may be a plain string or a list of multimodal content blocks
    (dicts with ``type`` keys such as ``text``, ``image``, ``image_url``).

    Falls back to a heuristic estimate when tiktoken is unavailable.
    """
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    total += _count_text_tokens(block.get("text", ""), encoding_name)
                elif block.get("type") in ("image", "image_url"):
                    total += 170  # conservative estimate per image
                else:
                    total += _count_text_tokens(str(block), encoding_name)
            else:
                total += _count_text_tokens(str(block), encoding_name)
        return total
    return _count_text_tokens(content, encoding_name)


def estimate_chat_request_tokens(
    messages: Iterable[Any],
    tools: Iterable[Any] | None = None,
    encoding_name: str = "cl100k_base",
) -> int:
    """Estimate the input-token footprint of an assembled chat request.

    The providers tokenize message envelopes and tool schemas differently, so
    this deliberately returns an estimate rather than a billing-grade count.
    Message content uses :func:`count_tokens`, including its conservative image
    estimate; roles, tool calls, and selected tool schemas are serialized into a
    stable JSON representation and counted as ordinary text.
    """
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        total += count_tokens(content, encoding_name)

        envelope = {
            "role": getattr(message, "role", ""),
            "name": getattr(message, "name", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            envelope["tool_calls"] = [
                call.model_dump() if hasattr(call, "model_dump") else call
                for call in tool_calls
            ]
        total += count_tokens(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, default=str),
            encoding_name,
        )
        # Small per-message allowance for provider-specific separators.
        total += 4

    for tool in tools or []:
        args_schema = getattr(tool, "args_schema", None)
        if hasattr(args_schema, "model_json_schema"):
            parameters = args_schema.model_json_schema()
        elif hasattr(args_schema, "schema"):
            parameters = args_schema.schema()
        else:
            parameters = {}
        schema = {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", "") or "",
            "parameters": parameters,
        }
        total += count_tokens(
            json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str),
            encoding_name,
        )

    return total


def _count_text_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in a plain text string."""
    text = text or ""
    if not text.strip():
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception as e:
        global _tiktoken_fallback_warned
        if not _tiktoken_fallback_warned:
            _tiktoken_fallback_warned = True
            logger.warning("tiktoken count failed: %s", e)
        else:
            logger.debug("tiktoken count failed: %s", e)
        # Fallback token estimate to keep chunking and limits functional even
        # when tiktoken cannot download/load encoding data.
        # Use both word- and char-based heuristics to avoid severe
        # under-counting for long strings with little/no whitespace.
        word_estimate = len(text.split())
        char_estimate = (len(text) + 3) // 4
        return max(1, word_estimate, char_estimate)
