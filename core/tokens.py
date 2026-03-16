"""Shared token-counting utility."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
        logger.warning("tiktoken count failed: %s", e)
        # Fallback token estimate to keep chunking and limits functional even
        # when tiktoken cannot download/load encoding data.
        # Use both word- and char-based heuristics to avoid severe
        # under-counting for long strings with little/no whitespace.
        word_estimate = len(text.split())
        char_estimate = (len(text) + 3) // 4
        return max(1, word_estimate, char_estimate)
