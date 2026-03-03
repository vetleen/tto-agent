"""Shared token-counting utility."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in *text* using tiktoken.

    Falls back to a heuristic estimate when tiktoken is unavailable.
    """
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
