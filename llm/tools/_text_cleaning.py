"""Shared text-cleaning utilities for web content tools.

Used by web_fetch and brave_search to normalize text extracted from
external sources before it enters the LLM context.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and invisible Unicode characters that can be used
# to hide text from human readers while remaining in extracted text.
_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff]"
)

# Three or more consecutive newlines → collapse to two.
_EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize extracted web text.

    * Strips zero-width / invisible Unicode characters.
    * Applies NFC normalization.
    * Collapses excessive blank lines (3+ newlines → 2).
    * Strips leading/trailing whitespace.
    """
    if not text:
        return text
    text = _ZERO_WIDTH_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    text = _EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()
