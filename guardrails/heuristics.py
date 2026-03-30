"""Layer 0: Regex/keyword heuristic scanner for guardrail pre-filtering.

Zero latency, zero cost. Catches high-confidence injection patterns before
they reach the LLM classifier. Each pattern set has an associated confidence
weight; aggregated hits produce a combined score.
"""

from __future__ import annotations

import base64
import logging
import re

from guardrails.schemas import HeuristicResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern definitions — compiled once at module level
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE), 0.9),
    (re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.IGNORECASE), 0.9),
    (re.compile(r"new\s+instructions?\s*:", re.IGNORECASE), 0.85),
    (re.compile(r"system\s+prompt\s*:", re.IGNORECASE), 0.8),
    (re.compile(r"forget\s+(your|all|previous)\s+instructions", re.IGNORECASE), 0.9),
    (re.compile(r"override\s+(your|all|previous)\s+(instructions|rules|guidelines)", re.IGNORECASE), 0.85),
    (re.compile(r"you\s+are\s+now\s+(in\s+)?a\s+new\s+mode", re.IGNORECASE), 0.8),
    (re.compile(r"from\s+now\s+on\s*,?\s*(you\s+)?(will|must|should)\s+ignore", re.IGNORECASE), 0.85),
]

_JAILBREAK_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bDAN\s+mode\b", re.IGNORECASE), 0.9),
    (re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE), 0.7),
    (re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE), 0.9),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE), 0.6),
    (re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.IGNORECASE), 0.9),
    (re.compile(r"jailbreak", re.IGNORECASE), 0.85),
    (re.compile(r"bypass\s+(your\s+)?(safety|content|ethical)\s+(filters?|guidelines?|restrictions?)", re.IGNORECASE), 0.9),
    (re.compile(r"you\s+have\s+no\s+(ethical\s+)?guidelines", re.IGNORECASE), 0.85),
]

_DATA_EXTRACTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE), 0.9),
    (re.compile(r"show\s+me\s+your\s+(instructions|system\s+prompt|rules)", re.IGNORECASE), 0.85),
    (re.compile(r"what\s+are\s+your\s+(rules|instructions|guidelines)", re.IGNORECASE), 0.5),
    (re.compile(r"repeat\s+everything\s+above", re.IGNORECASE), 0.85),
    (re.compile(r"print\s+your\s+(prompt|instructions|system)", re.IGNORECASE), 0.85),
    (re.compile(r"output\s+(the|your)\s+(system\s+)?(prompt|instructions)", re.IGNORECASE), 0.85),
    (re.compile(r"tell\s+me\s+(the\s+)?system\s+prompt", re.IGNORECASE), 0.85),
]

_DELIMITER_INJECTION_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"<\|im_start\|>", re.IGNORECASE), 0.95),
    (re.compile(r"<\|im_end\|>", re.IGNORECASE), 0.95),
    (re.compile(r"\[INST\]", re.IGNORECASE), 0.9),
    (re.compile(r"<<SYS>>", re.IGNORECASE), 0.9),
    (re.compile(r"<\|system\|>", re.IGNORECASE), 0.95),
    (re.compile(r"<\|user\|>", re.IGNORECASE), 0.9),
    (re.compile(r"<\|assistant\|>", re.IGNORECASE), 0.9),
    (re.compile(r"```system", re.IGNORECASE), 0.7),
]

# Chinese gambling / lottery spam — extremely common in web-based prompt injection.
_WEB_SPAM_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"[\u4e00-\u9fff]{0,4}(赛车|彩票|彩神|娱乐平台|博彩|赌场|棋牌|时时彩|幸运飞艇)"), 0.85),
    (re.compile(r"(主管|代理|招商).{0,4}(注册|充值|返水|优惠|开户)"), 0.8),
]

# Tag mapping for each pattern set
_PATTERN_SETS: list[tuple[str, list[tuple[re.Pattern, float]]]] = [
    ("prompt_injection", _PROMPT_INJECTION_PATTERNS),
    ("jailbreak", _JAILBREAK_PATTERNS),
    ("data_extraction", _DATA_EXTRACTION_PATTERNS),
    ("delimiter_injection", _DELIMITER_INJECTION_PATTERNS),
    ("web_spam", _WEB_SPAM_PATTERNS),
]

# Zero-width and excessive unicode detection
_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff]"
)
_EXCESSIVE_ZERO_WIDTH_THRESHOLD = 5

# Base64 block detection (at least 20 chars of base64)
_BASE64_BLOCK_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

# Suspicious keywords to check in decoded base64 content
_BASE64_SUSPICIOUS_KEYWORDS = [
    "ignore", "instructions", "system prompt", "jailbreak",
    "disregard", "override", "bypass",
]


def _check_encoding_bypass(text: str) -> tuple[float, list[str]]:
    """Check for encoding bypass attempts (zero-width chars, suspicious base64).

    Returns (confidence, list_of_matched_descriptions).
    """
    matches: list[str] = []
    max_confidence = 0.0

    # Zero-width character abuse
    zero_width_count = len(_ZERO_WIDTH_RE.findall(text))
    if zero_width_count >= _EXCESSIVE_ZERO_WIDTH_THRESHOLD:
        matches.append(f"excessive_zero_width_chars ({zero_width_count})")
        max_confidence = max(max_confidence, 0.7)

    # Suspicious base64 content
    for b64_match in _BASE64_BLOCK_RE.finditer(text):
        try:
            decoded = base64.b64decode(b64_match.group() + "==", validate=False).decode(
                "utf-8", errors="ignore"
            ).lower()
            for keyword in _BASE64_SUSPICIOUS_KEYWORDS:
                if keyword in decoded:
                    matches.append(f"base64_contains_{keyword.replace(' ', '_')}")
                    max_confidence = max(max_confidence, 0.8)
                    break
        except Exception:
            continue

    return max_confidence, matches


def heuristic_scan(text: str) -> HeuristicResult:
    """Run all heuristic patterns against the input text.

    Returns a HeuristicResult with aggregated tags, confidence, and
    matched pattern descriptions. Returns quickly with a clean result
    for non-suspicious input.
    """
    if not text or not text.strip():
        return HeuristicResult()

    tags: set[str] = set()
    matched_patterns: list[str] = []
    max_confidence = 0.0

    # Check regex pattern sets
    for tag, patterns in _PATTERN_SETS:
        for pattern, confidence in patterns:
            if pattern.search(text):
                tags.add(tag)
                matched_patterns.append(pattern.pattern)
                max_confidence = max(max_confidence, confidence)

    # Check encoding bypass
    enc_confidence, enc_matches = _check_encoding_bypass(text)
    if enc_matches:
        tags.add("encoding_bypass")
        matched_patterns.extend(enc_matches)
        max_confidence = max(max_confidence, enc_confidence)

    if not tags:
        return HeuristicResult()

    return HeuristicResult(
        is_suspicious=True,
        tags=sorted(tags),
        confidence=max_confidence,
        matched_patterns=matched_patterns,
    )
