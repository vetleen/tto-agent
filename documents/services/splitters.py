"""
Structure-aware text splitting for parent-child chunking.

Detects document structure (markdown, slides, plain text) and splits into
hierarchical parent/child chunks that respect structural boundaries.
Reusable module decoupled from Django.
"""
from __future__ import annotations

import re
from typing import Any

from core.tokens import count_tokens as _count_tokens


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def detect_structure(text: str) -> str:
    """Detect document structure type: "markdown", "slides", or "plain"."""
    trimmed = text.strip()
    # Slides: form feeds (from PPTX), or 3+ occurrences of --- on its own line
    if "\f" in trimmed:
        return "slides"
    slide_delimiters = re.findall(r"(?m)^---\s*$", trimmed)
    if len(slide_delimiters) >= 2:
        return "slides"
    # Markdown: starts with # or has ## headers
    if trimmed.startswith("#") or "\n## " in trimmed or "\n# " in trimmed:
        return "markdown"
    return "plain"


# ---------------------------------------------------------------------------
# Structural splitting into units
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"(?m)^(#{1,3})\s+(.+)$")
_SLIDE_DELIM_RE = re.compile(r"(?:\f|(?:^---\s*$))", re.MULTILINE)


def structural_split(text: str, structure_type: str) -> list[dict[str, Any]]:
    """Split text into structural units based on detected type.

    Returns list of dicts: {"text", "heading", "unit_type"}.
    """
    if structure_type == "markdown":
        return _split_markdown(text)
    if structure_type == "slides":
        return _split_slides(text)
    return _split_plain(text)


def _split_markdown(text: str) -> list[dict[str, Any]]:
    """Split markdown on H1/H2/H3 headings. Each section = one unit."""
    units: list[dict[str, Any]] = []
    # Find all heading positions
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        # No headings found — treat as single section
        if text.strip():
            return [{"text": text.strip(), "heading": None, "unit_type": "section"}]
        return []

    # Content before first heading
    pre = text[: matches[0].start()].strip()
    if pre:
        units.append({"text": pre, "heading": None, "unit_type": "section"})

    for i, m in enumerate(matches):
        heading_level = len(m.group(1))
        heading_text = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            units.append({
                "text": section_text,
                "heading": heading_text,
                "unit_type": "section",
                "heading_level": heading_level,
            })
    return units


def _split_slides(text: str) -> list[dict[str, Any]]:
    """Split on slide boundaries (---, form feeds)."""
    parts = _SLIDE_DELIM_RE.split(text)
    units = []
    for part in parts:
        part = part.strip()
        if part:
            # Try to extract a heading from the first line
            lines = part.split("\n", 1)
            heading = None
            first_line = lines[0].strip()
            heading_match = re.match(r"^#{1,3}\s+(.+)$", first_line)
            if heading_match:
                heading = heading_match.group(1).strip()
            elif len(first_line) < 100 and not first_line.endswith("."):
                heading = first_line
            units.append({"text": part, "heading": heading, "unit_type": "slide"})
    return units


def _split_plain(text: str) -> list[dict[str, Any]]:
    """Split plain text on paragraph boundaries (double newlines)."""
    paragraphs = re.split(r"\n\s*\n", text)
    units = []
    for para in paragraphs:
        para = para.strip()
        if para:
            units.append({"text": para, "heading": None, "unit_type": "paragraph"})
    return units


# ---------------------------------------------------------------------------
# Sentence-aware splitting for oversized structural units
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _force_split_unit(text: str, max_tokens: int) -> list[str]:
    """Split an oversized unit into pieces respecting sentence boundaries."""
    sentences = _split_sentences(text)
    if not sentences:
        # No sentence boundaries — split by words
        words = text.split()
        pieces = []
        current: list[str] = []
        current_tokens = 0
        for word in words:
            word_tokens = _count_tokens(word)
            if current and current_tokens + word_tokens > max_tokens:
                pieces.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens
        if current:
            pieces.append(" ".join(current))
        return pieces

    pieces = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sent_tokens = _count_tokens(sentence)
        if current and current_tokens + sent_tokens > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
            current_tokens = sent_tokens
        else:
            current.append(sentence)
            current_tokens += sent_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


# ---------------------------------------------------------------------------
# Parent-child splitting
# ---------------------------------------------------------------------------

_MERGE_MIN_TOKENS = 100  # Merge children smaller than this


def parent_child_split(
    text: str,
    structure_type: str | None = None,
    child_target_tokens: int = 300,
    child_overlap_pct: float = 0.20,
    max_child_tokens: int = 600,
) -> list[dict[str, Any]]:
    """Two-stage split: structural parents -> child chunks within each parent.

    Returns list of parent dicts, each with a "children" list.
    """
    if structure_type is None:
        structure_type = detect_structure(text)

    units = structural_split(text, structure_type)
    if not units:
        return []

    # Stage 1: Group units into parents
    parents = _group_into_parents(units, structure_type)

    # Stage 2: Split each parent into children
    overlap_tokens = int(child_target_tokens * child_overlap_pct)
    result = []
    offset = 0
    for parent in parents:
        parent_text = parent["text"]
        parent_tokens = _count_tokens(parent_text)
        children = _split_parent_into_children(
            parent, structure_type, child_target_tokens, max_child_tokens, overlap_tokens,
        )
        result.append({
            "text": parent_text,
            "heading": parent.get("heading"),
            "token_count": parent_tokens,
            "children": children,
            "source_offset_start": offset,
            "source_offset_end": offset + len(parent_text),
        })
        offset += len(parent_text)

    return result


def _group_into_parents(
    units: list[dict[str, Any]],
    structure_type: str,
) -> list[dict[str, Any]]:
    """Group structural units into parent segments."""
    if structure_type == "slides":
        # Each slide is its own parent
        return list(units)

    if structure_type == "markdown":
        return _group_markdown_parents(units)

    # Plain text: group consecutive paragraphs into ~1500 token parents
    return _group_plain_parents(units, soft_target=1500)


def _group_markdown_parents(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group markdown sections: H1/H2 = parent boundary, H3 stays within parent."""
    parents: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_heading: str | None = None

    for unit in units:
        level = unit.get("heading_level", 99)
        # H1 or H2 starts a new parent
        if level <= 2 and current_texts:
            parents.append({
                "text": "\n\n".join(current_texts),
                "heading": current_heading,
                "unit_type": "section",
            })
            current_texts = []
            current_heading = None

        if level <= 2:
            current_heading = unit.get("heading")
        elif current_heading is None:
            current_heading = unit.get("heading")
        current_texts.append(unit["text"])

    if current_texts:
        parents.append({
            "text": "\n\n".join(current_texts),
            "heading": current_heading,
            "unit_type": "section",
        })

    return parents


def _group_plain_parents(
    units: list[dict[str, Any]],
    soft_target: int = 1500,
) -> list[dict[str, Any]]:
    """Group plain-text paragraphs into parent segments around soft_target tokens."""
    parents: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _count_tokens(unit["text"])
        # If adding this unit would exceed soft target and we already have content,
        # start a new parent
        if current_texts and current_tokens + unit_tokens > soft_target:
            parents.append({
                "text": "\n\n".join(current_texts),
                "heading": None,
                "unit_type": "section",
            })
            current_texts = []
            current_tokens = 0
        current_texts.append(unit["text"])
        current_tokens += unit_tokens

    if current_texts:
        parents.append({
            "text": "\n\n".join(current_texts),
            "heading": None,
            "unit_type": "section",
        })

    return parents


def _split_parent_into_children(
    parent: dict[str, Any],
    structure_type: str,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    """Split a parent into child chunks respecting structural boundaries."""
    parent_text = parent["text"]

    # Slides: 1 slide = 1 child (unless absurdly large)
    if structure_type == "slides" or parent.get("unit_type") == "slide":
        slide_tokens = _count_tokens(parent_text)
        if slide_tokens <= max_tokens:
            return [{"text": parent_text, "token_count": slide_tokens, "child_index": 0}]
        # Force-split absurdly large slide
        pieces = _force_split_unit(parent_text, max_tokens)
        return [
            {"text": p, "token_count": _count_tokens(p), "child_index": i}
            for i, p in enumerate(pieces)
        ]

    # Split parent into sub-units (paragraphs within the section)
    sub_units = _split_plain(parent_text) if structure_type != "markdown" else _split_plain(parent_text)

    if not sub_units:
        tokens = _count_tokens(parent_text)
        if tokens > 0:
            return [{"text": parent_text, "token_count": tokens, "child_index": 0}]
        return []

    # Build children by accumulating sub-units
    children: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_tokens = 0

    for su in sub_units:
        su_text = su["text"]
        su_tokens = _count_tokens(su_text)

        # If this single unit exceeds max, force-split it
        if su_tokens > max_tokens:
            # Flush current accumulator first
            if current_texts:
                children.append(_make_child("\n\n".join(current_texts), len(children)))
                current_texts = []
                current_tokens = 0
            pieces = _force_split_unit(su_text, max_tokens)
            for piece in pieces:
                children.append(_make_child(piece, len(children)))
            continue

        # If adding this would exceed target and we have content, flush
        if current_texts and current_tokens + su_tokens > target_tokens:
            children.append(_make_child("\n\n".join(current_texts), len(children)))
            current_texts = []
            current_tokens = 0

        current_texts.append(su_text)
        current_tokens += su_tokens

    if current_texts:
        children.append(_make_child("\n\n".join(current_texts), len(children)))

    # Merge tiny children (< _MERGE_MIN_TOKENS) into adjacent
    children = _merge_tiny_children(children)

    # Apply overlap between adjacent children
    if overlap_tokens > 0 and len(children) > 1:
        children = _apply_overlap(children, overlap_tokens)

    return children


def _make_child(text: str, index: int) -> dict[str, Any]:
    return {"text": text.strip(), "token_count": _count_tokens(text), "child_index": index}


def _merge_tiny_children(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge children smaller than _MERGE_MIN_TOKENS into adjacent ones."""
    if len(children) <= 1:
        return children
    result = list(children)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            if result[i]["token_count"] < _MERGE_MIN_TOKENS and len(result) > 1:
                # Merge into smallest adjacent
                if i == 0:
                    merge_idx = 1
                elif i == len(result) - 1:
                    merge_idx = i - 1
                else:
                    merge_idx = i - 1 if result[i - 1]["token_count"] <= result[i + 1]["token_count"] else i + 1
                # Combine
                if merge_idx > i:
                    new_text = result[i]["text"] + "\n\n" + result[merge_idx]["text"]
                else:
                    new_text = result[merge_idx]["text"] + "\n\n" + result[i]["text"]
                result[merge_idx] = _make_child(new_text, result[merge_idx]["child_index"])
                result.pop(i)
                changed = True
                break
            i += 1
    # Re-index
    for i, child in enumerate(result):
        child["child_index"] = i
    return result


def _apply_overlap(children: list[dict[str, Any]], overlap_tokens: int) -> list[dict[str, Any]]:
    """Add overlap from the end of previous child to the start of next child."""
    if len(children) <= 1:
        return children
    result = [children[0]]
    for i in range(1, len(children)):
        prev_text = children[i - 1]["text"]
        # Get trailing words from previous child as overlap
        words = prev_text.split()
        overlap_words: list[str] = []
        tokens_so_far = 0
        for w in reversed(words):
            w_tokens = _count_tokens(w)
            if tokens_so_far + w_tokens > overlap_tokens:
                break
            overlap_words.insert(0, w)
            tokens_so_far += w_tokens
        if overlap_words:
            overlap_text = " ".join(overlap_words)
            new_text = overlap_text + " " + children[i]["text"]
            child = _make_child(new_text, i)
        else:
            child = dict(children[i])
            child["child_index"] = i
        result.append(child)
    return result
