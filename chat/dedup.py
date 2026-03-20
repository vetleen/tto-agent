"""Deduplicate tool results to reduce token consumption.

When the LLM calls both search_documents and read_document for the same
document, the same chunk content can appear multiple times.  This module
provides a stateless pass that redacts duplicate content from *older* tool
results (keeping the newest copy intact) and redacts descriptions already
present in the dynamic context.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llm.types.messages import Message

logger = logging.getLogger(__name__)

# Placeholder text inserted in place of redacted content
_CONTENT_PLACEHOLDER = "[Content already provided in a later tool result]"
_DESC_PLACEHOLDER = "[See document list in context]"


@dataclass
class _ToolResultInfo:
    """Parsed metadata about a single tool-result message."""

    msg_index: int
    tool_name: str
    content: str
    # {doc_index: set(chunk_indices)} covered by this result
    coverage: dict[int, set[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def deduplicate_tool_results(
    messages: list[Message],
    dynamic_context: str = "",
) -> list[Message]:
    """Return a new message list with duplicate tool-result content redacted.

    - Scans ``role="tool"`` messages for ``search_documents`` /
      ``read_document`` results.
    - Tracks chunk coverage per ``doc_index``.
    - If ALL chunks in an older result are covered by newer results for the
      same doc_index, replaces chunk content with a short placeholder.
    - Descriptions already present in ``dynamic_context`` are replaced with a
      placeholder.
    - Returns new ``Message`` objects (via ``model_copy``) only for modified
      messages; originals are never mutated.
    """
    tool_infos = _identify_tool_messages(messages)
    if not tool_infos:
        return messages

    # Parse coverage for each tool result
    for info in tool_infos:
        if info.tool_name == "search_documents":
            info.coverage = _parse_search_coverage(info.content)
        elif info.tool_name == "read_document":
            info.coverage = _parse_read_coverage(info.content)

    # Build "seen" coverage scanning newest-first
    seen: dict[int, set[int]] = {}  # doc_index -> set(chunk_indices)
    # Track which infos need chunk redaction and which doc_indices
    chunk_redact: dict[int, set[int]] = {}  # msg_index -> set(doc_indices)

    # Sort by message index descending (newest first)
    sorted_infos = sorted(tool_infos, key=lambda x: x.msg_index, reverse=True)

    for info in sorted_infos:
        doc_indices_to_redact: set[int] = set()
        for doc_idx, chunks in info.coverage.items():
            if not chunks:
                continue
            if doc_idx in seen and chunks.issubset(seen[doc_idx]):
                # All chunks already provided by a newer result
                doc_indices_to_redact.add(doc_idx)
            # Merge into seen (whether or not we're redacting this one)
            seen.setdefault(doc_idx, set()).update(chunks)

        if doc_indices_to_redact:
            chunk_redact[info.msg_index] = doc_indices_to_redact

    # Determine description redaction from dynamic context
    context_doc_indices = _extract_context_doc_indices(dynamic_context)

    # Apply redactions
    result = list(messages)  # shallow copy of list
    for info in tool_infos:
        chunk_docs = chunk_redact.get(info.msg_index, set())
        desc_docs = context_doc_indices if info.tool_name == "search_documents" else set()

        if not chunk_docs and not desc_docs:
            continue

        if info.tool_name == "search_documents":
            new_content = _redact_search_content(info.content, chunk_docs, desc_docs)
        elif info.tool_name == "read_document":
            new_content = _redact_read_content(info.content, chunk_docs)
        else:
            continue

        if new_content != info.content:
            result[info.msg_index] = messages[info.msg_index].model_copy(
                update={"content": new_content}
            )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identify_tool_messages(messages: list[Message]) -> list[_ToolResultInfo]:
    """Find tool-result messages for search_documents / read_document.

    Correlates ``role="tool"`` messages to their tool names by matching
    ``tool_call_id`` against preceding assistant messages' ``tool_calls[].id``.
    """
    # Build lookup: tool_call_id -> tool_name
    call_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id_to_name[tc.id] = tc.name

    target_tools = {"search_documents", "read_document"}
    infos: list[_ToolResultInfo] = []

    for idx, msg in enumerate(messages):
        if msg.role != "tool" or not msg.tool_call_id:
            continue
        if not isinstance(msg.content, str):
            continue
        tool_name = call_id_to_name.get(msg.tool_call_id, "")
        if tool_name in target_tools:
            infos.append(_ToolResultInfo(
                msg_index=idx,
                tool_name=tool_name,
                content=msg.content,
            ))

    return infos


def _parse_search_coverage(content: str) -> dict[int, set[int]]:
    """Parse search_documents markdown output to extract chunk coverage.

    Returns ``{doc_index: set(chunk_indices)}`` for each result block.
    """
    coverage: dict[int, set[int]] = {}

    # Split into result blocks by "## N." headers
    blocks = re.split(r"(?=^## \d+\.)", content, flags=re.MULTILINE)

    for block in blocks:
        # Extract doc_index from [doc #N]
        doc_match = re.search(r"\[doc #(\d+)\]", block)
        if not doc_match:
            continue
        doc_index = int(doc_match.group(1))

        # Extract chunk range from "Chunk #N of M" or "Chunks #N–#M of T"
        chunk_match = re.search(
            r"Chunks?\s+#(\d+)(?:\u2013#(\d+))?\s+of\s+(\d+)", block
        )
        if chunk_match:
            start = int(chunk_match.group(1))
            end = int(chunk_match.group(2)) if chunk_match.group(2) else start
            chunks = set(range(start, end + 1))
        else:
            # No chunk label — can't determine coverage
            continue

        coverage.setdefault(doc_index, set()).update(chunks)

    return coverage


def _parse_read_coverage(content: str) -> dict[int, set[int]]:
    """Parse read_document JSON output to extract chunk coverage.

    Returns ``{doc_index: set(chunk_indices)}`` for each document entry.
    """
    coverage: dict[int, set[int]] = {}

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return coverage

    documents = data.get("documents", [])
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        doc_index = doc.get("doc_index")
        if doc_index is None:
            continue
        if "error" in doc and "content" not in doc:
            continue

        total_chunks = doc.get("total_chunks", 0)
        chunk_range = doc.get("chunk_range")

        if chunk_range and isinstance(chunk_range, str) and "-" in chunk_range:
            parts = chunk_range.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
                chunks = set(range(start, end + 1))
            except (ValueError, IndexError):
                chunks = set(range(total_chunks)) if total_chunks else set()
        else:
            # Full document read — all chunks from 0 to total_chunks-1
            chunks = set(range(total_chunks)) if total_chunks else set()

        if chunks:
            coverage.setdefault(doc_index, set()).update(chunks)

    return coverage


def _redact_search_content(
    content: str,
    chunk_redact_doc_indices: set[int],
    desc_redact_doc_indices: set[int],
) -> str:
    """Redact chunk content and/or descriptions from search_documents output.

    Keeps all metadata lines (Document, Type, Description header, Data room,
    Section, chunk label) but replaces the actual chunk text with a placeholder.
    """
    if not chunk_redact_doc_indices and not desc_redact_doc_indices:
        return content

    # Split into blocks by ## headers, preserving the header/preamble/postamble
    parts = re.split(r"(^## \d+\.)", content, flags=re.MULTILINE)
    # parts = [preamble, "## 1.", body1, "## 2.", body2, ...]

    result_parts: list[str] = []
    i = 0

    # Preamble (before first ## block)
    if parts and not parts[0].startswith("## "):
        result_parts.append(parts[0])
        i = 1

    while i < len(parts):
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        full_block = header + body

        doc_match = re.search(r"\[doc #(\d+)\]", full_block)
        if doc_match:
            doc_index = int(doc_match.group(1))

            # Redact description if needed
            if doc_index in desc_redact_doc_indices:
                body = re.sub(
                    r"(\*\*Description:\*\* )(.+)",
                    r"\g<1>" + _DESC_PLACEHOLDER,
                    body,
                )

            # Redact chunk content if needed
            if doc_index in chunk_redact_doc_indices:
                body = _redact_search_block_content(body)

        result_parts.append(header + body)
        i += 2

    return "".join(result_parts)


def _redact_search_block_content(body: str) -> str:
    """Replace chunk text in a single search result block with a placeholder.

    Keeps metadata lines (those starting with **) and the chunk label line,
    replacing everything after the chunk label with the placeholder.
    """
    lines = body.split("\n")
    new_lines: list[str] = []
    found_chunk_label = False
    content_replaced = False

    for line in lines:
        # Check if this is the chunk label line (e.g. "**Chunks #4–#5 of 10:**")
        if re.match(r"\*\*Chunks?\s+#\d+", line):
            new_lines.append(line)
            found_chunk_label = True
            content_replaced = False
            continue

        if found_chunk_label and not content_replaced:
            # This is where chunk content starts — replace it
            new_lines.append(_CONTENT_PLACEHOLDER)
            content_replaced = True
            # Skip remaining content lines until next metadata or end
            continue

        if content_replaced:
            # Skip content lines — but keep trailing separators and metadata
            if line.startswith("**") or line.startswith("---") or line.startswith("## "):
                new_lines.append(line)
                found_chunk_label = False
                content_replaced = False
            # else: skip (it's part of the redacted content)
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


def _redact_read_content(content: str, doc_indices_to_redact: set[int]) -> str:
    """Replace content field in read_document JSON for matching doc_indices."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content

    documents = data.get("documents")
    if not documents:
        return content

    modified = False
    new_documents = []
    for doc in documents:
        if isinstance(doc, dict) and doc.get("doc_index") in doc_indices_to_redact and "content" in doc:
            doc = dict(doc)  # shallow copy
            doc["content"] = _CONTENT_PLACEHOLDER
            modified = True
        new_documents.append(doc)

    if not modified:
        return content

    data = dict(data)
    data["documents"] = new_documents
    return json.dumps(data)


def _extract_context_doc_indices(dynamic_context: str) -> set[int]:
    """Extract doc_indices that have descriptions in the dynamic context.

    Parses the ``# Retrieved Documents`` section, looking for lines like:
    ``1. [1] "filename.pdf" (type) (~1,234 tokens) — description``
    """
    if not dynamic_context:
        return set()

    doc_indices: set[int] = set()

    # Match lines like: N. [N] "filename" ... — description
    # The description is after the " — " separator
    for match in re.finditer(
        r"^\d+\.\s+\[(\d+)\]\s+\"[^\"]+\".*?\s+\u2014\s+\S",
        dynamic_context,
        re.MULTILINE,
    ):
        doc_indices.add(int(match.group(1)))

    return doc_indices
