"""Deduplicate tool results to reduce token consumption.

When the LLM calls both document_search and document_read for the same
document, the same chunk content can appear multiple times. Two passes:

- ``deduplicate_tool_results`` (keep-newest) redacts duplicate content from
  *older* tool results and descriptions already present in the dynamic context.
  It edits earlier history, so the tool loop only runs it at edit points
  (``normalize_keep_newest``) and between turns.
- ``redact_new_duplicates`` (append-only) redacts duplicate content from the
  results a tool-loop round just appended, before they are ever sent, and never
  touches earlier messages — keeping the request prefix byte-stable (prompt
  cache, Anthropic preserved thinking) between edit points.

Each redaction stashes the original content in ``metadata["wf_original_content"]``
(in-memory only) so ``normalize_keep_newest`` can restore and re-decide.
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
_EARLIER_PLACEHOLDER = "[Content already provided in an earlier tool result above]"
_DESC_PLACEHOLDER = "[See document list in context]"
_PLACEHOLDERS = (_CONTENT_PLACEHOLDER, _EARLIER_PLACEHOLDER)
# Prefix of a pruned tool result (chat/tool_stub.py) — carries no content.
_STUB_PREFIX = "[Earlier result of "
# Message.metadata key holding a redacted result's original content.
ORIGINAL_CONTENT_KEY = "wf_original_content"


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

    - Scans ``role="tool"`` messages for ``document_search`` /
      ``document_read`` results.
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
        desc_docs = context_doc_indices if info.tool_name == "document_search" else set()

        if not chunk_docs and not desc_docs:
            continue

        new_content = _redact(info, chunk_docs, desc_docs, _CONTENT_PLACEHOLDER)
        if new_content != info.content:
            # Between turns (dynamic_context given) redactions are final; in the
            # tool loop they are stashed so the next edit point can re-decide.
            result[info.msg_index] = _with_redacted_content(
                messages[info.msg_index], new_content, stash=not dynamic_context,
            )

    return result


def redact_new_duplicates(messages: list[Message], new_start: int) -> list[Message]:
    """Append-only dedup: redact duplicate content only in ``messages[new_start:]``.

    Coverage is built from the live (non-stub, non-placeholder) results before
    ``new_start``; a new result whose chunks for a doc are all already present
    gets that doc's content replaced with a pointer to the earlier copy. Earlier
    messages are returned by reference, untouched. New results are processed
    oldest-first, so a later result in the same round dedups against an earlier
    one.
    """
    tool_infos = _identify_tool_messages(messages)
    if not any(info.msg_index >= new_start for info in tool_infos):
        return messages

    seen: dict[int, set[int]] = {}
    result = list(messages)
    for info in sorted(tool_infos, key=lambda x: x.msg_index):
        if info.msg_index >= new_start:
            redact_docs = {
                doc_idx for doc_idx, chunks in info.coverage.items()
                if chunks and doc_idx in seen and chunks.issubset(seen[doc_idx])
            }
            if redact_docs:
                new_content = _redact(info, redact_docs, set(), _EARLIER_PLACEHOLDER)
                if new_content != info.content:
                    result[info.msg_index] = _with_redacted_content(
                        messages[info.msg_index], new_content,
                    )
        for doc_idx, chunks in info.coverage.items():
            seen.setdefault(doc_idx, set()).update(chunks)
    return result


def normalize_keep_newest(messages: list[Message], dynamic_context: str = "") -> list[Message]:
    """Restore every redacted result (unless it was since pruned to a stub), then
    run the keep-newest pass. Used at tool-loop edit points, where earlier
    history is being rewritten anyway: it turns append-only redactions (newer
    copy redacted) into keep-newest ones (older copy redacted), so the full copy
    is the one the pruner's recency window protects."""
    return deduplicate_tool_results(restore_redacted(messages), dynamic_context)


def restore_redacted(messages: list[Message]) -> list[Message]:
    """Undo stashed redactions. A result pruned to a stub since keeps its stub
    (only the stash is dropped)."""
    result = list(messages)
    for idx, msg in enumerate(messages):
        meta = msg.metadata or {}
        if ORIGINAL_CONTENT_KEY not in meta:
            continue
        rest = {k: v for k, v in meta.items() if k != ORIGINAL_CONTENT_KEY}
        if isinstance(msg.content, str) and msg.content.startswith(_STUB_PREFIX):
            result[idx] = msg.model_copy(update={"metadata": rest})
        else:
            result[idx] = msg.model_copy(
                update={"content": meta[ORIGINAL_CONTENT_KEY], "metadata": rest},
            )
    return result


def _with_redacted_content(msg: Message, new_content: str, *, stash: bool = True) -> Message:
    if not stash:
        return msg.model_copy(update={"content": new_content})
    meta = dict(msg.metadata or {})
    meta.setdefault(ORIGINAL_CONTENT_KEY, msg.content)
    return msg.model_copy(update={"content": new_content, "metadata": meta})


def _redact(info: "_ToolResultInfo", chunk_docs: set[int], desc_docs: set[int], placeholder: str) -> str:
    if info.tool_name == "document_search":
        return _redact_search_content(info.content, chunk_docs, desc_docs, placeholder)
    if info.tool_name == "document_read":
        return _redact_read_content(info.content, chunk_docs, placeholder)
    return info.content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identify_tool_messages(messages: list[Message]) -> list[_ToolResultInfo]:
    """Find tool-result messages for document_search / document_read, with
    their parsed chunk coverage. Pruned stubs are skipped.

    Correlates ``role="tool"`` messages to their tool names by matching
    ``tool_call_id`` against preceding assistant messages' ``tool_calls[].id``.
    """
    # Build lookup: tool_call_id -> tool_name
    call_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                call_id_to_name[tc.id] = tc.name

    target_tools = {"document_search", "document_read"}
    infos: list[_ToolResultInfo] = []

    for idx, msg in enumerate(messages):
        if msg.role != "tool" or not msg.tool_call_id:
            continue
        if not isinstance(msg.content, str):
            continue
        if msg.content.startswith(_STUB_PREFIX):
            continue  # pruned — no content left to cover or redact
        tool_name = call_id_to_name.get(msg.tool_call_id, "")
        if tool_name in target_tools:
            info = _ToolResultInfo(
                msg_index=idx,
                tool_name=tool_name,
                content=msg.content,
            )
            if tool_name == "document_search":
                info.coverage = _parse_search_coverage(msg.content)
            else:
                info.coverage = _parse_read_coverage(msg.content)
            infos.append(info)

    return infos


def _parse_search_coverage(content: str) -> dict[int, set[int]]:
    """Parse document_search markdown output to extract chunk coverage.

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
        if any(p in block for p in _PLACEHOLDERS):
            continue  # redacted copy — its content lives elsewhere

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
    """Parse document_read JSON output to extract chunk coverage.

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
        if doc.get("content") in _PLACEHOLDERS:
            continue  # redacted copy — its content lives elsewhere

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
    placeholder: str = _CONTENT_PLACEHOLDER,
) -> str:
    """Redact chunk content and/or descriptions from document_search output.

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
                body = _redact_search_block_content(body, placeholder)

        result_parts.append(header + body)
        i += 2

    return "".join(result_parts)


def _redact_search_block_content(body: str, placeholder: str = _CONTENT_PLACEHOLDER) -> str:
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
            new_lines.append(placeholder)
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


def _redact_read_content(
    content: str, doc_indices_to_redact: set[int], placeholder: str = _CONTENT_PLACEHOLDER,
) -> str:
    """Replace content field in document_read JSON for matching doc_indices."""
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
            doc["content"] = placeholder
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
