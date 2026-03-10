"""
Text extraction and chunking for project documents.

Uses LangChain loaders and splitters: heading-first for markdown,
token-based fallback with tiktoken. Outputs chunk dicts with text,
heading, token_count, and source location fields.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from django.conf import settings

from core.tokens import count_tokens as _count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes for clean_extracted_text
# ---------------------------------------------------------------------------
_RE_HYPHENATED_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_RE_DOI_LINE = re.compile(r"^\s*https?://doi\.org/\S+\s*$", re.MULTILINE)
_RE_JOURNAL_HEADER = re.compile(r"^[A-Z :&]{10,80}$", re.MULTILINE)
_RE_PAGE_NUMBER = re.compile(
    r"^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|\d+\s+of\s+\d+|\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_EXCESS_INLINE_WS = re.compile(r"[^\S\n]{2,}")
_RE_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_extracted_text(text: str) -> str:
    """Apply general-purpose cleaning to PDF-extracted text.

    Order matters:
    1. Rejoin hyphenated line breaks (before line-level removals)
    2. Remove DOI-only lines
    3. Remove all-caps journal/publication header lines
    4. Remove standalone page numbers
    5. Collapse excess inline whitespace
    6. Collapse 3+ consecutive newlines to \\n\\n
    """
    text = _RE_HYPHENATED_BREAK.sub(r"\1\2", text)
    text = _RE_DOI_LINE.sub("", text)
    text = _RE_JOURNAL_HEADER.sub("", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_EXCESS_INLINE_WS.sub(" ", text)
    text = _RE_EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# Lazy imports to avoid loading LangChain at module import when not needed
def _get_loaders():
    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
    return Docx2txtLoader, PyPDFLoader, TextLoader


def _strip_nul_bytes(docs: list[Any]) -> list[Any]:
    """Remove NUL (0x00) bytes from document page_content.

    Some PDF extractors produce NUL bytes that PostgreSQL text fields reject.
    """
    for doc in docs:
        if "\x00" in (doc.page_content or ""):
            doc.page_content = doc.page_content.replace("\x00", "")
    return docs


def load_documents(file_path: str | Path, file_extension: str) -> list[Any]:
    """
    Load a file into a list of LangChain Document objects.
    file_extension should be lowercased (e.g. 'pdf', 'txt', 'md', 'html').
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = file_extension.lower().lstrip(".")
    Docx2txtLoader, PyPDFLoader, TextLoader = _get_loaders()
    if ext == "pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        _strip_nul_bytes(docs)
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    if ext == "docx":
        loader = Docx2txtLoader(str(path))
        docs = loader.load()
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    if ext in ("txt", "md", "html", "csv", "json", "xml", "rst", "tex", "yaml", "yml", "log"):
        # TextLoader works for all text-based; use utf-8 with errors=replace
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    raise ValueError(f"Unsupported file type: {ext}")


def _looks_like_markdown(text: str) -> bool:
    trimmed = text.strip()
    return bool(trimmed.startswith("#") or "\n## " in trimmed or "\n# " in trimmed)


MIN_CHUNK_TOKENS = 200


def _safe_min(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _safe_max(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _merge_small_chunks(
    chunks: list[dict[str, Any]],
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[dict[str, Any]]:
    """
    Iteratively merge any chunk with token_count < min_tokens into the smallest
    adjacent chunk (prev or next). Merged chunks may exceed max chunk size; we
    leave them as-is. Recomputes token_count and source fields after each merge.
    """
    chunks = [dict(c) for c in chunks]
    while True:
        i = next((idx for idx, c in enumerate(chunks) if c["token_count"] < min_tokens), None)
        if i is None:
            break
        c = chunks[i]
        prev = chunks[i - 1] if i > 0 else None
        nxt = chunks[i + 1] if i < len(chunks) - 1 else None
        if prev is None and nxt is None:
            break
        if prev is None:
            target_idx = i + 1
            target = chunks[i + 1]
            new_text = c["text"] + "\n\n" + target["text"]
        elif nxt is None:
            target_idx = i - 1
            target = chunks[i - 1]
            new_text = target["text"] + "\n\n" + c["text"]
        else:
            if prev["token_count"] <= nxt["token_count"]:
                target_idx = i - 1
                target = chunks[i - 1]
                new_text = target["text"] + "\n\n" + c["text"]
            else:
                target_idx = i + 1
                target = chunks[i + 1]
                new_text = c["text"] + "\n\n" + target["text"]
        new_text = new_text.strip()
        chunks[target_idx] = {
            "text": new_text,
            "heading": target.get("heading"),
            "token_count": _count_tokens(new_text),
            "source_page_start": _safe_min(c.get("source_page_start"), target.get("source_page_start")),
            "source_page_end": _safe_max(c.get("source_page_end"), target.get("source_page_end")),
            "source_offset_start": _safe_min(c.get("source_offset_start"), target.get("source_offset_start")),
            "source_offset_end": _safe_max(c.get("source_offset_end"), target.get("source_offset_end")),
        }
        chunks.pop(i)
    return chunks


def chunk_text(
    documents: list[Any],
    *,
    target_chunk_tokens: int | None = None,
    max_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """
    Chunk a list of LangChain Documents using structure-first parent-child splitting.

    Returns list of parent chunk dicts, each with a "children" list containing
    child chunk dicts. Parent dicts have keys: text, heading, token_count, is_child,
    source_page_start, source_page_end, source_offset_start, source_offset_end, children.
    Child dicts have: text, heading, token_count, is_child, child_index.
    """
    from documents.services.splitters import detect_structure, parent_child_split

    child_target = getattr(settings, "CHILD_CHUNK_TARGET_TOKENS", 300)
    child_max = getattr(settings, "CHILD_CHUNK_MAX_TOKENS", 600)
    child_overlap_pct = getattr(settings, "CHILD_CHUNK_OVERLAP_PCT", 0.20)

    # Combine all document contents with page metadata
    full_parts = []
    for doc in documents:
        content = getattr(doc, "page_content", "") or ""
        meta = getattr(doc, "metadata", None) or {}
        full_parts.append({"content": content, "metadata": meta})
    if not full_parts:
        return []

    combined_text = "\n\n".join(p["content"] for p in full_parts)
    if not combined_text.strip():
        return []

    # Clean extraction artifacts (page numbers, DOI lines, etc.)
    combined_text = clean_extracted_text(combined_text)

    # Detect structure (uses \f for slide boundaries)
    structure_type = detect_structure(combined_text)

    # Strip form-feed characters after structure detection but before splitting
    combined_text = combined_text.replace("\f", "")

    pc_result = parent_child_split(
        combined_text,
        structure_type=structure_type,
        child_target_tokens=child_target,
        child_overlap_pct=child_overlap_pct,
        max_child_tokens=child_max,
    )

    if not pc_result:
        return []

    # Resolve page info from document metadata
    first_meta = full_parts[0]["metadata"] if full_parts else {}
    last_meta = full_parts[-1]["metadata"] if full_parts else {}
    page_start = first_meta.get("page") if first_meta else None
    page_end = last_meta.get("page") if last_meta else None

    # Convert to chunk dicts
    chunks_out = []
    for parent_data in pc_result:
        children_out = []
        for child in parent_data.get("children", []):
            children_out.append({
                "text": child["text"],
                "heading": parent_data.get("heading"),
                "token_count": child["token_count"],
                "is_child": True,
                "child_index": child["child_index"],
            })
        chunks_out.append({
            "text": parent_data["text"],
            "heading": parent_data.get("heading"),
            "token_count": parent_data["token_count"],
            "is_child": False,
            "source_page_start": page_start,
            "source_page_end": page_end,
            "source_offset_start": parent_data.get("source_offset_start"),
            "source_offset_end": parent_data.get("source_offset_end"),
            "children": children_out,
        })

    return chunks_out


def extract_and_chunk_file(
    file_path: str | Path,
    file_extension: str,
) -> list[dict[str, Any]]:
    """
    Load a file, then chunk it. Returns list of chunk dicts ready for
    ProjectDocumentChunk creation.
    """
    docs = load_documents(file_path, file_extension)
    return chunk_text(docs)
