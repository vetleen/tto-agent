"""
Text extraction and semantic chunking for project documents.

Uses LangChain loaders for extraction and SemanticChunker for
embedding-based splitting into ~300-token flat chunks.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

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


def _is_list_line(line: str) -> bool:
    """Check if a line starts a list item (-, *, or numbered)."""
    stripped = line.lstrip()
    if stripped.startswith(("- ", "* ")):
        return True
    # Numbered list: "1. ", "2. ", etc.
    if re.match(r"\d+\.\s", stripped):
        return True
    return False


def _is_table_line(line: str) -> bool:
    """Check if a line is part of a Markdown table (contains pipes)."""
    stripped = line.strip()
    return "|" in stripped and len(stripped) > 1


# Regex for Markdown headings: ATX style (# Heading) and underline style (===, ---)
_RE_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_RE_UNDERLINE_HEADING = re.compile(r"^(.+)\n([=\-]{3,})\s*$", re.MULTILINE)


def _split_into_structural_sections(text: str) -> list[dict[str, Any]]:
    """Split text into sections based on structural boundaries.

    Returns list of dicts: {heading: str|None, text: str}
    """
    lines = text.split("\n")
    sections: list[dict[str, Any]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for ATX-style heading: # Heading
        atx_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if atx_match:
            # Flush current section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append({"heading": current_heading, "text": section_text})
                current_lines = []
            current_heading = atx_match.group(2).strip()
            i += 1
            continue

        # Check for underline-style heading: text followed by === or ---
        if (
            i + 1 < len(lines)
            and line.strip()
            and re.match(r"^[=\-]{3,}\s*$", lines[i + 1])
        ):
            # Flush current section
            if current_lines:
                section_text = "\n".join(current_lines).strip()
                if section_text:
                    sections.append({"heading": current_heading, "text": section_text})
                current_lines = []
            current_heading = line.strip()
            i += 2  # Skip heading and underline
            continue

        current_lines.append(line)
        i += 1

    # Flush final section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append({"heading": current_heading, "text": section_text})

    return sections


def _split_section_preserving_blocks(text: str) -> list[str]:
    """Split a section's text into blocks, keeping tables and lists atomic.

    Returns a list of text blocks. Tables and lists are kept as single blocks.
    Paragraph breaks (double newlines) serve as secondary boundaries.
    """
    lines = text.split("\n")
    blocks: list[str] = []
    current_block: list[str] = []
    in_table = False
    in_list = False

    for line in lines:
        is_table = _is_table_line(line)
        is_list = _is_list_line(line)
        is_blank = not line.strip()

        # Transition out of table
        if in_table and not is_table and not is_blank:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            in_table = False

        # Transition out of list
        if in_list and not is_list and not line.startswith("  ") and is_blank:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            in_list = False

        # Start table
        if is_table and not in_table:
            # Flush any pending paragraph
            if current_block and not in_list:
                blocks.append("\n".join(current_block))
                current_block = []
            in_table = True

        # Start list
        if is_list and not in_list and not in_table:
            # Flush any pending paragraph
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            in_list = True

        # Paragraph break (double newline) — split if not in atomic block
        if is_blank and not in_table and not in_list:
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            continue

        current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return [b.strip() for b in blocks if b.strip()]


def structure_aware_chunk(text: str) -> list[dict[str, Any]]:
    """Split text into chunks using structural boundaries (headings, tables, lists).

    Algorithm:
    1. Pre-split on structural boundaries (headings, tables, lists, paragraph breaks)
    2. Group into sections under headings
    3. Per section: if tokens <= TARGET_CHUNK_TOKENS, emit as single chunk.
       If larger, delegate to semantic_chunk() for sub-splitting.
    4. Propagate headings to each chunk
    5. Number all chunks sequentially 0, 1, 2, ...

    Returns list of dicts: {text, token_count, chunk_index, heading}
    """
    if not text.strip():
        return []

    from django.conf import settings
    target = getattr(settings, "TARGET_CHUNK_TOKENS", 768)

    sections = _split_into_structural_sections(text)
    chunks: list[dict[str, Any]] = []

    for section in sections:
        heading = section["heading"]
        section_text = section["text"]
        section_tokens = _count_tokens(section_text)

        if section_tokens <= target:
            # Small enough — emit as single chunk
            chunks.append({
                "text": section_text,
                "token_count": section_tokens,
                "chunk_index": 0,  # Will be renumbered
                "heading": heading,
            })
        else:
            # Large section — split into blocks first, then group/delegate
            blocks = _split_section_preserving_blocks(section_text)
            # Try to group blocks into chunks under the token budget
            current_group: list[str] = []
            current_tokens = 0

            for block in blocks:
                block_tokens = _count_tokens(block)

                if block_tokens > target:
                    # Flush current group
                    if current_group:
                        group_text = "\n\n".join(current_group)
                        chunks.append({
                            "text": group_text,
                            "token_count": _count_tokens(group_text),
                            "chunk_index": 0,
                            "heading": heading,
                        })
                        current_group = []
                        current_tokens = 0

                    # Oversized block — delegate to semantic_chunk
                    sub_chunks = semantic_chunk(block)
                    for sc in sub_chunks:
                        sc["heading"] = heading
                        chunks.append(sc)
                elif current_tokens + block_tokens > target:
                    # Group is full, flush it
                    group_text = "\n\n".join(current_group)
                    chunks.append({
                        "text": group_text,
                        "token_count": _count_tokens(group_text),
                        "chunk_index": 0,
                        "heading": heading,
                    })
                    current_group = [block]
                    current_tokens = block_tokens
                else:
                    current_group.append(block)
                    current_tokens += block_tokens

            # Flush remaining group
            if current_group:
                group_text = "\n\n".join(current_group)
                chunks.append({
                    "text": group_text,
                    "token_count": _count_tokens(group_text),
                    "chunk_index": 0,
                    "heading": heading,
                })

    # Renumber all chunks sequentially
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


def semantic_chunk(text: str) -> list[dict[str, Any]]:
    """Split text into semantic chunks using embedding-based breakpoints.

    Uses SemanticChunker from langchain_experimental with OpenAI embeddings.
    Returns flat list of dicts: {text, token_count, chunk_index}.
    """
    if not text.strip():
        return []

    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    docs = chunker.create_documents([text])
    chunks = []
    for i, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "token_count": _count_tokens(chunk_text),
                "chunk_index": i,
            })

    return chunks
