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
