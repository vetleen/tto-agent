"""
Text extraction and chunking for project documents.

Uses LangChain loaders and splitters: heading-first for markdown,
token-based fallback with tiktoken. Outputs chunk dicts with text,
heading, token_count, and source location fields.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading LangChain at module import when not needed
def _get_loaders():
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    return PyPDFLoader, TextLoader


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
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


def load_documents(file_path: str | Path, file_extension: str) -> list[Any]:
    """
    Load a file into a list of LangChain Document objects.
    file_extension should be lowercased (e.g. 'pdf', 'txt', 'md', 'html').
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = file_extension.lower().lstrip(".")
    PyPDFLoader, TextLoader = _get_loaders()
    if ext == "pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()
    if ext in ("txt", "md", "html"):
        # TextLoader works for all text-based; use utf-8 with errors=replace
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        return loader.load()
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
    Chunk a list of LangChain Documents (e.g. from load_documents).
    Uses MarkdownHeaderTextSplitter when content looks like markdown,
    then enforces max token size. Otherwise uses RecursiveCharacterTextSplitter
    with tiktoken. Returns list of dicts with keys: text, heading, token_count,
    source_page_start, source_page_end, source_offset_start, source_offset_end.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    target = target_chunk_tokens or getattr(settings, "TARGET_CHUNK_TOKENS", 768)
    max_tokens = max_chunk_tokens or getattr(settings, "MAX_CHUNK_TOKENS", 1200)
    overlap = chunk_overlap_tokens or getattr(settings, "CHUNK_OVERLAP_TOKENS", 100)

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

    total_tokens = _count_tokens(combined_text)
    if total_tokens <= max_tokens:
        first_meta = full_parts[0]["metadata"] if full_parts else {}
        last_meta = full_parts[-1]["metadata"] if full_parts else {}
        page_start = first_meta.get("page") if first_meta else None
        page_end = last_meta.get("page") if last_meta else None
        return [{
            "text": combined_text.strip(),
            "heading": None,
            "token_count": total_tokens,
            "source_page_start": page_start,
            "source_page_end": page_end,
            "source_offset_start": 0,
            "source_offset_end": len(combined_text),
        }]

    chunks_out = []
    use_markdown = _looks_like_markdown(combined_text)
    md_splits = []

    if use_markdown:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        try:
            md_splits = splitter.split_text(combined_text)
        except Exception as e:
            logger.warning("Markdown split failed, falling back to recursive: %s", e)
            use_markdown = False

    if use_markdown and md_splits:
        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            encoding_name="cl100k_base",
        )
        current_offset = 0
        for i, split_doc in enumerate(md_splits):
            content = getattr(split_doc, "page_content", "") or ""
            meta = getattr(split_doc, "metadata", None) or {}
            heading = meta.get("Header 1") or meta.get("Header 2") or meta.get("Header 3") or ""
            if _count_tokens(content) <= max_tokens and content.strip():
                chunks_out.append({
                    "text": content.strip(),
                    "heading": heading[:512] if heading else None,
                    "token_count": _count_tokens(content),
                    "source_page_start": None,
                    "source_page_end": None,
                    "source_offset_start": current_offset,
                    "source_offset_end": current_offset + len(content),
                })
                current_offset += len(content)
            else:
                sub_splits = token_splitter.split_text(content)
                for j, sub in enumerate(sub_splits):
                    if not sub.strip():
                        continue
                    tok = _count_tokens(sub)
                    chunks_out.append({
                        "text": sub.strip(),
                        "heading": heading[:512] if heading else None,
                        "token_count": tok,
                        "source_page_start": None,
                        "source_page_end": None,
                        "source_offset_start": current_offset,
                        "source_offset_end": current_offset + len(sub),
                    })
                    current_offset += len(sub)
    else:
        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=overlap,
            encoding_name="cl100k_base",
        )
        page_start = None
        for doc in documents:
            content = getattr(doc, "page_content", "") or ""
            meta = getattr(doc, "metadata", None) or {}
            page = meta.get("page") if meta else None
            if page is not None and page_start is None:
                page_start = page
            splits = token_splitter.split_text(content)
            for part in splits:
                if not part.strip():
                    continue
                chunks_out.append({
                    "text": part.strip(),
                    "heading": None,
                    "token_count": _count_tokens(part),
                    "source_page_start": page if page is not None else None,
                    "source_page_end": page if page is not None else None,
                    "source_offset_start": None,
                    "source_offset_end": None,
                })

    return _merge_small_chunks(chunks_out)


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
