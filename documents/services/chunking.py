"""
Text extraction and semantic chunking for project documents.

Uses LangChain loaders for extraction and SemanticChunker for
embedding-based splitting into ~300-token flat chunks.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
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
    """Apply general-purpose cleaning to extracted text.

    Order matters:
    0. Normalise literal escape sequences (\\n, \\r\\n) to real newlines
    1. Rejoin hyphenated line breaks (before line-level removals)
    2. Remove DOI-only lines
    3. Remove all-caps journal/publication header lines
    4. Remove standalone page numbers
    5. Collapse excess inline whitespace
    6. Collapse 3+ consecutive newlines to \\n\\n
    """
    # Step 0: Normalise literal escape sequences commonly found in
    # JSON exports, LLM outputs, or copy-paste artefacts.  Only apply
    # when the text is mostly a single blob (few real newlines) that
    # contains many literal \n sequences — avoids false-positive
    # replacement in files that legitimately reference escape codes.
    actual_newlines = text.count("\n")
    literal_newlines = text.count("\\n")
    if literal_newlines > actual_newlines and literal_newlines >= 4:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")

    text = _RE_HYPHENATED_BREAK.sub(r"\1\2", text)
    text = _RE_DOI_LINE.sub("", text)
    text = _RE_JOURNAL_HEADER.sub("", text)
    text = _RE_PAGE_NUMBER.sub("", text)
    text = _RE_EXCESS_INLINE_WS.sub(" ", text)
    text = _RE_EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# Lazy imports to avoid loading LangChain at module import when not needed
def _get_loaders():
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    return PyPDFLoader, TextLoader


def _strip_nul_bytes(docs: list[Any]) -> list[Any]:
    """Remove NUL (0x00) bytes from document page_content.

    Some PDF extractors produce NUL bytes that PostgreSQL text fields reject.
    """
    for doc in docs:
        if "\x00" in (doc.page_content or ""):
            doc.page_content = doc.page_content.replace("\x00", "")
    return docs


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable size string."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes} B"


@dataclass
class EmailAttachment:
    """An email attachment with optional extracted text content."""
    filename: str
    size_str: str
    content: str | None  # Extracted text, or None if unsupported/failed


def _extract_attachment_content(
    data: bytes, filename: str, *, _depth: int = 0
) -> str | None:
    """Try to extract text content from an email attachment.

    Returns the extracted text, or None if the file type is unsupported,
    extraction fails, or depth limit is exceeded for nested emails.
    """
    from django.conf import settings

    ext = Path(filename).suffix.lower().lstrip(".")
    allowed = getattr(settings, "DOCUMENT_ALLOWED_EXTENSIONS", set())
    if ext not in allowed:
        return None

    # Cap recursion for nested emails
    if ext in ("msg", "eml") and _depth >= 1:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=f".{ext}", delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        if ext == "msg":
            docs = _load_msg_as_markdown(tmp_path, _depth=_depth + 1)
        elif ext == "eml":
            docs = _load_eml_as_markdown(tmp_path, _depth=_depth + 1)
        else:
            docs = load_documents(tmp_path, ext)

        text = "\n\n".join(d.page_content for d in docs if d.page_content)
        return text if text.strip() else None
    except Exception:
        logger.debug(
            "Failed to extract content from attachment %r", filename, exc_info=True
        )
        return None
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _load_docx_as_markdown(path: Path) -> list[Any]:
    """Extract DOCX content as Markdown using mammoth + markdownify.

    Produces much richer text than docx2txt: headings, lists, tables and
    bold/italic are preserved as Markdown, which the structure-aware chunker
    can then split on.  Images are replaced with a simple placeholder.
    """
    import mammoth
    from langchain_core.documents import Document
    from markdownify import markdownify as md

    image_counter = 0

    def _image_placeholder(image):
        nonlocal image_counter
        image_counter += 1
        return {"alt": f"[Image {image_counter}]", "src": "#"}

    with open(path, "rb") as f:
        result = mammoth.convert_to_html(
            f, convert_image=mammoth.images.img_element(_image_placeholder)
        )

    content = md(result.value, heading_style="ATX")

    # Clean up markdownify image syntax — ![alt](#) → just alt
    content = re.sub(r"!\[(\[Image \d+\])\]\([^)]*\)", r"\1", content)

    return [Document(page_content=content.strip())]


def _format_email_as_markdown(
    subject: str | None,
    from_addr: str | None,
    to_addr: str | None,
    date: str | None,
    cc: str | None,
    body_markdown: str,
    attachments: list[EmailAttachment] | None = None,
) -> str:
    """Format email headers + body into structured Markdown."""
    subject = subject or "(No Subject)"
    parts = [f"# {subject}", ""]

    # Header table
    headers = [
        ("**From**", from_addr or ""),
        ("**To**", to_addr or ""),
        ("**Date**", date or ""),
    ]
    if cc:
        headers.append(("**CC**", cc))

    parts.append("| Header | Value |")
    parts.append("|--------|-------|")
    for label, value in headers:
        parts.append(f"| {label} | {value} |")

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(body_markdown.strip())

    if attachments:
        extracted = [a for a in attachments if a.content]
        unextracted = [a for a in attachments if not a.content]

        # Extracted attachments as subsections
        for att in extracted:
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append(f"## Attachment: {att.filename} ({att.size_str})")
            parts.append("")
            parts.append(att.content)

        # Unsupported/failed attachments as bullet list
        if unextracted:
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append("**Attachments:**")
            for att in unextracted:
                parts.append(f"- {att.filename} ({att.size_str})")

    return "\n".join(parts)


def _load_msg_as_markdown(path: Path, *, _depth: int = 0) -> list[Any]:
    """Extract .msg (Outlook) email as a Markdown LangChain Document."""
    import extract_msg
    from langchain_core.documents import Document
    from markdownify import markdownify as md

    msg = extract_msg.Message(str(path))
    try:
        subject = msg.subject
        from_addr = msg.sender
        to_addr = msg.to
        date = str(msg.date) if msg.date else None
        cc = msg.cc

        # Prefer HTML body, fall back to plain text
        html_body = msg.htmlBody
        plain_body = msg.body

        if html_body:
            if isinstance(html_body, bytes):
                html_body = html_body.decode("utf-8", errors="replace")
            body_md = md(html_body, heading_style="ATX")
        elif plain_body:
            body_md = plain_body
        else:
            raise ValueError("Email has no body content (no HTML, no plain text)")

        # Extract attachments
        attachments = []
        for att in msg.attachments:
            name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or "unnamed"
            data = getattr(att, "data", None)
            size = getattr(att, "dataLength", None) or (len(data) if data else 0)
            size_str = _format_size(size)
            extracted = _extract_attachment_content(data, name, _depth=_depth) if data else None
            attachments.append(EmailAttachment(filename=name, size_str=size_str, content=extracted))

        content = _format_email_as_markdown(
            subject=subject, from_addr=from_addr, to_addr=to_addr,
            date=date, cc=cc, body_markdown=body_md, attachments=attachments,
        )
        return [Document(page_content=content)]
    finally:
        msg.close()


def _load_eml_as_markdown(path: Path, *, _depth: int = 0) -> list[Any]:
    """Extract .eml (RFC 822) email as a Markdown LangChain Document."""
    import email
    import email.policy

    from langchain_core.documents import Document
    from markdownify import markdownify as md

    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=email.policy.default)

    subject = msg["subject"]
    from_addr = msg["from"]
    to_addr = msg["to"]
    date = msg["date"]
    cc = msg["cc"]

    # Get body: prefer HTML, fall back to plain
    body_part = msg.get_body(preferencelist=("html", "plain"))
    if body_part is None:
        raise ValueError("Email has no body content (no HTML, no plain text)")

    body_content = body_part.get_content()
    if body_part.get_content_type() == "text/html":
        body_md = md(body_content, heading_style="ATX")
    else:
        body_md = body_content

    # Extract attachments
    attachments = []
    for att in msg.iter_attachments():
        filename = att.get_filename() or "unnamed"
        data = att.get_payload(decode=True)
        size = len(data) if data else 0
        size_str = _format_size(size)
        extracted = _extract_attachment_content(data, filename, _depth=_depth) if data else None
        attachments.append(EmailAttachment(filename=filename, size_str=size_str, content=extracted))

    content = _format_email_as_markdown(
        subject=subject, from_addr=from_addr, to_addr=to_addr,
        date=str(date) if date else None, cc=cc,
        body_markdown=body_md, attachments=attachments,
    )
    return [Document(page_content=content)]


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
        docs = loader.load()
        _strip_nul_bytes(docs)
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    if ext == "docx":
        docs = _load_docx_as_markdown(path)
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    if ext == "msg":
        docs = _load_msg_as_markdown(path)
        logger.debug("load_documents output (%d doc(s)):\n%s", len(docs), "\n---\n".join(d.page_content for d in docs))
        return docs
    if ext == "eml":
        docs = _load_eml_as_markdown(path)
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
