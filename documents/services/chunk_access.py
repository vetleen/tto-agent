"""
Memory-bounded access to a document's chunks.

These helpers read a document's chunks without ever materializing all of them at
once. They back three callers: the embedding pipeline (``process_document``), the
windowed full-document PII scan (``pii_scan``), and the description's head/tail
reconstruction (``finalize_document_metadata``).

Why keyset pagination instead of ``QuerySet.iterator()``: the web/worker dynos route
DB connections through an in-dyno PgBouncer in transaction-pooling mode, so
``DISABLE_SERVER_SIDE_CURSORS=True`` is set (see ``config/settings.py``). With
server-side cursors disabled, ``.iterator()`` does NOT stream from Postgres — psycopg
fetches the entire result set client-side. Keyset pagination over the indexed
``(document, chunk_index)`` keeps each query bounded to a single page.
"""
from __future__ import annotations

import logging

from documents.models import DataRoomDocument, DataRoomDocumentChunk

logger = logging.getLogger(__name__)


def _keyset_pages(document_id, fields, page_size, *, reverse=False):
    """Yield chunk dicts in chunk_index order (ascending, or descending if reverse).

    Pages through the chunks with a keyset cursor on ``chunk_index`` so at most
    ``page_size`` rows are held at a time. ``chunk_index`` is always selected (it is
    the cursor) even when the caller omits it from ``fields``.
    """
    select = tuple(fields)
    if "chunk_index" not in select:
        select = select + ("chunk_index",)
    order = "-chunk_index" if reverse else "chunk_index"
    bound = "chunk_index__lt" if reverse else "chunk_index__gt"
    cursor = None
    while True:
        qs = DataRoomDocumentChunk.objects.filter(document_id=document_id)
        if cursor is not None:
            qs = qs.filter(**{bound: cursor})
        page = list(qs.order_by(order).values(*select)[:page_size])
        if not page:
            return
        yield from page
        cursor = page[-1]["chunk_index"]
        if len(page) < page_size:
            return


def iter_document_chunks(document_id, *, fields=("id", "text", "chunk_index"), page_size=1000):
    """Yield a document's chunks as dicts in ascending chunk_index order.

    Never holds more than ``page_size`` rows in memory at once. Includes every chunk
    (no quarantine filter) so embedding, description, and PII all see the full
    document — matching the pre-refactor behaviour that fed them the whole text.
    """
    yield from _keyset_pages(document_id, fields, page_size, reverse=False)


def _join_chunks(chunks) -> str:
    """Join chunk dicts into text, prefixing each chunk's heading when present."""
    parts = []
    for c in chunks:
        heading = (c.get("heading") or "").strip()
        text = c.get("text") or ""
        parts.append(f"{heading}\n{text}" if heading else text)
    return "\n\n".join(parts)


def _document_token_total(document_id) -> int:
    """Total token count for a document — from the stored field, or summed from chunks."""
    row = DataRoomDocument.objects.filter(pk=document_id).values("token_count").first()
    total = (row or {}).get("token_count") or 0
    if total:
        return total
    return sum(
        (c["token_count"] or 0)
        for c in iter_document_chunks(document_id, fields=("token_count",), page_size=1000)
    )


def build_head_tail_text(document_id, head_tokens=None, tail_tokens=None) -> str:
    """Reconstruct document text for the description LLM, bounded to head + tail.

    Small documents (<= ``_MAX_INPUT_TOKENS``) are returned whole. Larger ones return
    the first ``head_tokens`` + last ``tail_tokens`` worth of chunks with the middle
    omitted — the same slice ``_prepare_document_text`` would keep, but without ever
    materializing the full document text. Chunks are reconstructed with headings
    prepended, so the result differs slightly from the original cleaned text
    (acceptable for a relevance gist).
    """
    from documents.services.description import _HEAD_TOKENS, _MAX_INPUT_TOKENS, _TAIL_TOKENS

    if head_tokens is None:
        head_tokens = _HEAD_TOKENS
    if tail_tokens is None:
        tail_tokens = _TAIL_TOKENS

    fields = ("heading", "text", "chunk_index", "token_count")
    total = _document_token_total(document_id)

    if total <= _MAX_INPUT_TOKENS:
        return _join_chunks(iter_document_chunks(document_id, fields=fields, page_size=1000))

    # Head: forward-scan until the head budget is reached.
    head, head_tok, head_indexes = [], 0, set()
    for c in iter_document_chunks(document_id, fields=fields, page_size=128):
        head.append(c)
        head_tok += c["token_count"] or 0
        head_indexes.add(c["chunk_index"])
        if head_tok >= head_tokens:
            break

    # Tail: reverse-scan until the tail budget is reached, skipping any chunk already
    # in the head (defensive against a pathological single huge chunk straddling both).
    tail, tail_tok = [], 0
    for c in _keyset_pages(document_id, fields, 128, reverse=True):
        if c["chunk_index"] in head_indexes:
            continue
        tail.append(c)
        tail_tok += c["token_count"] or 0
        if tail_tok >= tail_tokens:
            break
    tail.reverse()  # restore ascending order

    parts = [_join_chunks(head)]
    omitted = total - head_tok - tail_tok
    if omitted > 0:
        parts.append(f"[... middle ~{omitted} tokens omitted ...]")
    if tail:
        parts.append(_join_chunks(tail))
    return "\n\n".join(parts)
