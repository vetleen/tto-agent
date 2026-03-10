"""
Backend retrieval: get chunks by data room/document (ordered), by similarity
search, by full-text search, or hybrid (semantic + full-text with RRF).
Provides dynamic context expansion via get_chunk_with_context().
"""
from __future__ import annotations

import logging
from typing import Any

from django.conf import settings as django_settings
from django.contrib.postgres.search import SearchQuery, SearchRank

from documents.models import DataRoomDocumentChunk, DataRoomDocument
from documents.services import vector_store as vs

logger = logging.getLogger(__name__)


def get_chunks_by_document(document_id: int) -> list[dict[str, Any]]:
    """Return chunks for a document in order (from DB)."""
    chunks = DataRoomDocumentChunk.objects.filter(
        document_id=document_id,
    ).order_by("chunk_index")
    return [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
            "source_page_start": c.source_page_start,
            "source_page_end": c.source_page_end,
        }
        for c in chunks
    ]


def get_chunks_by_data_room(data_room_id: int) -> list[dict[str, Any]]:
    """Return chunks for a data room, grouped by document (order preserved). Excludes failed documents."""
    chunks = (
        DataRoomDocumentChunk.objects.filter(
            document__data_room_id=data_room_id,
        )
        .exclude(document__status=DataRoomDocument.Status.FAILED)
        .exclude(document__is_archived=True)
        .select_related("document")
        .order_by("document_id", "chunk_index")
    )
    return [
        {
            "id": c.id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
        }
        for c in chunks
    ]


def fulltext_search_chunks(
    data_room_ids: list[int],
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Full-text search over chunk search_vector field using Postgres tsquery.
    Returns chunk dicts ranked by ts_rank, highest first.
    Uses 'websearch' search_type so natural queries like "patent 123" work.
    """
    search_query = SearchQuery(query, config="english", search_type="websearch")
    qs = (
        DataRoomDocumentChunk.objects.filter(
            document__data_room_id__in=data_room_ids,
            search_vector__isnull=False,
        )
        .exclude(document__status=DataRoomDocument.Status.FAILED)
        .exclude(document__is_archived=True)
        .annotate(rank=SearchRank("search_vector", search_query))
        .filter(rank__gt=0)
        .order_by("-rank")
    )
    if document_id is not None:
        qs = qs.filter(document_id=document_id)
    qs = qs[: max(1, min(k, 50))]
    return [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "heading": c.heading,
            "document_id": c.document_id,
            "rank": float(c.rank),
        }
        for c in qs
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

_RRF_K = 60  # Standard RRF constant (mitigates impact of high-rank outliers)


def _rrf_score(rank_position: int, weight: float = 1.0, rrf_k: int = _RRF_K) -> float:
    """Reciprocal Rank Fusion score for a single result at *rank_position* (0-based)."""
    return weight / (rrf_k + rank_position + 1)


def hybrid_search_chunks(
    data_room_ids: list[int],
    query: str,
    k: int = 10,
    document_id: int | None = None,
    semantic_weight: float = 1.0,
    fulltext_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining pgvector semantic similarity and Postgres full-text
    search using Reciprocal Rank Fusion (RRF).

    Over-fetches ``2*k`` from each backend, computes per-chunk RRF scores,
    and returns the top *k* results.  Gracefully degrades: if one backend is
    unavailable the other's results are returned alone.
    """
    fetch_k = max(1, min(k * 2, 50))

    # ---- Semantic results (pgvector) ----------------------------------------
    semantic_results: list[Any] = []
    if semantic_weight > 0:
        try:
            semantic_results = vs.similarity_search(
                data_room_ids=data_room_ids, query=query, k=fetch_k, document_id=document_id,
            )
        except Exception:
            logger.exception("hybrid_search: semantic search failed, continuing with fulltext only")

    # ---- Full-text results (tsvector) ----------------------------------------
    fts_results: list[dict[str, Any]] = []
    if fulltext_weight > 0:
        try:
            fts_results = fulltext_search_chunks(
                data_room_ids=data_room_ids, query=query, k=fetch_k, document_id=document_id,
            )
        except Exception:
            logger.exception("hybrid_search: fulltext search failed, continuing with semantic only")

    # ---- Exclude archived documents from semantic results ----------------------
    if semantic_results:
        archived_doc_ids = set(
            DataRoomDocument.objects.filter(
                data_room_id__in=data_room_ids, is_archived=True,
            ).values_list("pk", flat=True)
        )
        if archived_doc_ids:
            semantic_results = [
                doc for doc in semantic_results
                if (getattr(doc, "metadata", {}) or {}).get("document_id") not in archived_doc_ids
            ]

    # ---- Fuse with RRF -------------------------------------------------------
    # Keyed by chunk DB id
    scored: dict[int, dict[str, Any]] = {}

    for rank_pos, doc in enumerate(semantic_results):
        meta = getattr(doc, "metadata", {}) or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id is None:
            continue

        entry = scored.setdefault(chunk_id, {
            "id": chunk_id,
            "chunk_index": meta.get("chunk_index", 0),
            "text": getattr(doc, "page_content", ""),
            "heading": None,
            "document_id": meta.get("document_id"),
            "data_room_id": meta.get("data_room_id"),
            "rrf_score": 0.0,
        })
        entry["rrf_score"] += _rrf_score(rank_pos, weight=semantic_weight)

    for rank_pos, fts_hit in enumerate(fts_results):
        chunk_id = fts_hit["id"]
        entry = scored.setdefault(chunk_id, {
            "id": chunk_id,
            "chunk_index": fts_hit.get("chunk_index", 0),
            "text": fts_hit["text"],
            "heading": fts_hit.get("heading"),
            "document_id": fts_hit.get("document_id"),
            "rrf_score": 0.0,
        })
        # Prefer the richer FTS dict for text/heading if already present
        if fts_hit.get("heading"):
            entry["heading"] = fts_hit["heading"]
        entry["rrf_score"] += _rrf_score(rank_pos, weight=fulltext_weight)

    ranked = sorted(scored.values(), key=lambda r: r["rrf_score"], reverse=True)
    return ranked[:k]


def similarity_search_chunks(
    data_room_ids: list[int],
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[Any]:
    """
    Run hybrid search (semantic + full-text) and return LangChain Document
    objects with metadata doc_index (data-room-scoped), data_room_id, chunk_index.
    """
    from langchain_core.documents import Document

    results = hybrid_search_chunks(
        data_room_ids=data_room_ids, query=query, k=k, document_id=document_id,
    )

    # Resolve database PKs to data-room-scoped doc_index values
    doc_pks = {r["document_id"] for r in results if r.get("document_id")}
    doc_index_map: dict[int, int] = {}
    doc_room_map: dict[int, int] = {}
    if doc_pks:
        for pk, doc_index, room_id in DataRoomDocument.objects.filter(pk__in=doc_pks).values_list("pk", "doc_index", "data_room_id"):
            doc_index_map[pk] = doc_index
            doc_room_map[pk] = room_id

    return [
        Document(
            page_content=r["text"],
            metadata={
                "chunk_id": r["id"],
                "doc_index": doc_index_map.get(r["document_id"], 0),
                "data_room_id": doc_room_map.get(r["document_id"], 0),
                "chunk_index": r["chunk_index"],
            },
        )
        for r in results
    ]


def get_chunk_with_context(
    chunk_id: int,
    target_tokens: int | None = None,
) -> dict[str, Any]:
    """Fetch a chunk and expand with neighboring chunks until reaching token budget.

    Expands symmetrically (alternating left/right neighbors by chunk_index).
    Returns dict with: id, document_id, chunk_index, text, context_text,
    token_count, context_token_count, chunks_included.
    """
    if target_tokens is None:
        target_tokens = getattr(django_settings, "RETRIEVAL_CONTEXT_TARGET_TOKENS", 1200)

    try:
        center = DataRoomDocumentChunk.objects.get(pk=chunk_id)
    except DataRoomDocumentChunk.DoesNotExist:
        return {"error": f"Chunk {chunk_id} not found"}

    # Fetch all chunks for the same document, ordered by chunk_index
    all_chunks = list(
        DataRoomDocumentChunk.objects.filter(document_id=center.document_id)
        .order_by("chunk_index")
        .values("id", "chunk_index", "text", "token_count")
    )

    # Find center position in list
    center_pos = None
    for i, c in enumerate(all_chunks):
        if c["id"] == chunk_id:
            center_pos = i
            break

    if center_pos is None:
        return {"error": f"Chunk {chunk_id} not found in document chunks"}

    # Start with center chunk
    included_indices = [center_pos]
    total_tokens = all_chunks[center_pos]["token_count"]

    # Expand symmetrically
    left = center_pos - 1
    right = center_pos + 1

    while total_tokens < target_tokens:
        added = False
        # Try left
        if left >= 0:
            candidate_tokens = all_chunks[left]["token_count"]
            if total_tokens + candidate_tokens <= target_tokens:
                included_indices.insert(0, left)
                total_tokens += candidate_tokens
                left -= 1
                added = True
            else:
                left = -1  # Stop trying left
        # Try right
        if right < len(all_chunks):
            candidate_tokens = all_chunks[right]["token_count"]
            if total_tokens + candidate_tokens <= target_tokens:
                included_indices.append(right)
                total_tokens += candidate_tokens
                right += 1
                added = True
            else:
                right = len(all_chunks)  # Stop trying right

        if not added:
            break

    included_indices.sort()
    context_chunks = [all_chunks[i] for i in included_indices]
    context_text = "\n\n".join(c["text"] for c in context_chunks)

    return {
        "id": chunk_id,
        "document_id": center.document_id,
        "chunk_index": center.chunk_index,
        "text": center.text,
        "context_text": context_text,
        "token_count": center.token_count,
        "context_token_count": total_tokens,
        "chunks_included": [c["chunk_index"] for c in context_chunks],
    }
