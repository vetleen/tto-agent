"""
Backend retrieval: get chunks by data room/document (ordered), by similarity
search, by full-text search, or hybrid (semantic + full-text with RRF).
"""
from __future__ import annotations

import logging
from typing import Any

from django.contrib.postgres.search import SearchQuery, SearchRank

from documents.models import DataRoomDocumentChunk, DataRoomDocument
from documents.services import vector_store as vs

logger = logging.getLogger(__name__)


def get_chunks_by_document(document_id: int) -> list[dict[str, Any]]:
    """Return chunks for a document in order (from DB)."""
    chunks = DataRoomDocumentChunk.objects.filter(document_id=document_id).order_by("chunk_index")
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
    """Return all chunks for a data room, grouped by document (order preserved). Excludes failed documents."""
    chunks = (
        DataRoomDocumentChunk.objects.filter(document__data_room_id=data_room_id)
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
    # Keyed by chunk DB id → merged dict
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
