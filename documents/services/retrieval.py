"""
Backend retrieval: get chunks by data room/document (ordered), by similarity
search, by full-text search, or hybrid (semantic + full-text with RRF).
Provides dynamic context expansion via get_chunk_with_context() and merged
context windows via get_merged_context_windows(). Optionally reranks results
with FlashRank when RERANK_ENABLED is True.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

from django.conf import settings as django_settings
from django.contrib.postgres.search import SearchQuery, SearchRank

from documents.models import DataRoomDocumentChunk, DataRoomDocument
from documents.services import vector_store as vs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reranking (FlashRank)
# ---------------------------------------------------------------------------

_ranker_cache: Any = None
_ranker_lock = threading.Lock()


def _get_ranker():
    """Lazy-init FlashRank Ranker with module-level cache + threading lock."""
    global _ranker_cache
    if _ranker_cache is not None:
        return _ranker_cache
    with _ranker_lock:
        if _ranker_cache is not None:
            return _ranker_cache
        from flashrank import Ranker
        _ranker_cache = Ranker()
        return _ranker_cache


def rerank_chunks(
    results: list[dict[str, Any]],
    query: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rerank search results using FlashRank. Falls back to truncation on error.

    Args:
        results: list of chunk dicts with at least a 'text' key.
        query: the user's search query.
        top_n: number of results to return after reranking.

    Returns:
        Reranked (or truncated) list of chunk dicts.
    """
    if not getattr(django_settings, "RERANK_ENABLED", True):
        return results[:top_n]

    if not results:
        return []

    try:
        from flashrank import RerankRequest
    except ImportError:
        logger.warning("flashrank not installed; skipping reranking")
        return results[:top_n]

    try:
        ranker = _get_ranker()
        passages = [{"id": i, "text": r.get("text", "")} for i, r in enumerate(results)]
        request = RerankRequest(query=query, passages=passages)
        reranked = ranker.rerank(request)

        # Map reranked results back to original dicts by id
        id_to_original = {i: r for i, r in enumerate(results)}
        output = []
        for item in reranked[:top_n]:
            orig_id = item["id"]
            output.append(id_to_original[orig_id])
        return output
    except Exception:
        logger.warning("Reranking failed; returning unranked results", exc_info=True)
        return results[:top_n]


def get_chunks_by_document(document_id: int) -> list[dict[str, Any]]:
    """Return chunks for a document in order (from DB)."""
    chunks = DataRoomDocumentChunk.objects.filter(
        document_id=document_id,
        is_quarantined=False,
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
    """Return chunks for a data room, grouped by document (order preserved). Excludes failed/quarantined."""
    chunks = (
        DataRoomDocumentChunk.objects.filter(
            document__data_room_id=data_room_id,
            is_quarantined=False,
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
            is_quarantined=False,
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
    Run hybrid search (semantic + full-text), optionally rerank with FlashRank,
    and return LangChain Document objects with metadata doc_index
    (data-room-scoped), data_room_id, chunk_index.
    """
    from langchain_core.documents import Document

    # Over-fetch for reranking (2x), then rerank down to k
    fetch_k = k * 2 if getattr(django_settings, "RERANK_ENABLED", True) else k
    results = hybrid_search_chunks(
        data_room_ids=data_room_ids, query=query, k=fetch_k, document_id=document_id,
    )

    # Rerank (returns top k results, or falls back to truncation)
    results = rerank_chunks(results, query, top_n=k)

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
                "doc_index": doc_index_map[r["document_id"]],
                "data_room_id": doc_room_map[r["document_id"]],
                "chunk_index": r["chunk_index"],
            },
        )
        for r in results
        if r.get("document_id") in doc_index_map
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

    # Fetch all non-quarantined chunks for the same document, ordered by chunk_index
    all_chunks = list(
        DataRoomDocumentChunk.objects.filter(
            document_id=center.document_id,
            is_quarantined=False,
        )
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

    # Expand symmetrically around center chunk
    start_pos, end_pos, total_tokens = _expand_window(center_pos, all_chunks, target_tokens)
    context_chunks = all_chunks[start_pos:end_pos + 1]
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


# ---------------------------------------------------------------------------
# Merged context windows (deduplication)
# ---------------------------------------------------------------------------


def _expand_window(
    center_pos: int,
    all_chunks: list[dict[str, Any]],
    target_tokens: int,
) -> tuple[int, int, int]:
    """Expand a window around center_pos, returning (start_pos, end_pos, total_tokens).

    start_pos and end_pos are inclusive indices into all_chunks.
    """
    start_pos = center_pos
    end_pos = center_pos
    total_tokens = all_chunks[center_pos]["token_count"]

    left = center_pos - 1
    right = center_pos + 1

    while total_tokens < target_tokens:
        added = False
        if left >= 0:
            candidate = all_chunks[left]["token_count"]
            if total_tokens + candidate <= target_tokens:
                total_tokens += candidate
                start_pos = left
                left -= 1
                added = True
            else:
                left = -1
        if right < len(all_chunks):
            candidate = all_chunks[right]["token_count"]
            if total_tokens + candidate <= target_tokens:
                total_tokens += candidate
                end_pos = right
                right += 1
                added = True
            else:
                right = len(all_chunks)
        if not added:
            break

    return start_pos, end_pos, total_tokens


def get_merged_context_windows(
    chunk_ids: list[int],
    target_tokens_per_window: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch chunks, expand into context windows, and merge overlapping windows.

    Groups chunk_ids by document. For each document, expands each hit chunk
    into a context window (symmetric expansion like get_chunk_with_context),
    then merges overlapping or adjacent windows using interval merge.

    Cross-document windows are never merged.

    Returns list of dicts:
        {chunk_ids: [...], document_id, context_text, context_token_count, chunks_included: [...]}
    """
    if not chunk_ids:
        return []

    if target_tokens_per_window is None:
        target_tokens_per_window = getattr(django_settings, "RETRIEVAL_CONTEXT_TARGET_TOKENS", 1200)

    # Fetch hit chunks to get their document_id and chunk_index
    hit_chunks = DataRoomDocumentChunk.objects.filter(pk__in=chunk_ids).values(
        "id", "document_id", "chunk_index"
    )
    # Group by document
    doc_hits: dict[int, list[dict]] = defaultdict(list)
    for hc in hit_chunks:
        doc_hits[hc["document_id"]].append(hc)

    merged_windows: list[dict[str, Any]] = []

    for doc_id, hits in doc_hits.items():
        # Fetch all non-quarantined chunks for this document, ordered
        all_chunks = list(
            DataRoomDocumentChunk.objects.filter(
                document_id=doc_id,
                is_quarantined=False,
            )
            .order_by("chunk_index")
            .values("id", "chunk_index", "text", "token_count")
        )
        if not all_chunks:
            continue

        # Build chunk_index → position map
        idx_to_pos = {c["chunk_index"]: pos for pos, c in enumerate(all_chunks)}

        # Expand each hit into a window interval (start_pos, end_pos)
        intervals: list[tuple[int, int]] = []
        hit_chunk_ids_per_interval: list[list[int]] = []

        for hit in hits:
            pos = idx_to_pos.get(hit["chunk_index"])
            if pos is None:
                continue
            start, end, _ = _expand_window(pos, all_chunks, target_tokens_per_window)
            intervals.append((start, end))
            hit_chunk_ids_per_interval.append([hit["id"]])

        if not intervals:
            continue

        # Sort by start position
        combined = sorted(zip(intervals, hit_chunk_ids_per_interval), key=lambda x: x[0][0])

        # Merge overlapping/adjacent intervals
        merged: list[tuple[int, int, list[int]]] = []
        for (start, end), cids in combined:
            if merged and start <= merged[-1][1] + 1:
                # Overlapping or adjacent — extend
                prev_start, prev_end, prev_cids = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end), prev_cids + cids)
            else:
                merged.append((start, end, list(cids)))

        # Build output
        for start_pos, end_pos, hit_cids in merged:
            window_chunks = all_chunks[start_pos:end_pos + 1]
            context_text = "\n\n".join(c["text"] for c in window_chunks)
            total_tokens = sum(c["token_count"] for c in window_chunks)
            merged_windows.append({
                "chunk_ids": hit_cids,
                "document_id": doc_id,
                "context_text": context_text,
                "context_token_count": total_tokens,
                "chunks_included": [c["chunk_index"] for c in window_chunks],
            })

    return merged_windows
