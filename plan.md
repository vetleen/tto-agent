# Plan: Add Hybrid Search (pgvector + Postgres Full-Text Search)

## Goal
Augment the existing semantic-only RAG retrieval with Postgres full-text search (tsvector/tsquery) using Reciprocal Rank Fusion (RRF) to combine results. Zero new infrastructure — leverages existing Postgres.

---

## Step 1: Add `search_vector` field to `ProjectDocumentChunk` model

**File:** `documents/models.py`

- Add `django.contrib.postgres.search.SearchVectorField` to `ProjectDocumentChunk`
- Add a `GinIndex` on the new field for fast full-text lookups
- The field is nullable (null=True) so existing rows don't break and we can backfill

```python
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex

class ProjectDocumentChunk(models.Model):
    # ... existing fields ...
    search_vector = SearchVectorField(null=True)  # NEW

    class Meta:
        indexes = [
            models.Index(fields=["document", "chunk_index"]),
            GinIndex(fields=["search_vector"], name="chunk_search_vector_gin"),  # NEW
        ]
```

## Step 2: Create Django migration

**File:** `documents/migrations/0004_add_search_vector.py` (auto-generated)

- Run `makemigrations` to generate the migration adding the `search_vector` field and GIN index
- The migration will be safe to run on existing data (nullable field, no data migration required for deployment)

## Step 3: Populate `search_vector` during document processing

**File:** `documents/services/process_document.py`

After `bulk_create` of chunk objects, update the `search_vector` for all new chunks using a single SQL UPDATE:

```python
from django.contrib.postgres.search import SearchVector

# After bulk_create, populate search vectors in one query
ProjectDocumentChunk.objects.filter(document=doc).update(
    search_vector=SearchVector("text", config="english")
)
```

This runs after `bulk_create` and before vector store embedding, so it's naturally part of the processing pipeline. If heading is present, we can weight it higher:

```python
from django.contrib.postgres.search import SearchVector

ProjectDocumentChunk.objects.filter(document=doc).update(
    search_vector=(
        SearchVector("heading", weight="A", config="english")
        + SearchVector("text", weight="B", config="english")
    )
)
```

Using weight "A" for headings and "B" for body text gives headings a ranking boost in full-text results.

## Step 4: Add full-text search function to retrieval

**File:** `documents/services/retrieval.py`

Add a new function `fulltext_search_chunks()` that queries the `search_vector` field:

```python
from django.contrib.postgres.search import SearchQuery, SearchRank

def fulltext_search_chunks(
    project_id: int,
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Full-text search over chunk search_vector field.
    Returns chunk dicts ranked by ts_rank, highest first.
    """
    search_query = SearchQuery(query, config="english", search_type="websearch")
    qs = (
        ProjectDocumentChunk.objects
        .filter(
            document__project_id=project_id,
            search_vector__isnull=False,
        )
        .exclude(document__status=ProjectDocument.Status.FAILED)
        .annotate(rank=SearchRank("search_vector", search_query))
        .filter(rank__gt=0)
        .order_by("-rank")
    )
    if document_id is not None:
        qs = qs.filter(document_id=document_id)
    qs = qs[:k]
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
```

Key details:
- Uses `search_type="websearch"` so queries like `"patent 123" OR grant` work naturally
- Filters `rank > 0` to exclude non-matching chunks
- Returns plain dicts (not LangChain Documents) — the hybrid function normalizes both sources

## Step 5: Add hybrid search with Reciprocal Rank Fusion (RRF)

**File:** `documents/services/retrieval.py`

Add `hybrid_search_chunks()` that runs both searches and fuses results:

```python
def hybrid_search_chunks(
    project_id: int,
    query: str,
    k: int = 10,
    document_id: int | None = None,
    semantic_weight: float = 1.0,
    fulltext_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining pgvector semantic similarity and Postgres full-text search
    using Reciprocal Rank Fusion (RRF).
    """
```

Algorithm:
1. Fetch `2 * k` results from semantic search (pgvector) — over-fetch to improve fusion quality
2. Fetch `2 * k` results from full-text search (tsvector)
3. Compute RRF score for each unique chunk: `score = sum(weight / (rrf_k + rank_position))` across both result sets
4. Sort by combined RRF score descending, return top `k`
5. If either search backend is unavailable (e.g., pgvector not configured), gracefully fall back to the other

RRF is standard for hybrid search (used by Elasticsearch, Pinecone, etc.) and doesn't require score normalization between the two backends.

The function converts pgvector LangChain Documents into the same dict format as FTS results, keyed by chunk `id` for deduplication.

## Step 6: Update `similarity_search_chunks` to use hybrid search

**File:** `documents/services/retrieval.py`

Replace the body of `similarity_search_chunks()` to call `hybrid_search_chunks()` and convert back to LangChain Document objects (to preserve the existing return type contract used by `SearchDocumentsTool`):

```python
def similarity_search_chunks(
    project_id: int,
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[Any]:
    """
    Run hybrid search (semantic + full-text) and return LangChain Document objects.
    Maintains backward compatibility with existing callers.
    """
    from langchain_core.documents import Document

    results = hybrid_search_chunks(
        project_id=project_id, query=query, k=k, document_id=document_id
    )
    return [
        Document(
            page_content=r["text"],
            metadata={
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "project_id": project_id,
                "chunk_index": r["chunk_index"],
            },
        )
        for r in results
    ]
```

This means `SearchDocumentsTool` and any other caller of `similarity_search_chunks()` gets hybrid search **with zero changes** to the tool layer. The API contract is preserved.

## Step 7: Add a management command to backfill existing chunks

**File:** `documents/management/commands/backfill_search_vectors.py` (new file)

A simple management command that populates `search_vector` for any chunks that have `search_vector IS NULL`:

```python
# python manage.py backfill_search_vectors
```

Updates in batches (by document) to avoid locking the whole table. This is needed once for any data that existed before the migration. After that, `process_document` handles it automatically for new uploads.

## Step 8: Add tests

**File:** `documents/tests/test_services.py` (extend existing)

Tests to add:
1. **`test_fulltext_search_chunks_returns_matching_results`** — insert chunks, run FTS query, verify matches
2. **`test_fulltext_search_chunks_filters_by_project`** — verify project scoping
3. **`test_fulltext_search_chunks_excludes_failed_documents`** — verify FAILED status excluded
4. **`test_hybrid_search_fuses_results`** — mock both backends, verify RRF fusion logic
5. **`test_hybrid_search_falls_back_to_fulltext_when_pgvector_unavailable`** — verify graceful degradation
6. **`test_hybrid_search_falls_back_to_semantic_when_no_fts_results`** — verify one-sided results still work
7. **`test_process_document_populates_search_vector`** — verify search_vector is set after processing

These follow the existing test patterns (Django TestCase, `@patch` for vector store, `override_settings` for config).

---

## Files Changed Summary

| File | Change |
|------|--------|
| `documents/models.py` | Add `search_vector` field + GIN index |
| `documents/migrations/0004_*.py` | Auto-generated migration |
| `documents/services/process_document.py` | Populate `search_vector` after chunk creation |
| `documents/services/retrieval.py` | Add `fulltext_search_chunks()`, `hybrid_search_chunks()`, update `similarity_search_chunks()` |
| `documents/management/commands/backfill_search_vectors.py` | New: one-time backfill command |
| `documents/tests/test_services.py` | New tests for FTS and hybrid search |

## Files NOT Changed

| File | Why |
|------|-----|
| `documents/services/vector_store.py` | No changes — pgvector layer stays as-is |
| `documents/services/chunking.py` | No changes — chunking logic unchanged |
| `chat/tools.py` | No changes — `SearchDocumentsTool` already calls `similarity_search_chunks()` which will now use hybrid search transparently |
| `config/settings.py` | No new settings needed (weights can be hardcoded initially; configurable later if needed) |

## Risk Assessment

- **Low risk**: The `search_vector` field is nullable, so the migration is safe for existing data
- **Low risk**: If FTS returns no results (e.g., before backfill), hybrid search falls back to semantic-only — identical to current behavior
- **No downtime**: Migration adds a nullable column + index, no table rewrite
- **Reversible**: Removing the feature is just dropping the column and reverting the retrieval code
