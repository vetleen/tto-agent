"""
pgvector store for chunk embeddings. Uses LangChain PGVector when
PGVECTOR_CONNECTION is set (Postgres). Idempotent: delete by document_id before add.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"

# Module-level cache for the vector store instance (lazy init). Use reset_vector_store() in tests to clear.
_vector_store_cache: Any = None
_vector_store_lock = threading.Lock()


def _get_connection_string() -> str | None:
    conn = getattr(settings, "PGVECTOR_CONNECTION", None) or ""
    if not conn or "sqlite" in conn.lower():
        return None
    # Normalise scheme for psycopg3 (required by langchain_postgres)
    if conn.startswith("postgres://"):
        conn = conn.replace("postgres://", "postgresql+psycopg://", 1)
    elif conn.startswith("postgresql://"):
        conn = conn.replace("postgresql://", "postgresql+psycopg://", 1)
    return conn


def reset_vector_store() -> None:
    """Clear the cached vector store instance. Use in tests for isolation."""
    global _vector_store_cache
    _vector_store_cache = None


def _get_vector_store():
    global _vector_store_cache
    if _vector_store_cache is not None:
        return _vector_store_cache
    with _vector_store_lock:
        if _vector_store_cache is not None:
            return _vector_store_cache
        from langchain_postgres import PGVector
        from langchain_openai import OpenAIEmbeddings

        conn = _get_connection_string()
        if not conn:
            return None
        timeout = getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120)
        embeddings = OpenAIEmbeddings(
            model=getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-large"),
            request_timeout=timeout,
        )
        _vector_store_cache = PGVector(
            embeddings,
            connection=conn,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        return _vector_store_cache


def delete_vectors_for_document(document_id: int) -> None:
    """Remove from vector store all chunks belonging to this document (for re-index)."""
    conn = _get_connection_string()
    if not conn:
        return
    # PGVector stores rows for all collections; scope deletion to this app collection.
    from django.db import connection, transaction
    from django.db.utils import ProgrammingError

    try:
        with transaction.atomic():
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM langchain_pg_embedding AS emb
                    USING langchain_pg_collection AS col
                    WHERE emb.collection_id = col.uuid
                      AND col.name = %s
                      AND emb.cmetadata->>'document_id' = %s
                    """,
                    [COLLECTION_NAME, str(document_id)],
                )
    except ProgrammingError:
        # Table doesn't exist yet; nothing to delete.
        logger.debug("langchain_pg_embedding table does not exist yet, skipping delete")


def add_chunk_vectors(chunks: list[dict[str, Any]], document_id: int, data_room_id: int) -> None:
    """
    Embed and store chunk vectors. chunks: list of dicts with 'id', 'text', and optional chunk_index.
    The store's embedding_function is used to embed page_content.
    """
    conn = _get_connection_string()
    if not conn:
        logger.warning("PGVECTOR_CONNECTION not set; skipping vector index")
        return
    from langchain_core.documents import Document
    store = _get_vector_store()
    if not store:
        return
    docs = []
    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_id": chunk["id"],
            "document_id": document_id,
            "data_room_id": data_room_id,
            "chunk_index": chunk.get("chunk_index", i),
        }
        if chunk.get("parent_id"):
            meta["parent_chunk_id"] = chunk["parent_id"]
        doc = Document(page_content=chunk["text"], metadata=meta)
        docs.append(doc)
    store.add_documents(docs)


def similarity_search(
    data_room_ids: list[int],
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[Any]:
    """
    Return chunks from the vector store most similar to query, filtered by data_room_ids
    (and optionally document_id). Iterates per room and merges results.
    Returns list of LangChain Documents with metadata chunk_id, document_id, data_room_id.
    """
    conn = _get_connection_string()
    if not conn:
        return []
    k = max(1, min(k, 50))
    store = _get_vector_store()
    if not store:
        return []

    all_results = []
    for room_id in data_room_ids:
        filt = {"data_room_id": room_id}
        if document_id is not None:
            filt["document_id"] = document_id
        results = store.similarity_search(query, k=k, filter=filt)
        all_results.extend(results)

    return all_results
