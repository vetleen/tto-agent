"""
pgvector store for chunk embeddings. Uses LangChain PGVector when
PGVECTOR_CONNECTION is set (Postgres). Idempotent: delete by document_id before add.
"""
from __future__ import annotations

import logging
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"


def _get_connection_string() -> str | None:
    conn = getattr(settings, "PGVECTOR_CONNECTION", None) or ""
    if not conn or "sqlite" in conn.lower():
        return None
    if conn.startswith("postgres://"):
        conn = conn.replace("postgres://", "postgresql://", 1)
    return conn


def _get_vector_store():
    from langchain_core.documents import Document
    from langchain_community.vectorstores import PGVector
    from langchain_openai import OpenAIEmbeddings
    conn = _get_connection_string()
    if not conn:
        return None
    timeout = getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120)
    embeddings = OpenAIEmbeddings(
        model=getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-large"),
        request_timeout=timeout,
    )
    return PGVector(
        connection_string=conn,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )


def delete_vectors_for_document(document_id: int) -> None:
    """Remove from vector store all chunks belonging to this document (for re-index)."""
    conn = _get_connection_string()
    if not conn:
        return
    try:
        from langchain_community.vectorstores import PGVector
        from langchain_openai import OpenAIEmbeddings
        timeout = getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120)
        embeddings = OpenAIEmbeddings(
            model=getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-large"),
            request_timeout=timeout,
        )
        store = PGVector(
            connection_string=conn,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
        )
        # PGVector stores in a table; filter by metadata document_id and delete
        # The store may not expose delete by filter; we use the underlying connection
        from django.db import connection
        with connection.cursor() as cursor:
            # LangChain PGVector table name: langchain_pg_collection + langchain_pg_embedding
            # Collection name is stored; we need to delete rows where metadata->>'document_id' = doc_id
            cursor.execute(
                "DELETE FROM langchain_pg_embedding WHERE cmetadata->>'document_id' = %s",
                [str(document_id)],
            )
    except Exception as e:
        logger.warning("delete_vectors_for_document failed: %s", e)


def add_chunk_vectors(chunks: list[dict[str, Any]], document_id: int, project_id: int) -> None:
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
            "project_id": project_id,
            "chunk_index": chunk.get("chunk_index", i),
        }
        doc = Document(page_content=chunk["text"], metadata=meta)
        docs.append(doc)
    store.add_documents(docs)


def similarity_search(
    project_id: int,
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[Any]:
    """
    Return chunks from the vector store most similar to query, filtered by project_id (and optionally document_id).
    Returns list of LangChain Documents with metadata chunk_id, document_id, project_id.
    """
    conn = _get_connection_string()
    if not conn:
        return []
    store = _get_vector_store()
    if not store:
        return []
    filt = {"project_id": str(project_id)}
    if document_id is not None:
        filt["document_id"] = str(document_id)
    results = store.similarity_search(query, k=k, filter=filt)
    return results
