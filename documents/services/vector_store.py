"""
pgvector store for chunk embeddings. Uses LangChain PGVector when
PGVECTOR_CONNECTION is set (Postgres). Idempotent: delete by document_id before add.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from django.conf import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"

# Chunks embedded + inserted per store.add_documents() call. Overridable via the
# EMBEDDING_BATCH_SIZE setting; bounds peak vector memory during indexing.
DEFAULT_EMBEDDING_BATCH_SIZE = 256

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
        from sqlalchemy.pool import NullPool

        _vector_store_cache = PGVector(
            embeddings,
            connection=conn,
            collection_name=COLLECTION_NAME,
            use_jsonb=True,
            # prepare_threshold=None disables psycopg3 prepared statements so this
            # engine is safe through PgBouncer transaction pooling (Django's own
            # connection already defaults to this; see config/settings.py DATABASES).
            engine_args={"poolclass": NullPool, "connect_args": {"prepare_threshold": None}},
        )
        return _vector_store_cache


def delete_vectors_for_document(document_id: int) -> None:
    """Remove from vector store all chunks belonging to this document.

    Called on re-index (``process_document``) and on document deletion (the
    ``post_delete`` signal) — the embedding rows carry the full chunk text in
    their ``document`` column, so GDPR erasure requires removing them too.

    Runs on the same SQLAlchemy engine that ``add_chunk_vectors`` writes
    through (built from PGVECTOR_CONNECTION), so deletes can never silently
    target a different database than the inserts.
    """
    conn = _get_connection_string()
    if not conn:
        return
    store = _get_vector_store()
    if not store:
        return
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    # PGVector stores rows for all collections; scope deletion to this app
    # collection. Filters on cmetadata->>'document_id' (no index) — fine at
    # current volume; add an expression index if bulk deletes ever get slow.
    stmt = text(
        "DELETE FROM langchain_pg_embedding AS emb "
        "USING langchain_pg_collection AS col "
        "WHERE emb.collection_id = col.uuid "
        "  AND col.name = :collection "
        "  AND emb.cmetadata->>'document_id' = :document_id"
    )
    try:
        with store.session_maker() as session:
            session.execute(
                stmt,
                {"collection": COLLECTION_NAME, "document_id": str(document_id)},
            )
            session.commit()
    except ProgrammingError:
        # Table doesn't exist yet; nothing to delete.
        logger.debug("langchain_pg_embedding table does not exist yet, skipping delete")


def add_chunk_vectors(
    chunks: Iterable[dict[str, Any]],
    document_id: int,
    data_room_id: int,
    *,
    version_id: int,
    is_searchable: bool = False,
    batch_size: int | None = None,
) -> None:
    """
    Embed and store chunk vectors. ``chunks`` is any iterable of dicts with 'id',
    'text', and optional 'chunk_index' — consumed in a single pass, so a streaming
    generator can be passed to bound memory. Documents are embedded and inserted in
    batches of ``batch_size`` (default ``EMBEDDING_BATCH_SIZE``) so neither all chunk
    text nor all vectors are ever materialized at once.

    ``version_id`` and ``is_searchable`` are written into cmetadata so retrieval can
    filter to the document's active searchable version. New versions are embedded
    with ``is_searchable=False``; the flag is flipped to True (via
    ``set_searchable_for_version``) only once the version becomes the active one.
    """
    conn = _get_connection_string()
    if not conn:
        logger.warning("PGVECTOR_CONNECTION not set; skipping vector index")
        return
    from langchain_core.documents import Document
    store = _get_vector_store()
    if not store:
        return
    if batch_size is None:
        batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)

    batch: list[Any] = []
    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_id": chunk["id"],
            "document_id": document_id,
            "data_room_id": data_room_id,
            "version_id": version_id,
            "is_searchable": bool(is_searchable),
            "chunk_index": chunk.get("chunk_index", i),
        }
        batch.append(Document(page_content=chunk["text"], metadata=meta))
        if len(batch) >= batch_size:
            store.add_documents(batch)
            batch = []
    if batch:
        store.add_documents(batch)


def delete_vectors_for_version(version_id: int) -> None:
    """Remove from the vector store all chunks belonging to a single version.

    Called by the prune task when an old version is dropped. Scoped by the
    ``version_id`` cmetadata key (written by ``add_chunk_vectors``).
    """
    conn = _get_connection_string()
    if not conn:
        return
    store = _get_vector_store()
    if not store:
        return
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    stmt = text(
        "DELETE FROM langchain_pg_embedding AS emb "
        "USING langchain_pg_collection AS col "
        "WHERE emb.collection_id = col.uuid "
        "  AND col.name = :collection "
        "  AND emb.cmetadata->>'version_id' = :version_id"
    )
    try:
        with store.session_maker() as session:
            session.execute(
                stmt,
                {"collection": COLLECTION_NAME, "version_id": str(version_id)},
            )
            session.commit()
    except ProgrammingError:
        logger.debug("langchain_pg_embedding table does not exist yet, skipping delete")


def set_searchable_for_version(version_id: int, value: bool) -> None:
    """Flip the ``is_searchable`` cmetadata flag for a version's vectors.

    This is the rollback / pointer-advance mechanism: a cheap in-place jsonb
    update, never a re-embed. The app-DB pointer
    (DataRoomDocument.active_searchable_version) remains the authoritative gate;
    this only keeps the pgvector filter in sync for recall.
    """
    conn = _get_connection_string()
    if not conn:
        return
    store = _get_vector_store()
    if not store:
        return
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    stmt = text(
        "UPDATE langchain_pg_embedding AS emb "
        "SET cmetadata = jsonb_set(emb.cmetadata, '{is_searchable}', to_jsonb(:value)) "
        "FROM langchain_pg_collection AS col "
        "WHERE emb.collection_id = col.uuid "
        "  AND col.name = :collection "
        "  AND emb.cmetadata->>'version_id' = :version_id"
    )
    try:
        with store.session_maker() as session:
            session.execute(
                stmt,
                {"collection": COLLECTION_NAME, "version_id": str(version_id), "value": bool(value)},
            )
            session.commit()
    except ProgrammingError:
        logger.debug("langchain_pg_embedding table does not exist yet, skipping flag flip")


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

    # Only the active searchable version of each document carries
    # is_searchable=True; this keeps stale-version embeddings out of the
    # candidate set (the app-DB post-filter is the authoritative gate).
    filt: dict[str, Any] = {"data_room_id": {"$in": data_room_ids}, "is_searchable": True}
    if document_id is not None:
        filt["document_id"] = document_id
    return store.similarity_search(query, k=k, filter=filt)
