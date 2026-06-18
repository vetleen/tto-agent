"""Backfill ``version_id`` + ``is_searchable`` into pgvector cmetadata.

Existing embedding rows carry ``chunk_id`` in their cmetadata. We add
``version_id`` (the chunk's version) and ``is_searchable`` (whether that version
is the document's active searchable version) so retrieval can filter in pgvector.

This is best-effort: the app-DB pointer (DataRoomDocument.active_searchable_version)
is the authoritative gate (the semantic post-filter re-checks it), so a missing or
stale cmetadata flag never leaks a non-active version. No-ops under SQLite/tests
(no langchain_pg_embedding table) and logs+skips if pgvector is a separate DB.
"""
from __future__ import annotations

import logging

from django.db import migrations

logger = logging.getLogger(__name__)


def backfill_vector_metadata(apps, schema_editor):
    from documents.services.vector_store import (
        COLLECTION_NAME,
        _get_connection_string,
        _get_vector_store,
    )

    if not _get_connection_string():
        return  # pgvector not configured (e.g. SQLite tests)
    store = _get_vector_store()
    if store is None:
        return

    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError, OperationalError

    # Same-DB set-based update: join embeddings → chunk → version. Works when
    # pgvector shares the Django Postgres (the default, settings.PGVECTOR_CONNECTION
    # falls back to DATABASE_URL). If pgvector is a separate DB the join fails and
    # we log+skip — the authoritative app-DB pointer still gates retrieval.
    stmt = text(
        "UPDATE langchain_pg_embedding AS emb "
        "SET cmetadata = emb.cmetadata "
        "  || jsonb_build_object('version_id', c.version_id, "
        "                        'is_searchable', v.is_searchable) "
        "FROM documents_dataroomdocumentchunk AS c "
        "JOIN documents_dataroomdocumentversion AS v ON v.id = c.version_id "
        "JOIN langchain_pg_collection AS col ON col.uuid = emb.collection_id "
        "WHERE col.name = :collection "
        "  AND (emb.cmetadata->>'chunk_id') ~ '^[0-9]+$' "
        "  AND (emb.cmetadata->>'chunk_id')::bigint = c.id"
    )
    try:
        with store.session_maker() as session:
            session.execute(stmt, {"collection": COLLECTION_NAME})
            session.commit()
    except (ProgrammingError, OperationalError) as exc:
        # Missing table (no embeddings yet) or pgvector on a separate DB.
        logger.warning(
            "0015 vector metadata backfill skipped (%s). If pgvector is a separate "
            "database, re-run the backfill manually after deploy.",
            type(exc).__name__,
        )


def reverse(apps, schema_editor):
    # Non-destructive forward op; nothing to undo (extra cmetadata keys are inert).
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("documents", "0014_repoint_chunks_tags_to_version"),
    ]

    operations = [
        migrations.RunPython(backfill_vector_metadata, reverse),
    ]
