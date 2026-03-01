"""
End-to-end document processing: parse, chunk, embed, and persist chunks.
Loads document, extracts text, chunks, persists to DB, then embeds and writes to vector store.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from django.contrib.postgres.search import SearchVector

from documents.models import ProjectDocument, ProjectDocumentChunk
from documents.services.chunking import extract_and_chunk_file

logger = logging.getLogger(__name__)


def process_document(document_id: int) -> None:
    """
    Load a ProjectDocument (with select_for_update to avoid concurrent processing),
    extract text, chunk, persist chunks, then embed and index in vector store.
    Sets status to PROCESSING then READY or FAILED.
    """
    with transaction.atomic():
        doc = (
            ProjectDocument.objects.filter(pk=document_id)
            .select_for_update(skip_locked=True)
            .first()
        )
    if not doc:
        logger.warning("process_document: document_id=%s not found or locked by another task", document_id)
        return
    doc.status = ProjectDocument.Status.PROCESSING
    doc.save(update_fields=["status", "updated_at"])
    logger.info("process_document: document_id=%s project_id=%s stage=processing", document_id, doc.project_id)
    started_at = time.perf_counter()
    try:
        file_path = None
        if doc.original_file:
            file_path = Path(doc.original_file.path)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {doc.original_file}")

        ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"
        logger.info("process_document: document_id=%s stage=extracting", document_id)
        chunks_data = extract_and_chunk_file(file_path, ext)
        logger.info("process_document: document_id=%s stage=chunked count=%s", document_id, len(chunks_data))
        doc.parser_type = "pypdf" if ext == "pdf" else "text"
        doc.chunking_strategy = "markdown_first" if any(c.get("heading") for c in chunks_data) else "recursive_token"

        # Remove existing chunks (idempotent re-run)
        doc.chunks.all().delete()

        chunk_objects = [
            ProjectDocumentChunk(
                document=doc,
                chunk_index=i,
                heading=c.get("heading"),
                text=c["text"],
                token_count=c.get("token_count", 0),
                source_page_start=c.get("source_page_start"),
                source_page_end=c.get("source_page_end"),
                source_offset_start=c.get("source_offset_start"),
                source_offset_end=c.get("source_offset_end"),
            )
            for i, c in enumerate(chunks_data)
        ]
        ProjectDocumentChunk.objects.bulk_create(chunk_objects, batch_size=500)

        # Populate full-text search vectors for hybrid retrieval
        try:
            doc.chunks.filter(search_vector__isnull=True).update(
                search_vector=(
                    SearchVector("heading", weight="A", config="english")
                    + SearchVector("text", weight="B", config="english")
                )
            )
        except Exception as fts_err:
            # FTS is non-critical; log and continue (e.g. SQLite in dev)
            logger.warning("process_document: document_id=%s fts update failed: %s", document_id, fts_err)

        # Persist token_count (and chunking metadata) immediately so they're saved even if vector store fails
        doc.token_count = sum(c.get("token_count", 0) for c in chunks_data)
        doc.save(update_fields=["parser_type", "chunking_strategy", "token_count", "updated_at"])

        # Embed and index in vector store
        from documents.services import vector_store as vs
        chunk_records = list(
            doc.chunks.order_by("chunk_index").values("id", "text", "chunk_index")
        )
        if chunk_records and getattr(settings, "PGVECTOR_CONNECTION", None):
            logger.info("process_document: document_id=%s stage=vector_delete", document_id)
            vs.delete_vectors_for_document(doc.id)
            logger.info("process_document: document_id=%s stage=embedding", document_id)
            vs.add_chunk_vectors(chunk_records, document_id=doc.id, project_id=doc.project_id)
            logger.info("process_document: document_id=%s stage=vector_done", document_id)
        doc.embedding_model = getattr(settings, "EMBEDDING_MODEL", "")
        doc.status = ProjectDocument.Status.READY
        doc.processing_error = None
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processing_error", "processed_at", "embedding_model", "updated_at"])
        duration_seconds = time.perf_counter() - started_at
        logger.info(
            "process_document: document_id=%s project_id=%s stage=ready chunk_count=%s duration_seconds=%.2f",
            document_id, doc.project_id, len(chunks_data), duration_seconds,
        )
    except Exception as e:
        duration_seconds = time.perf_counter() - started_at
        logger.exception(
            "process_document: document_id=%s project_id=%s stage=failed duration_seconds=%.2f",
            document_id, doc.project_id, duration_seconds,
        )
        doc.status = ProjectDocument.Status.FAILED
        doc.processing_error = str(e)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])
