"""
End-to-end document processing: parse, chunk, persist chunks.
Phase 2: no embeddings. Phase 3 will add embedding and vector write.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from documents.models import ProjectDocument, ProjectDocumentChunk
from documents.services.chunking import extract_and_chunk_file

logger = logging.getLogger(__name__)


def process_document(document_id: int) -> None:
    """
    Load a ProjectDocument, extract text, chunk, and persist chunks.
    Sets status to PROCESSING then READY or FAILED. Does not generate embeddings (Phase 3).
    """
    doc = ProjectDocument.objects.filter(pk=document_id).first()
    if not doc:
        logger.warning("process_document: document_id=%s not found", document_id)
        return
    doc.status = ProjectDocument.Status.PROCESSING
    doc.save(update_fields=["status", "updated_at"])
    logger.info("process_document: document_id=%s project_id=%s stage=processing", document_id, doc.project_id)
    print(f"[document_id={document_id}] stage=processing", flush=True)
    started_at = time.perf_counter()
    try:
        file_path = None
        if doc.original_file:
            file_path = Path(doc.original_file.path)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {doc.original_file}")

        ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"
        if ext not in getattr(settings, "DOCUMENT_ALLOWED_EXTENSIONS", {"pdf", "txt", "md", "html"}):
            raise ValueError(f"Unsupported extension: {ext}")

        print(f"[document_id={document_id}] stage=extracting", flush=True)
        logger.info("process_document: document_id=%s stage=extracting", document_id)
        chunks_data = extract_and_chunk_file(file_path, ext)
        print(f"[document_id={document_id}] stage=chunked count={len(chunks_data)}", flush=True)
        logger.info("process_document: document_id=%s stage=chunked count=%s", document_id, len(chunks_data))
        doc.parser_type = "pypdf" if ext == "pdf" else "text"
        doc.chunking_strategy = "markdown_first" if any(c.get("heading") for c in chunks_data) else "recursive_token"

        # Remove existing chunks (idempotent re-run)
        doc.chunks.all().delete()

        for i, c in enumerate(chunks_data):
            ProjectDocumentChunk.objects.create(
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

        # Persist token_count (and chunking metadata) immediately so they're saved even if vector store fails
        doc.token_count = sum(c.get("token_count", 0) for c in chunks_data)
        doc.save(update_fields=["parser_type", "chunking_strategy", "token_count", "updated_at"])

        # Embed and index in vector store (Phase 3)
        from documents.services import vector_store as vs
        chunk_records = list(
            doc.chunks.order_by("chunk_index").values("id", "text", "chunk_index")
        )
        if chunk_records and getattr(settings, "PGVECTOR_CONNECTION", None):
            print(f"[document_id={document_id}] stage=vector_delete", flush=True)
            logger.info("process_document: document_id=%s stage=vector_delete", document_id)
            vs.delete_vectors_for_document(doc.id)
            print(f"[document_id={document_id}] stage=embedding", flush=True)
            logger.info("process_document: document_id=%s stage=embedding", document_id)
            vs.add_chunk_vectors(chunk_records, document_id=doc.id, project_id=doc.project_id)
            print(f"[document_id={document_id}] stage=vector_done", flush=True)
            logger.info("process_document: document_id=%s stage=vector_done", document_id)
        doc.embedding_model = getattr(settings, "EMBEDDING_MODEL", "")
        doc.status = ProjectDocument.Status.READY
        doc.processing_error = None
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processing_error", "processed_at", "embedding_model", "updated_at"])
        duration_seconds = time.perf_counter() - started_at
        print(f"[document_id={document_id}] stage=ready chunk_count={len(chunks_data)} duration_seconds={duration_seconds:.1f}", flush=True)
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
