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

from documents.models import DataRoomDocument, DataRoomDocumentChunk
from documents.services.chunking import chunk_from_string, clean_extracted_text, load_documents

logger = logging.getLogger(__name__)


def process_document(document_id: int) -> None:
    """
    Load a DataRoomDocument (with select_for_update to avoid concurrent processing),
    extract text, chunk, persist chunks, then embed and index in vector store.
    Sets status to PROCESSING then READY or FAILED.
    """
    with transaction.atomic():
        doc = (
            DataRoomDocument.objects.filter(pk=document_id)
            .select_for_update(skip_locked=True)
            .first()
        )
    if not doc:
        logger.warning("process_document: document_id=%s not found or locked by another task", document_id)
        return
    doc.status = DataRoomDocument.Status.PROCESSING
    doc.save(update_fields=["status", "updated_at"])
    logger.info("process_document: document_id=%s data_room_id=%s stage=processing", document_id, doc.data_room_id)
    started_at = time.perf_counter()
    try:
        file_path = None
        if doc.original_file:
            file_path = Path(doc.original_file.path)
        if not file_path or not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {doc.original_file}")

        ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"
        doc.parser_type = "pypdf" if ext == "pdf" else "text"

        # 1. Extract
        logger.info("process_document: document_id=%s stage=extracting", document_id)
        docs = load_documents(file_path, ext)
        combined = "\n\n".join(getattr(d, "page_content", "") or "" for d in docs)
        cleaned = clean_extracted_text(combined)

        # 2. Generate description early (before normalization)
        description = ""
        if getattr(settings, "LLM_DEFAULT_CHEAP_MODEL", ""):
            try:
                from documents.services.description import generate_description_from_text
                description = generate_description_from_text(
                    cleaned, user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id
                )
                doc.description = description
                doc.save(update_fields=["description", "updated_at"])
            except Exception:
                logger.exception("process_document: document_id=%s description generation failed", document_id)

        # 3. Normalize to markdown
        try:
            from documents.services.normalization import normalize_text
            normalized = normalize_text(
                cleaned, description=description,
                user_id=doc.uploaded_by_id, data_room_id=doc.data_room_id,
            )
        except Exception:
            logger.exception("process_document: document_id=%s normalization failed, using cleaned text", document_id)
            normalized = cleaned

        # 4. Chunk from normalized string
        logger.info("process_document: document_id=%s stage=chunking", document_id)
        chunks_data = chunk_from_string(normalized)
        logger.info("process_document: document_id=%s stage=chunked count=%s", document_id, len(chunks_data))

        # Determine chunking strategy from structure
        has_children = any(c.get("children") for c in chunks_data)
        if has_children:
            has_headings = any(c.get("heading") for c in chunks_data)
            # Detect slides by checking if all children map 1:1 to parents
            is_slides = all(
                len(c.get("children", [])) == 1
                and c.get("children", [{}])[0].get("text") == c["text"]
                for c in chunks_data if c.get("children")
            )
            if is_slides:
                doc.chunking_strategy = "slides_parent_child"
            elif has_headings:
                doc.chunking_strategy = "markdown_parent_child"
            else:
                doc.chunking_strategy = "structure_parent_child"
        else:
            doc.chunking_strategy = "markdown_first" if any(c.get("heading") for c in chunks_data) else "recursive_token"

        # Remove existing chunks (idempotent re-run)
        doc.chunks.all().delete()

        # Bulk-create parent chunks
        parent_objects = [
            DataRoomDocumentChunk(
                document=doc,
                chunk_index=i,
                heading=c.get("heading"),
                text=c["text"],
                token_count=c.get("token_count", 0),
                is_child=False,
                parent=None,
                source_page_start=c.get("source_page_start"),
                source_page_end=c.get("source_page_end"),
                source_offset_start=c.get("source_offset_start"),
                source_offset_end=c.get("source_offset_end"),
            )
            for i, c in enumerate(chunks_data)
        ]
        DataRoomDocumentChunk.objects.bulk_create(parent_objects, batch_size=500)

        # Refresh parent objects to get their PKs
        parent_map = {
            obj.chunk_index: obj
            for obj in doc.chunks.filter(is_child=False).order_by("chunk_index")
        }

        # Bulk-create child chunks with globally sequential chunk_index
        child_objects = []
        global_child_index = 0
        for i, c in enumerate(chunks_data):
            parent_obj = parent_map.get(i)
            for child in c.get("children", []):
                child_objects.append(
                    DataRoomDocumentChunk(
                        document=doc,
                        chunk_index=global_child_index,
                        heading=child.get("heading"),
                        text=child["text"],
                        token_count=child.get("token_count", 0),
                        is_child=True,
                        parent=parent_obj,
                        source_page_start=None,
                        source_page_end=None,
                        source_offset_start=None,
                        source_offset_end=None,
                    )
                )
                global_child_index += 1
        if child_objects:
            DataRoomDocumentChunk.objects.bulk_create(child_objects, batch_size=500)

        # Populate full-text search vectors for parent chunks only.
        # Wrapped in its own savepoint so a failure (e.g. SQLite in dev/test) does
        # not abort the outer transaction and leave the document stuck as PROCESSING.
        try:
            with transaction.atomic():
                doc.chunks.filter(is_child=False, search_vector__isnull=True).update(
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

        # Embed and index in vector store: child chunks only (precise retrieval)
        # Falls back to parent chunks if no children exist (backward compat)
        from documents.services import vector_store as vs
        if child_objects:
            chunk_records = list(
                doc.chunks.filter(is_child=True).order_by("parent__chunk_index", "chunk_index")
                .values("id", "text", "chunk_index", "parent_id")
            )
        else:
            chunk_records = list(
                doc.chunks.order_by("chunk_index").values("id", "text", "chunk_index")
            )
        if chunk_records and getattr(settings, "PGVECTOR_CONNECTION", None):
            try:
                logger.info("process_document: document_id=%s stage=vector_delete", document_id)
                vs.delete_vectors_for_document(doc.id)
                logger.info("process_document: document_id=%s stage=embedding", document_id)
                vs.add_chunk_vectors(chunk_records, document_id=doc.id, data_room_id=doc.data_room_id)
                logger.info("process_document: document_id=%s stage=vector_done", document_id)
            except Exception as vec_err:
                logger.warning(
                    "process_document: document_id=%s vector embedding failed (non-critical): %s",
                    document_id, vec_err,
                )
        doc.embedding_model = getattr(settings, "EMBEDDING_MODEL", "")
        doc.status = DataRoomDocument.Status.READY
        doc.processing_error = None
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processing_error", "processed_at", "embedding_model", "updated_at"])

        duration_seconds = time.perf_counter() - started_at
        logger.info(
            "process_document: document_id=%s data_room_id=%s stage=ready chunk_count=%s duration_seconds=%.2f",
            document_id, doc.data_room_id, len(chunks_data), duration_seconds,
        )
    except Exception as e:
        duration_seconds = time.perf_counter() - started_at
        logger.exception(
            "process_document: document_id=%s data_room_id=%s stage=failed duration_seconds=%.2f",
            document_id, doc.data_room_id, duration_seconds,
        )
        doc.status = DataRoomDocument.Status.FAILED
        doc.processing_error = str(e)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])
