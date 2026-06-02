"""
End-to-end document processing: parse, chunk, embed, and persist chunks.
Loads document, extracts text, chunks semantically, persists to DB,
then embeds and writes to vector store.
"""
from __future__ import annotations

import datetime
import logging
import time

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from django.contrib.postgres.search import SearchVector

# If a document has been PROCESSING longer than this, treat it as stuck and allow reprocessing.
STALE_PROCESSING_MINUTES = 15

from documents.models import DataRoomDocument, DataRoomDocumentChunk
from documents.services.chunking import clean_extracted_text, extract_file_metadata_date, load_documents, semantic_chunk, structure_aware_chunk
from documents.services.storage_utils import local_copy

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
        if doc.status == DataRoomDocument.Status.PROCESSING:
            stale_threshold = timezone.now() - datetime.timedelta(minutes=STALE_PROCESSING_MINUTES)
            if doc.updated_at > stale_threshold:
                logger.info("process_document: document_id=%s already processing, skipping", document_id)
                return
            logger.warning(
                "process_document: document_id=%s stuck as PROCESSING since %s, reprocessing",
                document_id, doc.updated_at,
            )
        doc.status = DataRoomDocument.Status.PROCESSING
        doc.save(update_fields=["status", "updated_at"])
    logger.info("process_document: document_id=%s data_room_id=%s stage=processing", document_id, doc.data_room_id)
    started_at = time.perf_counter()
    try:
        if not doc.original_file:
            raise FileNotFoundError(f"Document file not found: {doc.original_file}")

        ext = (doc.original_filename or "").rsplit(".", 1)[-1].lower() or "txt"

        with local_copy(doc.original_file) as file_path:
            from llm.transcription_registry import AUDIO_EXTENSIONS

            if ext in AUDIO_EXTENSIONS:
                # --- Audio transcription branch ---
                doc.parser_type = "audio"
                from core.preferences import get_preferences
                prefs = get_preferences(doc.uploaded_by)
                if not prefs.allowed_transcription_models:
                    raise ValueError("Audio transcription is not enabled for your organization.")
                transcription_model_id = prefs.transcription_model
                if not transcription_model_id:
                    raise ValueError("No transcription model available.")

                logger.info("process_document: document_id=%s stage=transcribing model=%s", document_id, transcription_model_id)
                from documents.services.transcription import transcribe_audio
                transcript_text = transcribe_audio(file_path, transcription_model_id, user=doc.uploaded_by)
                doc.transcript = transcript_text
                doc.transcription_model = transcription_model_id
                doc.save(update_fields=["parser_type", "transcript", "transcription_model", "updated_at"])
                cleaned = clean_extracted_text(transcript_text)
            else:
                # --- Text extraction branch ---
                if ext == "pdf":
                    doc.parser_type = "pypdf"
                elif ext in ("msg", "eml"):
                    doc.parser_type = ext
                else:
                    doc.parser_type = "text"

                # 1. Extract
                logger.info("process_document: document_id=%s stage=extracting", document_id)
                docs = load_documents(file_path, ext)
                combined = "\n\n".join(getattr(d, "page_content", "") or "" for d in docs)
                cleaned = clean_extracted_text(combined)
                # Free the extraction intermediates now — only `cleaned` is needed downstream.
                del docs, combined

            # Extract date from file metadata (best-effort, all formats)
            file_meta_date = extract_file_metadata_date(file_path, ext)
            if file_meta_date:
                doc.file_metadata_date = file_meta_date

        if not cleaned or not cleaned.strip():
            raise ValueError(
                "No text could be extracted from this document. "
                "It may be a scanned PDF or image-only file that requires OCR."
            )

        # 2. Chunk (strategy from settings)
        logger.info("process_document: document_id=%s stage=chunking", document_id)
        strategy = getattr(settings, "CHUNKING_STRATEGY", "structure_aware")
        if strategy == "structure_aware":
            chunks_data = structure_aware_chunk(cleaned)
            doc.chunking_strategy = "structure_aware"
        else:
            chunks_data = semantic_chunk(cleaned)
            doc.chunking_strategy = "semantic"
        chunk_count = len(chunks_data)
        # `cleaned` is no longer needed — description/PII moved to finalize_document_metadata.
        del cleaned
        logger.info("process_document: document_id=%s stage=chunked count=%s strategy=%s", document_id, chunk_count, doc.chunking_strategy)

        if not chunks_data:
            raise ValueError(
                "Document text was extracted but produced 0 chunks. "
                "The file may have unusual formatting that the parser cannot split."
            )

        # Remove existing chunks (idempotent re-run)
        doc.chunks.all().delete()

        # Bulk-create flat chunks
        chunk_objects = [
            DataRoomDocumentChunk(
                document=doc,
                chunk_index=c["chunk_index"],
                heading=c.get("heading"),
                text=c["text"],
                token_count=c.get("token_count", 0),
                source_page_start=c.get("source_page_start"),
                source_page_end=c.get("source_page_end"),
                source_offset_start=c.get("source_offset_start"),
                source_offset_end=c.get("source_offset_end"),
            )
            for c in chunks_data
        ]
        DataRoomDocumentChunk.objects.bulk_create(chunk_objects, batch_size=500)
        del chunk_objects  # persisted — drop the in-memory model instances

        # Populate full-text search vectors for all chunks.
        # Wrapped in its own savepoint so a failure (e.g. SQLite in dev/test) does
        # not abort the outer transaction and leave the document stuck as PROCESSING.
        try:
            with transaction.atomic():
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
        del chunks_data  # done with the in-memory chunk dicts
        doc.save(update_fields=["parser_type", "chunking_strategy", "token_count", "file_metadata_date", "updated_at"])

        # Embed and index all chunks in vector store. Chunks are streamed from the DB
        # in keyset pages and embedded in batches, so neither all chunk text nor all
        # vectors are ever materialized at once.
        from documents.services import vector_store as vs
        from documents.services.chunk_access import iter_document_chunks
        if getattr(settings, "PGVECTOR_CONNECTION", None):
            try:
                logger.info("process_document: document_id=%s stage=vector_delete", document_id)
                vs.delete_vectors_for_document(doc.id)
                logger.info("process_document: document_id=%s stage=embedding", document_id)
                vs.add_chunk_vectors(
                    iter_document_chunks(doc.id, fields=("id", "text", "chunk_index")),
                    document_id=doc.id, data_room_id=doc.data_room_id,
                )
                logger.info("process_document: document_id=%s stage=vector_done", document_id)
            except Exception as vec_err:
                logger.warning(
                    "process_document: document_id=%s vector embedding failed (non-critical): %s",
                    document_id, vec_err,
                )

        # Mark document as READY before description generation
        doc.embedding_model = getattr(settings, "EMBEDDING_MODEL", "")
        doc.status = DataRoomDocument.Status.READY
        doc.processing_error = None
        doc.processed_at = timezone.now()
        doc.save(update_fields=["status", "processing_error", "processed_at", "embedding_model", "updated_at"])

        duration_seconds = time.perf_counter() - started_at
        logger.info(
            "process_document: document_id=%s data_room_id=%s stage=ready chunk_count=%s duration_seconds=%.2f",
            document_id, doc.data_room_id, chunk_count, duration_seconds,
        )

        # Scan chunks for adversarial content (fire-and-forget, after READY)
        try:
            from guardrails.tasks import scan_document_chunks
            scan_document_chunks.delay(document_id)
        except Exception:
            logger.exception("process_document: document_id=%s guardrail scan dispatch failed (non-critical)", document_id)

        # Generate description + tags and run the full-document PII scan in a separate
        # task, so this heavy frame returns and frees every copy of the document text
        # before any LLM work runs. The document is already READY and usable.
        try:
            from documents.tasks import finalize_document_metadata
            finalize_document_metadata.delay(document_id)
        except Exception:
            logger.exception("process_document: document_id=%s finalize dispatch failed (non-critical)", document_id)

    except ValueError as e:
        duration_seconds = time.perf_counter() - started_at
        logger.warning(
            "process_document: document_id=%s data_room_id=%s stage=failed duration_seconds=%.2f — %s",
            document_id, doc.data_room_id, duration_seconds, e,
        )
        doc.status = DataRoomDocument.Status.FAILED
        doc.processing_error = str(e)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])
    except Exception as e:
        duration_seconds = time.perf_counter() - started_at
        logger.exception(
            "process_document: document_id=%s data_room_id=%s stage=failed duration_seconds=%.2f",
            document_id, doc.data_room_id, duration_seconds,
        )
        doc.status = DataRoomDocument.Status.FAILED
        doc.processing_error = str(e)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])
