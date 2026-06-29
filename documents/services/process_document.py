"""
End-to-end version processing: parse, chunk, embed, and persist chunks.

A ``DataRoomDocumentVersion`` is the unit of processing. Upload-origin versions
(v0) extract text from the native bytes; edit/save versions use the markdown
already stored in ``version.content``. Either way we chunk, persist chunks scoped
to the version, embed them (is_searchable=False), then hold the version in
SCANNING and hand off to the guardrail scan, which hands off to
``finalize_document_metadata`` — the sole releaser of SCANNING and the only place
the document's ``active_searchable_version`` pointer advances.
"""
from __future__ import annotations

import datetime
import logging
import time

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from django.contrib.postgres.search import SearchVector

# If a version has been PROCESSING longer than this, treat it as stuck and allow
# reprocessing. The in-process guard below intentionally defers crash recovery to
# the sweeper (requeue_stale_documents), which reclaims a row this long after its
# last update — so a worker hard-killed (OOM/SIGKILL) mid-PROCESSING is picked up
# within this window. Keep it SHORT for fast recovery but strictly ABOVE the
# worst-case single-version processing time, or the sweeper will requeue a job
# that is still legitimately running. Overridable for slow corpora.
STALE_PROCESSING_MINUTES = getattr(settings, "STALE_PROCESSING_MINUTES", 10)

from documents.models import (
    DataRoomDocument,
    DataRoomDocumentChunk,
    DataRoomDocumentVersion,
)
from documents.services.chunking import clean_extracted_text, extract_file_metadata_date, load_documents, semantic_chunk, structure_aware_chunk
from documents.services.storage_utils import local_copy

logger = logging.getLogger(__name__)


def process_document(document_id: int) -> None:
    """Back-compat entry point used by the upload path and the stale sweeper.

    Ensures a v0 exists for a freshly-uploaded document (no version yet), then
    processes the document's working version.
    """
    doc = DataRoomDocument.objects.filter(pk=document_id).first()
    if not doc:
        logger.warning("process_document: document_id=%s not found", document_id)
        return
    version_id = doc.current_version_id or _ensure_initial_version(doc).id
    process_document_version(version_id)


def _ensure_initial_version(doc) -> DataRoomDocumentVersion:
    """Create v0 (origin=uploaded) for a document that has no version yet.

    The native bytes stay on ``doc.original_file`` (no copy into native_blob);
    extraction reads them from there.
    """
    with transaction.atomic():
        d = DataRoomDocument.objects.select_for_update().get(pk=doc.pk)
        if d.current_version_id:
            return d.current_version
        v0 = DataRoomDocumentVersion.objects.create(
            document=d,
            version_index=0,
            origin=DataRoomDocumentVersion.Origin.UPLOADED,
            native_filename=d.original_filename,
            mime_type=d.mime_type or "",
            size_bytes=d.size_bytes,
            created_by=d.uploaded_by,
        )
        DataRoomDocument.objects.filter(pk=d.pk).update(current_version=v0)
        return v0


def _mirror_doc_status(doc, version, status) -> None:
    """Mirror status + processing_error from a version onto the document.

    Only mirrors when this version is (or will become) the live one — i.e. the
    document has no active searchable version yet (fresh upload) or this version
    already is the active one. Editing a new version while an older one is live
    must NOT change the document's user-visible status.
    """
    if doc.active_searchable_version_id in (None, version.id):
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            status=status, processing_error=version.processing_error, updated_at=timezone.now(),
        )


def _mirror_doc_metadata(doc, version) -> None:
    """Mirror a version's processing metadata (token_count, parser, dates) onto the
    document for UI/admin display — only when the version is (or will become) live."""
    if doc.active_searchable_version_id in (None, version.id):
        DataRoomDocument.objects.filter(pk=doc.pk).update(
            token_count=version.token_count,
            parser_type=version.parser_type,
            chunking_strategy=version.chunking_strategy,
            embedding_model=version.embedding_model,
            processed_at=version.processed_at,
            updated_at=timezone.now(),
        )


def _extract_native(version, doc):
    """Extract cleaned text + file metadata date from a version's native bytes.

    Source is ``version.native_blob`` when present, else the document's
    ``original_file`` (legacy/fresh v0 keep the bytes on the document). Returns
    ``(cleaned_text, file_metadata_date|None)``.
    """
    source_file = version.native_blob if version.native_blob else doc.original_file
    if not source_file:
        raise FileNotFoundError("No native source file for this version.")

    filename = version.native_filename or doc.original_filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    with local_copy(source_file) as file_path:
        from core.file_types import is_image_extension
        from llm.transcription_registry import AUDIO_EXTENSIONS

        if ext in AUDIO_EXTENSIONS:
            # --- Audio transcription branch (upload only) ---
            version.parser_type = "audio"
            from core.preferences import get_preferences
            prefs = get_preferences(doc.uploaded_by)
            if not prefs.allowed_transcription_models:
                raise ValueError("Audio transcription is not enabled for your organization.")
            transcription_model_id = prefs.transcription_model
            if not transcription_model_id:
                raise ValueError("No transcription model available.")

            logger.info("process_document_version: version_id=%s stage=transcribing model=%s", version.id, transcription_model_id)
            from documents.services.transcription import transcribe_audio
            transcript_text = transcribe_audio(file_path, transcription_model_id, user=doc.uploaded_by)
            # Transcript is document-level metadata.
            DataRoomDocument.objects.filter(pk=doc.pk).update(
                transcript=transcript_text, transcription_model=transcription_model_id, updated_at=timezone.now(),
            )
            cleaned = clean_extracted_text(transcript_text)
        elif is_image_extension(ext):
            # --- Image description branch ---
            # The vision-generated description becomes the document's searchable
            # text; the original image bytes stay on the source file (native_blob
            # for a re-uploaded version, else doc.original_file for a fresh v0)
            # for later viewing — see _collect_doc_images, which reads them back
            # with the same precedence. GDPR NOTE: only this description flows through the
            # guardrail + PII text scanners — the raw image bytes are NOT
            # independently PII-scanned (description-only for v1; a vision-based
            # PII signal is a planned follow-up).
            version.parser_type = "image"
            from chat.services import describe_image
            from core.file_types import canonical_mime_for_extension
            from core.preferences import resolve_org_feature_model
            from documents.services.pii_scan import org_id_for_document

            org_id = org_id_for_document(doc)
            model = resolve_org_feature_model(org_id, "document_image_description")
            if not model:
                raise ValueError("Image description is not enabled for your organization.")
            with open(file_path, "rb") as fh:
                img_bytes = fh.read()
            media_type = canonical_mime_for_extension(ext) or doc.mime_type or "image/png"
            logger.info(
                "process_document_version: version_id=%s stage=describing_image model=%s",
                version.id, model,
            )
            description = describe_image(img_bytes, media_type, doc.uploaded_by, model=model)
            if not description:
                raise ValueError("Could not generate a description for this image.")
            cleaned = clean_extracted_text(description)
        else:
            # --- Text extraction branch ---
            if ext == "pdf":
                version.parser_type = "pypdf"
            elif ext in ("msg", "eml"):
                version.parser_type = ext
            else:
                version.parser_type = "text"

            logger.info("process_document_version: version_id=%s stage=extracting", version.id)
            image_sink = None
            if ext in ("docx", "pdf"):
                # Embedded images become Assets (bytes preserved) + inline
                # [[image:uuid|...]] tokens (searchable descriptions).
                from documents.services.image_assets import image_asset_sink
                image_sink = image_asset_sink(version, doc)
            docs = load_documents(file_path, ext, image_sink=image_sink)
            combined = "\n\n".join(getattr(d, "page_content", "") or "" for d in docs)
            cleaned = clean_extracted_text(combined)
            del docs, combined

        file_meta_date = extract_file_metadata_date(file_path, ext)
    return cleaned, file_meta_date


def process_document_version(version_id: int, *, dispatch_scan: bool = True) -> None:
    """Process a single version: extract/use markdown, chunk, embed, hand off to scan.

    ``dispatch_scan=False`` suppresses only the async guardrail-scan dispatch at the
    tail, leaving the version held ``status="scanning"`` with chunks embedded
    (``is_searchable=False``). The synchronous save path (``sync_scan``) uses this so
    it can run the same scan inline and read the verdict back, instead of handing off
    to Celery.
    """
    with transaction.atomic():
        version = (
            DataRoomDocumentVersion.objects.filter(pk=version_id)
            .select_for_update(skip_locked=True)
            .select_related("document")
            .first()
        )
        if not version:
            logger.warning("process_document_version: version_id=%s not found or locked", version_id)
            return
        if version.status == DataRoomDocument.Status.PROCESSING:
            stale_threshold = timezone.now() - datetime.timedelta(minutes=STALE_PROCESSING_MINUTES)
            if version.updated_at > stale_threshold:
                logger.info("process_document_version: version_id=%s already processing, skipping", version_id)
                return
            logger.warning("process_document_version: version_id=%s stuck PROCESSING since %s, reprocessing", version_id, version.updated_at)
        version.status = DataRoomDocument.Status.PROCESSING
        version.save(update_fields=["status", "updated_at"])

    doc = version.document
    _mirror_doc_status(doc, version, "processing")
    logger.info("process_document_version: version_id=%s document_id=%s stage=processing", version_id, doc.id)
    started_at = time.perf_counter()
    try:
        # 1. Get cleaned text — markdown edits use stored content; uploads extract.
        if (version.content or "").strip():
            cleaned = clean_extracted_text(version.content)
            version.parser_type = "markdown"
            file_meta_date = None
        else:
            cleaned, file_meta_date = _extract_native(version, doc)

        if not cleaned or not cleaned.strip():
            raise ValueError(
                "No text could be extracted from this document. "
                "It may be a scanned PDF or image-only file that requires OCR."
            )

        max_chars = getattr(settings, "DOCUMENT_MAX_EXTRACTED_CHARS", 20_000_000)
        if len(cleaned) > max_chars:
            raise ValueError(
                "This document's extracted text is too large to process "
                f"(over {max_chars // 1_000_000} million characters)."
            )

        # 2. Chunk (strategy from settings)
        logger.info("process_document_version: version_id=%s stage=chunking", version_id)
        strategy = getattr(settings, "CHUNKING_STRATEGY", "structure_aware")
        if strategy == "structure_aware":
            chunks_data = structure_aware_chunk(cleaned)
            version.chunking_strategy = "structure_aware"
        else:
            chunks_data = semantic_chunk(cleaned)
            version.chunking_strategy = "semantic"
        chunk_count = len(chunks_data)
        del cleaned
        logger.info("process_document_version: version_id=%s stage=chunked count=%s", version_id, chunk_count)

        if not chunks_data:
            raise ValueError(
                "Document text was extracted but produced 0 chunks. "
                "The file may have unusual formatting that the parser cannot split."
            )

        # Remove this version's existing chunks (idempotent re-run)
        version.chunks.all().delete()

        chunk_objects = [
            DataRoomDocumentChunk(
                version=version,
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
        del chunk_objects

        # Full-text search vectors (own savepoint; non-critical, e.g. SQLite in dev).
        try:
            with transaction.atomic():
                version.chunks.filter(search_vector__isnull=True).update(
                    search_vector=(
                        SearchVector("heading", weight="A", config="english")
                        + SearchVector("text", weight="B", config="english")
                    )
                )
        except Exception as fts_err:
            logger.warning("process_document_version: version_id=%s fts update failed: %s", version_id, fts_err)

        version.token_count = sum(c.get("token_count", 0) for c in chunks_data)
        del chunks_data
        version.save(update_fields=["parser_type", "chunking_strategy", "token_count", "updated_at"])

        # Record file metadata date on the document (first non-null wins).
        if file_meta_date and not doc.file_metadata_date:
            DataRoomDocument.objects.filter(pk=doc.pk, file_metadata_date__isnull=True).update(
                file_metadata_date=file_meta_date,
            )

        # 3. Embed this version's chunks (is_searchable=False until it becomes active).
        from documents.services import vector_store as vs
        from documents.services.chunk_access import iter_version_chunks
        if getattr(settings, "PGVECTOR_CONNECTION", None):
            try:
                logger.info("process_document_version: version_id=%s stage=vector_delete", version_id)
                vs.delete_vectors_for_version(version.id)
                logger.info("process_document_version: version_id=%s stage=embedding", version_id)
                vs.add_chunk_vectors(
                    iter_version_chunks(version.id, fields=("id", "text", "chunk_index")),
                    document_id=doc.id, data_room_id=doc.data_room_id,
                    version_id=version.id, is_searchable=False,
                )
                logger.info("process_document_version: version_id=%s stage=vector_done", version_id)
            except Exception as vec_err:
                logger.warning(
                    "process_document_version: version_id=%s vector embedding failed (non-critical): %s",
                    version_id, vec_err,
                )

        # Hold the version in SCANNING — the guardrail scan runs first and hands off
        # to finalize_document_metadata, the sole releaser of SCANNING and the only
        # place active_searchable_version advances.
        version.embedding_model = getattr(settings, "EMBEDDING_MODEL", "")
        version.status = "scanning"
        version.processing_error = None
        version.processed_at = timezone.now()
        version.save(update_fields=["status", "processing_error", "processed_at", "embedding_model", "updated_at"])
        _mirror_doc_status(doc, version, "scanning")
        _mirror_doc_metadata(doc, version)

        duration_seconds = time.perf_counter() - started_at
        logger.info(
            "process_document_version: version_id=%s document_id=%s stage=scanning chunk_count=%s duration_seconds=%.2f",
            version_id, doc.id, chunk_count, duration_seconds,
        )

        # Dispatch the guardrail chunk scan (hands off to finalize on success).
        # Suppressed when the caller runs the scan synchronously (sync_scan).
        if dispatch_scan:
            try:
                from guardrails.tasks import scan_document_version
                scan_document_version.delay(version.id)
            except Exception:
                from documents.services.pii_scan import SCAN_FAILED_MESSAGE
                logger.exception(
                    "process_document_version: version_id=%s guardrail scan dispatch failed; marking scan_failed", version_id,
                )
                version.status = "scan_failed"
                version.processing_error = SCAN_FAILED_MESSAGE
                version.save(update_fields=["status", "processing_error", "updated_at"])
                _mirror_doc_status(doc, version, "scan_failed")

    except ValueError as e:
        duration_seconds = time.perf_counter() - started_at
        logger.warning(
            "process_document_version: version_id=%s stage=failed duration_seconds=%.2f — %s",
            version_id, duration_seconds, e,
        )
        version.status = "failed"
        version.processing_error = str(e)[:2000]
        version.save(update_fields=["status", "processing_error", "updated_at"])
        _mirror_doc_status(doc, version, "failed")
    except Exception as e:
        duration_seconds = time.perf_counter() - started_at
        logger.exception(
            "process_document_version: version_id=%s stage=failed duration_seconds=%.2f",
            version_id, duration_seconds,
        )
        version.status = "failed"
        version.processing_error = str(e)[:2000]
        version.save(update_fields=["status", "processing_error", "updated_at"])
        _mirror_doc_status(doc, version, "failed")
