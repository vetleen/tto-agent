"""Shared rules for turning a file into a data-room document.

The upload view (``documents.views.document_upload``) and the email-attachment
splitter (``documents.services.email_attachments``) both create documents from a
file + filename. They share the per-file checks, the dedupe lookup, the in-flight
cap and the create-v0-and-queue step from here, so an attachment split out of an
email is held to exactly the rules of a direct upload.
"""
from __future__ import annotations

import logging

from django.conf import settings
from django.db.models import Q

from core.files import safe_filename
from documents.models import DataRoomDocument

logger = logging.getLogger(__name__)

IN_FLIGHT_CAP_MESSAGE = "You're uploading too many files — wait for some to finish before adding more."

# Browsers send these for any type they don't recognize — they carry no signal,
# so they always pass the extension cross-check.
GENERIC_MIME_TYPES = {"", "application/octet-stream"}


def safe_original_filename(filename: str, max_length: int = 255) -> str:
    """Normalize and cap client-provided file names for safe persistence/display.

    Thin wrapper over the shared ``core.files.safe_filename`` (with a
    document-flavoured fallback) so the documents and meetings apps can't drift.
    """
    return safe_filename(filename, fallback="document", max_length=max_length)


def file_extension(filename: str) -> str:
    return (filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""


def allowed_extension(filename: str) -> bool:
    return file_extension(filename) in getattr(
        settings, "DOCUMENT_ALLOWED_EXTENSIONS", {"pdf", "txt", "md", "html"}
    )


def _allowed_mime(mime_type: str) -> bool:
    allowed_mime_types = getattr(settings, "DOCUMENT_ALLOWED_MIME_TYPES", None)
    # Empty/undefined allowlist means MIME checking is disabled.
    if not allowed_mime_types:
        return True
    return mime_type in allowed_mime_types


def mime_matches_extension(ext: str, mime_type: str) -> bool:
    """Cross-check the browser-supplied MIME type against the file extension.

    A mapped extension must carry one of its expected MIME types (or a generic
    one); unmapped extensions fall back to the global allowlist.
    """
    if mime_type in GENERIC_MIME_TYPES:
        return True
    allowed_for_ext = getattr(settings, "DOCUMENT_EXTENSION_MIME_MAP", {}).get(ext)
    if allowed_for_ext is None:
        return _allowed_mime(mime_type)
    return mime_type in allowed_for_ext


def live_documents_qs(data_room):
    """Documents in *data_room* that count as "already uploaded".

    Archived documents don't count — re-dropping a file you archived is a
    deliberate way to bring it back. Neither do failed ones: re-uploading is the
    normal way to retry a document whose processing broke. Shared by the upload
    view, the pre-check endpoint and the email splitter so they can't drift.
    """
    return DataRoomDocument.objects.filter(
        data_room=data_room, is_archived=False
    ).exclude(status=DataRoomDocument.Status.FAILED)


def duplicate_in_data_room(data_room, sha: str):
    """Existing live document in *data_room* with identical bytes, or None."""
    if not sha:
        return None
    return live_documents_qs(data_room).filter(content_sha256=sha).order_by("id").first()


def in_flight_count(user) -> int:
    """How many of *user*'s documents are non-terminal right now, across all their
    data rooms ("in flight" = uploaded/processing/scanning or auto-retrying — what
    still consumes worker + Redis capacity)."""
    from documents.services.pii_scan import SCAN_DISPATCH_RETRY_MESSAGE

    Status = DataRoomDocument.Status
    return DataRoomDocument.objects.filter(uploaded_by=user).filter(
        Q(status__in=[Status.UPLOADED, Status.PROCESSING, Status.SCANNING])
        | Q(status=Status.SCAN_FAILED, processing_error=SCAN_DISPATCH_RETRY_MESSAGE)
    ).count()


def in_flight_remaining(user) -> int:
    cap = getattr(settings, "DOCUMENT_MAX_IN_FLIGHT_PER_USER", 100)
    return max(0, cap - in_flight_count(user))


def check_file(user, filename: str, size: int, mime: str) -> str | None:
    """Per-file upload checks, in the upload view's order. Returns the user-facing
    reason (without the filename prefix) when the file is refused, else None.

    *filename* is the already-sanitized name.
    """
    from core.file_types import is_image_extension
    from llm.transcription_registry import AUDIO_EXTENSIONS

    ext = file_extension(filename)
    is_audio = ext in AUDIO_EXTENSIONS
    if size <= 0:
        return "file is empty."
    # Audio gets its own cap; the generic document cap must not also apply,
    # or AUDIO_UPLOAD_MAX_SIZE_BYTES could never exceed the document cap.
    if is_audio:
        audio_max_size = getattr(settings, "AUDIO_UPLOAD_MAX_SIZE_BYTES", 50_000_000)
        if size > audio_max_size:
            return f"audio file is too large (max {audio_max_size / 1_000_000:.0f} MB)."
    else:
        max_size = getattr(settings, "DOCUMENT_UPLOAD_MAX_SIZE_BYTES", 50_000_000)
        if size > max_size:
            return f"file is too large (max {max_size / 1_000_000:.0f} MB)."
    if not allowed_extension(filename):
        return "unsupported file type."
    if not mime_matches_extension(ext, mime or ""):
        return "file content doesn't match its extension."
    if is_audio:
        from core.preferences import get_preferences

        if not get_preferences(user).allowed_transcription_models:
            return "audio transcription is not enabled for your organization."
    if is_image_extension(ext):
        from core.preferences import feature_is_available

        if not feature_is_available(user, "document_image_description"):
            return "image uploads require a vision-capable model, which isn't enabled for your organization."
    return None


def queue_new_document(doc, *, tags: dict | None = None):
    """Create v0 eagerly and put it in the dispatch queue (the document dispatch
    gate hands at most DOCUMENT_WORKER_SLOTS versions to the worker at once — see
    documents.services.dispatch). The caller runs ``safe_dispatch`` once after its
    batch. A broker blip doesn't fail the document: the row simply stays queued
    and the beat backstop dispatches it. Only a genuine DB failure writing the
    queue row marks the document FAILED (returns None).

    *tags* are written on v0 before it is queued, so processing always sees them.
    Returns the queued version.
    """
    from documents.models import DataRoomDocumentTag
    from documents.services.dispatch import mark_version_queued
    from documents.services.process_document import ensure_initial_version

    try:
        version = ensure_initial_version(doc)
        if tags:
            DataRoomDocumentTag.objects.bulk_create([
                DataRoomDocumentTag(version=version, key=k, value=str(v)[:255]) for k, v in tags.items()
            ])
        mark_version_queued(version.id)
        return version
    except Exception as exc:
        logger.exception("queue_new_document: failed to queue processing for document_id=%s", doc.id)
        doc.status = DataRoomDocument.Status.FAILED
        doc.processing_error = str(exc)[:2000]
        doc.save(update_fields=["status", "processing_error", "updated_at"])
        return None
