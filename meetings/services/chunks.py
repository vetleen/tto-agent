"""Helpers for writing transcription chunks to temp files and recomputing
the denormalized ``Meeting.transcript`` field after segments arrive.

On Heroku the web dyno (Daphne/WS consumer) and the Celery worker dyno have
separate ephemeral filesystems, so writing a chunk to local disk on the web
dyno and then passing the path to a Celery task would result in
``FileNotFoundError`` on the worker.

When Django's default storage is a remote backend (S3), chunks are persisted
to shared storage.  The Celery task downloads the chunk to a local temp file
before transcribing and cleans up both local and remote copies afterwards.
When the default storage is local filesystem (dev), behaviour is unchanged.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.db import transaction
from django.utils import timezone

logger = logging.getLogger(__name__)


_MIME_TO_EXT = {
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/ogg": "ogg",
    "audio/ogg;codecs=opus": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/flac": "flac",
}

# Storage key prefix for meeting audio chunks (used in remote storage).
_CHUNK_STORAGE_PREFIX = "_meeting_chunks"


def _ext_for_mime(mime: str) -> str:
    if not mime:
        return "webm"
    normalized = mime.lower().split(";")[0].strip()
    return _MIME_TO_EXT.get(normalized) or _MIME_TO_EXT.get(mime.lower(), "webm")


def _safe_uuid_dir(meeting_uuid) -> str:
    """Refuse anything that is not a valid UUID-like path component."""
    s = str(meeting_uuid)
    if not re.fullmatch(r"[0-9a-fA-F-]{8,64}", s):
        raise ValueError(f"unsafe meeting uuid: {s!r}")
    return s


def _storage_key(meeting_uuid, segment_index: int, mime: str) -> str:
    """Return the Django storage key for a chunk."""
    uuid_dir = _safe_uuid_dir(meeting_uuid)
    ext = _ext_for_mime(mime)
    return f"{_CHUNK_STORAGE_PREFIX}/{uuid_dir}/{segment_index:06d}.{ext}"


def _uses_remote_storage() -> bool:
    """True when the default storage backend is remote (e.g. S3)."""
    backend = getattr(settings, "STORAGES", {}).get("default", {}).get("BACKEND", "")
    return "s3" in backend.lower() or "gcloud" in backend.lower() or "azure" in backend.lower()


def write_chunk_to_temp(meeting_uuid, segment_index: int, raw_bytes: bytes, mime: str) -> str:
    """Persist a single audio chunk so the Celery worker can read it.

    When the default storage is remote (S3), the chunk is saved there and
    the returned string is the storage key.  When local, it's written to
    ``MEETING_CHUNK_TEMP_DIR`` and the returned string is the absolute path.

    Caller is responsible for calling ``cleanup_temp`` after transcription.
    """
    if not isinstance(segment_index, int) or segment_index < 0:
        raise ValueError(f"invalid segment_index: {segment_index!r}")

    if _uses_remote_storage():
        key = _storage_key(meeting_uuid, segment_index, mime)
        default_storage.save(key, ContentFile(raw_bytes))
        return key

    # Local filesystem path (dev / single-dyno).
    base = Path(getattr(settings, "MEETING_CHUNK_TEMP_DIR", ""))
    if not str(base):
        raise RuntimeError("MEETING_CHUNK_TEMP_DIR is not configured")
    target_dir = base / _safe_uuid_dir(meeting_uuid)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{segment_index:06d}.{_ext_for_mime(mime)}"
    with open(target, "wb") as f:
        f.write(raw_bytes)
    return str(target)


def download_chunk_to_local(storage_path: str, mime: str = "") -> Path:
    """Download a chunk from storage to a local temp file.

    If *storage_path* is already an existing local path, returns it as-is.
    Otherwise downloads from Django's default storage to a temp file and
    returns the local ``Path``.
    """
    local = Path(storage_path)
    if local.exists():
        return local

    # Must be a remote storage key — download it.
    ext = _ext_for_mime(mime)
    tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False, prefix="meet_chunk_")
    tmp.close()
    local_tmp = Path(tmp.name)
    try:
        with default_storage.open(storage_path, "rb") as src:
            with open(local_tmp, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)
    except Exception:
        local_tmp.unlink(missing_ok=True)
        raise
    return local_tmp


def cleanup_temp(path) -> None:
    """Best-effort delete of a temp file (local and remote)."""
    # Delete local file if it exists.
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("cleanup_temp: failed to delete local %s: %s", path, exc)

    # Delete from remote storage if it looks like a storage key.
    try:
        if not Path(path).is_absolute() and default_storage.exists(path):
            default_storage.delete(path)
    except Exception as exc:
        logger.warning("cleanup_temp: failed to delete from storage %s: %s", path, exc)


def recompute_meeting_transcript(meeting_id: int) -> str:
    """Rebuild ``Meeting.transcript`` from all READY segments, in order.

    Uses ``select_for_update`` on the meeting row to serialize concurrent
    segment writes from multiple Celery workers. Returns the new transcript
    string. Touches updated_at via the standard auto_now path.
    """
    from meetings.models import Meeting, MeetingTranscriptSegment

    with transaction.atomic():
        try:
            meeting = Meeting.objects.select_for_update().get(pk=meeting_id)
        except Meeting.DoesNotExist:
            return ""

        segments = list(
            MeetingTranscriptSegment.objects
            .filter(meeting_id=meeting_id, status=MeetingTranscriptSegment.Status.READY)
            .order_by("segment_index")
            .values_list("text", flat=True)
        )
        joined = "\n\n".join(s for s in segments if s)
        meeting.transcript = joined
        meeting.transcript_updated_at = timezone.now()
        meeting.save(update_fields=["transcript", "transcript_updated_at", "updated_at"])
        return joined
