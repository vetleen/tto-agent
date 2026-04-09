"""Helpers for writing transcription chunks to temp files and recomputing
the denormalized ``Meeting.transcript`` field after segments arrive.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from django.conf import settings
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


def write_chunk_to_temp(meeting_uuid, segment_index: int, raw_bytes: bytes, mime: str) -> Path:
    """Persist a single audio chunk to disk under MEETING_CHUNK_TEMP_DIR.

    Returns the absolute path. Caller is responsible for unlinking the file
    after the chunk has been transcribed (success or failure).
    """
    base = Path(getattr(settings, "MEETING_CHUNK_TEMP_DIR", ""))
    if not str(base):
        raise RuntimeError("MEETING_CHUNK_TEMP_DIR is not configured")
    target_dir = base / _safe_uuid_dir(meeting_uuid)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(segment_index, int) or segment_index < 0:
        raise ValueError(f"invalid segment_index: {segment_index!r}")
    target = target_dir / f"{segment_index:06d}.{_ext_for_mime(mime)}"
    with open(target, "wb") as f:
        f.write(raw_bytes)
    return target


def cleanup_temp(path) -> None:
    """Best-effort delete of a temp file. Silently ignores missing files."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("cleanup_temp: failed to delete %s: %s", path, exc)


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
