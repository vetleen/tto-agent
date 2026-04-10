"""Celery tasks for the meetings app.

- ``transcribe_meeting_chunk_task`` — single live-transcription chunk: takes a
  temp audio file written by ``MeetingTranscribeConsumer``, transcribes it via
  the existing TranscriptionService, persists a ``MeetingTranscriptSegment``,
  recomputes the denormalized ``Meeting.transcript``, and pushes the result
  back to the WS group ``meetings.<uuid>`` via channel_layers.
  Audio is ALWAYS deleted in a ``finally:`` block, regardless of outcome.
- ``transcribe_uploaded_audio_task`` — single-shot for the audio-upload path.
  Uses the same ``transcribe_audio`` helper, writes to ``Meeting.transcript``
  directly, and deletes the audio file afterwards.
"""
from __future__ import annotations

import logging
from pathlib import Path

from celery import shared_task
from django.utils import timezone

from .services.chunks import cleanup_temp, download_chunk_to_local, recompute_meeting_transcript

logger = logging.getLogger(__name__)


def _channel_group(meeting_uuid) -> str:
    return f"meetings.{meeting_uuid}"


def _push_to_ws(meeting_uuid, payload: dict) -> None:
    """Best-effort group_send. Failures are logged but never raised."""
    try:
        from asgiref.sync import async_to_sync
        from channels.layers import get_channel_layer

        layer = get_channel_layer()
        if layer is None:
            return
        async_to_sync(layer.group_send)(_channel_group(meeting_uuid), payload)
    except Exception:
        logger.exception("meetings: failed to push WS event for meeting %s", meeting_uuid)


# NOTE: no autoretry. The temp audio file is unconditionally deleted in the
# `finally:` block below, so a Celery retry would just re-run with a missing
# file and fail with FileNotFoundError — masking the real first-attempt error.
# If we want retries on transient API failures later, we need to keep the file
# alive across attempts (skip cleanup until success or final failure).
@shared_task(
    time_limit=600,
    soft_time_limit=540,
)
def transcribe_meeting_chunk_task(
    meeting_id: int,
    segment_index: int,
    temp_path: str,
    mime: str,
    model_id: str,
    user_id: int,
    start_offset_seconds: float = 0.0,
) -> None:
    from django.contrib.auth import get_user_model

    from documents.services.transcription import transcribe_audio

    from .models import Meeting, MeetingTranscriptSegment

    User = get_user_model()
    meeting_uuid = None
    try:
        try:
            meeting = Meeting.objects.get(pk=meeting_id)
            meeting_uuid = str(meeting.uuid)
        except Meeting.DoesNotExist:
            logger.warning("transcribe_meeting_chunk_task: meeting %s not found", meeting_id)
            return

        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            user = None

        # get_or_create the segment row in PENDING state.
        segment, _created = MeetingTranscriptSegment.objects.get_or_create(
            meeting=meeting,
            segment_index=segment_index,
            defaults={
                "start_offset_seconds": start_offset_seconds,
                "transcription_model": model_id,
                "status": MeetingTranscriptSegment.Status.PENDING,
            },
        )

        # Build a transcription prompt from meeting metadata + tail of the
        # already-transcribed transcript so the model has continuity for
        # proper nouns and jargon across chunks.
        from .services.audio_transcription import (
            LIVE_PROMPT_TAIL_CHARS,
            build_transcription_prompt,
        )
        prior_tail = (meeting.transcript or "")[-LIVE_PROMPT_TAIL_CHARS:] or None
        prompt = build_transcription_prompt(meeting, prior_tail=prior_tail)

        local_path = download_chunk_to_local(temp_path, mime)
        try:
            text = transcribe_audio(local_path, model_id, user, prompt=prompt)
        except Exception as exc:
            err = str(exc)[:1000]
            logger.exception(
                "transcribe_meeting_chunk_task: transcription failed for meeting=%s segment=%s",
                meeting_id, segment_index,
            )
            MeetingTranscriptSegment.objects.filter(pk=segment.pk).update(
                status=MeetingTranscriptSegment.Status.FAILED,
                error=err,
                transcribed_at=timezone.now(),
            )
            _push_to_ws(meeting_uuid, {
                "type": "segment.failed",
                "segment_index": segment_index,
                "error": err,
            })
            # Don't re-raise: the failure is already recorded on the segment
            # and pushed to the WS. Re-raising here would only escalate to
            # Celery, which has no useful retry path (file is deleted in
            # `finally:` below) and the new error would just clobber `err`.
            return
        finally:
            # Clean up the downloaded local temp file (may differ from
            # the storage key when chunks are stored remotely).
            if str(local_path) != temp_path:
                local_path.unlink(missing_ok=True)

        MeetingTranscriptSegment.objects.filter(pk=segment.pk).update(
            text=text or "",
            status=MeetingTranscriptSegment.Status.READY,
            transcribed_at=timezone.now(),
            transcription_model=model_id,
        )

        # Stamp the model on the meeting too (first segment wins).
        Meeting.objects.filter(pk=meeting_id, transcription_model="").update(
            transcription_model=model_id,
        )

        recompute_meeting_transcript(meeting_id)

        _push_to_ws(meeting_uuid, {
            "type": "segment.ready",
            "segment_index": segment_index,
            "text": text or "",
            "start_offset_seconds": start_offset_seconds,
            "transcription_model": model_id,
        })
    finally:
        cleanup_temp(temp_path)


# NOTE: no Celery autoretry. The temp audio file is unlinked in the finally
# block below, so a Celery retry would FileNotFoundError immediately. Per-chunk
# transient retries (network flake / 429s) happen inside the orchestrator at
# the right level — see meetings/services/audio_transcription.py.
@shared_task(
    time_limit=1800,
    soft_time_limit=1740,
)
def transcribe_uploaded_audio_task(
    meeting_id: int,
    temp_path: str,
    model_id: str,
    user_id: int,
) -> None:
    """Transcribe an uploaded meeting audio file.

    Delegates to ``orchestrate_upload_transcription`` which handles overlap
    splitting, sequential per-chunk transcription with prompt carryover,
    fuzzy stitching, and progress field updates. The orchestrator also
    finalizes the Meeting row (status=READY/FAILED, transcript, etc.). This
    outer wrapper exists only to (a) catch any pre-orchestrator crash and
    mark the meeting failed defensively, and (b) unlink the original
    uploaded temp file in finally regardless of outcome.
    """
    from .models import Meeting
    from .services.audio_transcription import orchestrate_upload_transcription

    local_path = download_chunk_to_local(temp_path)
    try:
        try:
            orchestrate_upload_transcription(
                meeting_id=meeting_id,
                temp_path=local_path,
                model_id=model_id,
                user_id=user_id,
            )
        except Exception as exc:
            # Defensive: if the orchestrator already finalized the meeting as
            # FAILED with a per-chunk error message, this update is a no-op
            # for the meaningful fields. If the failure happened BEFORE the
            # orchestrator could set its own error (e.g. pydub blew up on
            # load, or the meeting row went missing), this is the only place
            # the meeting will be marked failed.
            err = str(exc)[:1000]
            logger.exception("transcribe_uploaded_audio_task: failed for meeting %s", meeting_id)
            Meeting.objects.filter(
                pk=meeting_id,
            ).exclude(
                status=Meeting.Status.FAILED,
            ).update(
                status=Meeting.Status.FAILED,
                transcription_error=err,
                ended_at=timezone.now(),
                transcription_chunks_total=0,
                transcription_chunks_done=0,
            )
            # Do NOT re-raise — there is no useful retry path (file is gone
            # in finally) and Celery would just log a redundant traceback.
            return
    finally:
        if str(local_path) != temp_path:
            local_path.unlink(missing_ok=True)
        cleanup_temp(temp_path)
