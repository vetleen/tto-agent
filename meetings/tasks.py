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

from .services.chunks import cleanup_temp, recompute_meeting_transcript

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


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
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

        try:
            text = transcribe_audio(Path(temp_path), model_id, user)
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
            raise

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


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    time_limit=1800,
    soft_time_limit=1740,
)
def transcribe_uploaded_audio_task(
    meeting_id: int,
    temp_path: str,
    model_id: str,
    user_id: int,
) -> None:
    """Single-shot transcription for the 'upload audio' path.

    Writes the result directly to ``Meeting.transcript`` and flips the meeting
    status to READY (or FAILED). Deletes the temp audio file unconditionally.
    """
    from django.contrib.auth import get_user_model

    from documents.services.transcription import transcribe_audio

    from .models import Meeting

    User = get_user_model()
    try:
        try:
            meeting = Meeting.objects.get(pk=meeting_id)
        except Meeting.DoesNotExist:
            logger.warning("transcribe_uploaded_audio_task: meeting %s not found", meeting_id)
            return

        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            user = None

        try:
            text = transcribe_audio(Path(temp_path), model_id, user)
        except Exception as exc:
            err = str(exc)[:1000]
            logger.exception("transcribe_uploaded_audio_task: failed for meeting %s", meeting_id)
            Meeting.objects.filter(pk=meeting_id).update(
                status=Meeting.Status.FAILED,
                transcription_error=err,
                ended_at=timezone.now(),
            )
            raise

        ended = timezone.now()
        duration = None
        if meeting.started_at:
            duration = max(0, int((ended - meeting.started_at).total_seconds()))

        Meeting.objects.filter(pk=meeting_id).update(
            transcript=text or "",
            transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
            transcription_model=model_id,
            transcription_error="",
            status=Meeting.Status.READY,
            ended_at=ended,
            duration_seconds=duration,
        )
    finally:
        cleanup_temp(temp_path)
