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
from .services.transcript_cleanup import collapse_repetitions

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
        from django.db.models.functions import Right

        from .services.audio_transcription import (
            LIVE_PROMPT_TAIL_CHARS,
            build_transcription_prompt,
        )
        try:
            # Defer the (potentially hundreds of KB) transcript column and fetch
            # only its last LIVE_PROMPT_TAIL_CHARS via SQL — the full column never
            # leaves Postgres just to build a 1200-char prompt tail.
            meeting = (
                Meeting.objects
                .defer("transcript")
                .annotate(transcript_tail=Right("transcript", LIVE_PROMPT_TAIL_CHARS))
                .get(pk=meeting_id)
            )
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
        # proper nouns and jargon across chunks. (transcript_tail is the last
        # LIVE_PROMPT_TAIL_CHARS, fetched via SQL above.)
        prior_tail = (meeting.transcript_tail or "") or None
        prompt = build_transcription_prompt(meeting, prior_tail=prior_tail)

        # Effective language: the meeting's own choice, else the resolved
        # user/org/system default. None = auto-detect. (Historically this
        # chunked-live path passed no language at all.)
        from core.languages import effective_meeting_language
        from core.preferences import get_preferences

        prefs = get_preferences(user) if user else None
        language = effective_meeting_language(
            meeting.forced_language,
            getattr(prefs, "transcription_language", "auto"),
        )

        local_path = download_chunk_to_local(temp_path, mime)
        try:
            text = transcribe_audio(local_path, model_id, user, prompt=prompt, language=language)
        except Exception as exc:
            from .services.errors import classify_transcription_error, log_unmapped

            classified = classify_transcription_error(exc)
            # Log at the classified level (undecodable chunks stay WARNING so they
            # don't storm Sentry; unknown failures keep the full stack trace).
            # The raw exception only goes to logs — the user sees user_message.
            log_fn = getattr(logger, classified.log_level, logger.error)
            log_fn(
                "transcribe_meeting_chunk_task: transcription failed for meeting=%s "
                "segment=%s error_code=%s",
                meeting_id, segment_index, classified.error_code,
                exc_info=(classified.log_level == "error"),
            )
            log_unmapped(classified, exc, context="chunk_task")
            MeetingTranscriptSegment.objects.filter(pk=segment.pk).update(
                status=MeetingTranscriptSegment.Status.FAILED,
                error=classified.user_message,
                transcribed_at=timezone.now(),
            )
            _push_to_ws(meeting_uuid, {
                "type": "segment.failed",
                "segment_index": segment_index,
                "error": classified.user_message,
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
            # De-looped for live display; raw text was stored on the segment row.
            "text": collapse_repetitions(text or ""),
            "start_offset_seconds": start_offset_seconds,
            "transcription_model": model_id,
        })
    finally:
        cleanup_temp(temp_path)


# NOTE: no Celery autoretry. The temp audio file is unlinked in the finally
# block below, so a Celery retry would FileNotFoundError immediately. Per-chunk
# transient retries (network flake / 429s) happen inside the orchestrator at
# the right level — see meetings/services/audio_transcription.py.
#
# Time limits are sized for the worst legal upload, not the typical one: the
# 50 MB byte cap admits many hours of low-bitrate audio (e.g. ~7h at 16 kbps),
# which the orchestrator processes as dozens of sequential ~5-min chunks. 3h
# wall clock covers that with margin. If the soft limit still fires,
# SoftTimeLimitExceeded surfaces inside the current chunk call, the
# orchestrator persists the partial transcript, and the classifier maps it to
# a friendly "ran out of time" message — no stranded LIVE_TRANSCRIBING row.
@shared_task(
    time_limit=10800,
    soft_time_limit=10500,
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

    local_path = None
    try:
        try:
            # Inside the try so a storage/download failure still hits the
            # defensive FAILED update below and the finally's cleanup_temp,
            # instead of stranding the meeting in LIVE_TRANSCRIBING and leaking
            # the uploaded audio.
            local_path = download_chunk_to_local(temp_path)
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
            from .services.errors import classify_transcription_error, log_unmapped

            classified = classify_transcription_error(exc)
            logger.exception("transcribe_uploaded_audio_task: failed for meeting %s", meeting_id)
            log_unmapped(classified, exc, context="upload_task")
            Meeting.objects.filter(
                pk=meeting_id,
            ).exclude(
                status=Meeting.Status.FAILED,
            ).update(
                status=Meeting.Status.FAILED,
                transcription_error=classified.user_message,
                ended_at=timezone.now(),
                transcription_chunks_total=0,
                transcription_chunks_done=0,
            )
            # Do NOT re-raise — there is no useful retry path (file is gone
            # in finally) and Celery would just log a redundant traceback.
            return
    finally:
        if local_path is not None and str(local_path) != temp_path:
            local_path.unlink(missing_ok=True)
        cleanup_temp(temp_path)


# Sweeper staleness thresholds. Chunk segments transcribe within the chunk
# task's 600s hard limit, so 15 min is safely past it. Upload staleness keys
# off Meeting.updated_at, which a healthy upload run refreshes continuously
# (per-chunk progress writes plus sub-second partial-transcript flushes), so
# 60 min of *silence* means the task is dead — the threshold does not need to
# cover the upload task's multi-hour wall-clock ceiling.
STALE_SEGMENT_MINUTES = 15
STALE_UPLOAD_MINUTES = 60
# Live-path staleness. The WS heartbeat/disconnect normally move a live meeting
# out of LIVE_TRANSCRIBING, but a hard dyno death (SIGKILL, Daphne crash) skips
# disconnect() and strands it — which then blocks the per-user transcription
# gate for ALL of that user's meetings. 6h is comfortably past the 4h
# MEETING_AUTO_STOP_MAX_SECONDS session ceiling, so a healthy live session is
# never swept mid-recording.
STALE_LIVE_HOURS = 6


@shared_task(time_limit=60)
def expire_stale_transcriptions() -> int:
    """Periodic recovery of transcription work stranded by a worker restart.

    Celery acks tasks early, so a dyno restart (Heroku cycles dynos daily)
    silently drops in-flight transcription tasks:

    - ``MeetingTranscriptSegment`` rows stay PENDING forever. Re-enqueueing is
      impossible — the chunk temp file was deleted (or lost with the dyno) —
      so stale segments are marked FAILED, with a best-effort WS push so an
      open live session shows the gap.
    - Upload-path meetings stay LIVE_TRANSCRIBING forever, which also blocks
      the per-user "already transcribing" gate in meetings.views. Stale ones
      are marked FAILED (mirrors the defensive update in
      ``transcribe_uploaded_audio_task``).
    - Live-path meetings are normally finalized by the WS consumer
      (Stop/disconnect set READY/INTERRUPTED), but a hard dyno death skips that,
      so live-source rows older than ``STALE_LIVE_HOURS`` (beyond the session
      ceiling) are marked INTERRUPTED — the transcript content is valid, so this
      is not a FAILED.

    Like chat.tasks.expire_stale_subagent_runs, transient DB unavailability is
    logged at WARNING and skipped — the next beat tick retries.
    """
    from datetime import timedelta

    from django.db.utils import InterfaceError, OperationalError

    from .models import Meeting, MeetingTranscriptSegment

    now = timezone.now()
    handled = 0

    try:
        stale_segments = list(
            MeetingTranscriptSegment.objects.filter(
                status=MeetingTranscriptSegment.Status.PENDING,
                created_at__lt=now - timedelta(minutes=STALE_SEGMENT_MINUTES),
            ).values_list("pk", "segment_index", "meeting__uuid")
        )
        if stale_segments:
            handled += MeetingTranscriptSegment.objects.filter(
                pk__in=[s[0] for s in stale_segments],
                status=MeetingTranscriptSegment.Status.PENDING,
            ).update(
                status=MeetingTranscriptSegment.Status.FAILED,
                error="Transcription was interrupted.",
                transcribed_at=now,
            )
            logger.warning(
                "expire_stale_transcriptions: %s segment(s) stuck in PENDING marked FAILED",
                len(stale_segments),
            )
            for _pk, segment_index, meeting_uuid in stale_segments:
                _push_to_ws(str(meeting_uuid), {
                    "type": "segment.failed",
                    "segment_index": segment_index,
                    "error": "Transcription was interrupted.",
                })

        stale_uploads = Meeting.objects.filter(
            status=Meeting.Status.LIVE_TRANSCRIBING,
            transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
            updated_at__lt=now - timedelta(minutes=STALE_UPLOAD_MINUTES),
        ).update(
            status=Meeting.Status.FAILED,
            transcription_error=(
                "Transcription was interrupted. Please upload the audio again."
            ),
            ended_at=now,
            transcription_chunks_total=0,
            transcription_chunks_done=0,
        )
        if stale_uploads:
            logger.warning(
                "expire_stale_transcriptions: %s upload meeting(s) stuck in "
                "LIVE_TRANSCRIBING marked FAILED",
                stale_uploads,
            )
            handled += stale_uploads

        stale_live = Meeting.objects.filter(
            status=Meeting.Status.LIVE_TRANSCRIBING,
            transcript_source=Meeting.TranscriptSource.LIVE,
            updated_at__lt=now - timedelta(hours=STALE_LIVE_HOURS),
        ).update(
            status=Meeting.Status.INTERRUPTED,
            transcription_error="Transcription was interrupted.",
            ended_at=now,
        )
        if stale_live:
            logger.warning(
                "expire_stale_transcriptions: %s live meeting(s) stuck in "
                "LIVE_TRANSCRIBING marked INTERRUPTED",
                stale_live,
            )
            handled += stale_live
    except (OperationalError, InterfaceError):
        logger.warning(
            "Skipping stale transcription sweep: database temporarily unavailable; "
            "will retry on next beat tick.",
            exc_info=True,
        )
        return 0

    return handled
