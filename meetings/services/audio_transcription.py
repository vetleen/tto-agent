"""Upload-path transcription orchestrator for the meetings app.

This module owns the *upload-side* complexity that the shared
``TranscriptionService`` deliberately doesn't carry:

* Builds a transcription prompt from a ``Meeting``'s metadata
  (name, agenda, participants, description) so the model is biased toward
  proper nouns and meeting-specific jargon. **No** linked data rooms or
  attachments are read — only the meeting's own metadata fields.
* Splits long uploads into overlapping chunks (default 15 s overlap) so
  word-boundary cuts at chunk seams can be recovered later.
* Processes chunks sequentially with prompt carryover (the previous
  chunk's transcript tail becomes context for the next chunk's prompt).
* Stitches consecutive chunk transcripts back together using a bounded
  fuzzy match against the overlap region.
* Streams progress to the UI via ``Meeting.transcription_chunks_total`` /
  ``transcription_chunks_done`` so the polling endpoint can show a bar.

The live-recording path (``meetings/consumers.py`` +
``transcribe_meeting_chunk_task``) does NOT use the orchestrator — it just
re-uses :func:`build_transcription_prompt` to seed each independently
transcribed chunk.
"""
from __future__ import annotations

import difflib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from django.utils import timezone

from llm.transcription_registry import get_transcription_model_info
from llm.types.context import RunContext

if TYPE_CHECKING:
    from meetings.models import Meeting

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — see plan file for rationale
# ---------------------------------------------------------------------------

DEFAULT_TARGET_CHUNK_SECONDS = 900   # 15 min target chunk length
DEFAULT_OVERLAP_SECONDS = 15         # audio overlap between consecutive chunks
DEFAULT_PRIOR_TAIL_CHARS = 800       # transcript tail length used in prompt carryover
LIVE_PROMPT_TAIL_CHARS = 1200        # tail length for live-path prompt seeding
TRANSIENT_RETRY_BACKOFFS = (1.0, 3.0)  # per-chunk transient retry sleep schedule
CHARS_PER_OVERLAP_SECOND = 15        # ~150 wpm * ~6 chars/word; used by stitcher
AUDIO_SPLIT_TIMEOUT_SECONDS = 120    # safety net: kill the split if ffmpeg hangs


class AudioSplitTimeoutError(RuntimeError):
    """Raised when the audio splitting operation exceeds AUDIO_SPLIT_TIMEOUT_SECONDS."""


@dataclass
class ChunkSpec:
    """A single audio chunk produced by ``split_audio_with_overlap``."""

    path: Path
    index: int
    start_ms: int
    end_ms: int


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_PROMPT_HEADER = (
    "Transcript of a business meeting. "
    "Preserve proper nouns, names, and acronyms."
)


def build_transcription_prompt(meeting: "Meeting", prior_tail: str | None = None) -> str:
    """Build a transcription prompt from a meeting's metadata.

    Only reads ``meeting.name``, ``meeting.agenda``, ``meeting.participants``,
    ``meeting.description``. Does not touch linked data rooms or attachments.
    Empty fields are skipped entirely (no empty label lines).

    ``prior_tail``, if provided, is appended as a "previous transcript excerpt"
    section that biases the model for jargon continuity across chunk seams.
    No truncation is applied — callers are responsible for keeping the tail
    bounded; the transcription service has a graceful BadRequestError fallback
    if the API rejects the prompt.
    """
    lines: list[str] = [_PROMPT_HEADER]

    name = (getattr(meeting, "name", "") or "").strip()
    agenda = (getattr(meeting, "agenda", "") or "").strip()
    participants = (getattr(meeting, "participants", "") or "").strip()
    description = (getattr(meeting, "description", "") or "").strip()

    if name:
        lines.append(f"Meeting: {name}")
    if agenda:
        lines.append(f"Agenda: {agenda}")
    if participants:
        lines.append(f"Participants: {participants}")
    if description:
        lines.append(f"Notes: {description}")

    text = "\n".join(lines)

    tail = (prior_tail or "").strip()
    if tail:
        text += (
            "\n\nPrevious transcript excerpt (for context only, do not repeat):\n"
            + tail
        )

    return text


# ---------------------------------------------------------------------------
# Audio splitter with overlap
# ---------------------------------------------------------------------------


def split_audio_with_overlap(
    file_path: Path,
    *,
    target_chunk_seconds: int = DEFAULT_TARGET_CHUNK_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    max_bytes: int,
    max_seconds: int,
) -> list[ChunkSpec]:
    """Split *file_path* into overlapping MP3 chunks ready for transcription.

    Each chunk i covers audio range
    ``[max(0, i*effective - overlap_ms), min(total, (i+1)*effective)]``,
    so chunk i contains ``overlap_seconds`` of audio from the tail of
    chunk i-1. The first chunk has no leading overlap; the last chunk
    is clamped to the file end.

    Returns an empty list if the audio is shorter than the smallest possible
    chunk. Returns a single-element list if the file fits in one chunk
    (no overlap needed). Caller is responsible for unlinking the temp files.

    Raises ValueError if ``overlap_seconds * 2 >= target_chunk_seconds``
    (degenerate overlap configuration).
    """
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")
    if target_chunk_seconds <= overlap_seconds * 2:
        raise ValueError(
            f"target_chunk_seconds ({target_chunk_seconds}) must be greater than "
            f"2 * overlap_seconds ({overlap_seconds * 2})"
        )

    from llm.service._audio_subprocess import (
        ffmpeg_available,
        ffmpeg_extract_chunk,
        ffprobe_duration_ms,
    )

    if not ffmpeg_available():
        file_size = file_path.stat().st_size
        if file_size > max_bytes:
            raise RuntimeError(
                f"Audio file is {file_size / 1_000_000:.1f} MB which exceeds "
                f"the {max_bytes / 1_000_000:.0f} MB API limit, and ffmpeg is "
                f"not installed to split it into smaller chunks."
            )
        return [ChunkSpec(path=file_path, index=0, start_ms=0, end_ms=0)]

    duration_ms = ffprobe_duration_ms(file_path)
    if duration_ms is None:
        logger.warning(
            "ffprobe could not determine duration for %s; falling back to single chunk",
            file_path,
        )
        file_size = file_path.stat().st_size
        if file_size > max_bytes:
            raise RuntimeError(
                f"Audio file is {file_size / 1_000_000:.1f} MB which exceeds "
                f"the {max_bytes / 1_000_000:.0f} MB API limit, and ffprobe "
                f"could not probe the file to split it into smaller chunks."
            )
        return [ChunkSpec(path=file_path, index=0, start_ms=0, end_ms=0)]

    total_ms = duration_ms
    if total_ms <= 0:
        return []

    # Effective chunk duration: respect both the user-supplied target and the
    # API duration cap, leaving headroom for the trailing overlap on each chunk.
    safe_seconds = max(0, max_seconds - overlap_seconds - 60)
    effective_seconds = min(target_chunk_seconds, safe_seconds) if safe_seconds else target_chunk_seconds
    if effective_seconds <= overlap_seconds:
        # Pathological max_seconds — fall back to target.
        effective_seconds = target_chunk_seconds
    effective_ms = effective_seconds * 1000
    overlap_ms = overlap_seconds * 1000

    # Single-chunk fast path: file fits inside a single API-legal chunk.
    if total_ms <= effective_ms:
        path = ffmpeg_extract_chunk(
            file_path, 0, total_ms, 0,
            output_prefix="meet_upload_seg",
        )
        _validate_chunk_size(path, 0, max_bytes)
        return [ChunkSpec(path=path, index=0, start_ms=0, end_ms=total_ms)]

    num_chunks = math.ceil(total_ms / effective_ms)
    specs: list[ChunkSpec] = []
    try:
        for i in range(num_chunks):
            raw_start = i * effective_ms
            start = max(0, raw_start - overlap_ms) if i > 0 else 0
            end = min((i + 1) * effective_ms, total_ms)
            if end <= start:
                break
            path = ffmpeg_extract_chunk(
                file_path, start, end, i,
                output_prefix="meet_upload_seg",
            )
            _validate_chunk_size(path, i, max_bytes)
            specs.append(ChunkSpec(path=path, index=i, start_ms=start, end_ms=end))
    except Exception:
        # On failure mid-split, clean up anything we already wrote.
        for s in specs:
            s.path.unlink(missing_ok=True)
        raise

    return specs


def _validate_chunk_size(path: Path, index: int, max_bytes: int) -> None:
    """Check that an exported chunk does not exceed the safe byte limit."""
    size = path.stat().st_size
    safe_bytes = int(max_bytes * 0.8)
    if size > safe_bytes:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"Audio chunk {index} too large after export: {size} bytes > {safe_bytes} safe limit"
        )


# ---------------------------------------------------------------------------
# Transcript stitcher
# ---------------------------------------------------------------------------


def stitch_transcripts(
    prev_text: str,
    next_text: str,
    *,
    expected_overlap_chars: int,
) -> str:
    """Merge two consecutive chunk transcripts whose audio ranges overlap.

    Uses a bounded ``difflib.SequenceMatcher.find_longest_match`` against the
    last ``expected_overlap_chars * 1.5`` chars of *prev_text* and the first
    ``expected_overlap_chars * 1.5`` chars of *next_text*. If a confident
    match is found (size >= ``max(40, expected_overlap_chars // 3)``), the
    output is ``prev_text`` truncated at the match start + ``next_text``
    starting from the match end. Otherwise the fallback drops the first
    ``expected_overlap_chars`` characters of ``next_text`` and concatenates
    with a single space — better to lose a few seconds than to duplicate.
    """
    if not prev_text:
        return next_text
    if not next_text:
        return prev_text
    if expected_overlap_chars <= 0:
        return prev_text.rstrip() + " " + next_text.lstrip()

    window = max(int(expected_overlap_chars * 1.5), 60)
    prev_tail = prev_text[-window:]
    next_head = next_text[:window]

    matcher = difflib.SequenceMatcher(a=prev_tail, b=next_head, autojunk=False)
    match = matcher.find_longest_match(0, len(prev_tail), 0, len(next_head))

    # Floor of 20 chars (≈4 words) is enough to avoid noise matches in a
    # bounded window while still tolerating short overlap configs. //3 of
    # the expected overlap dominates for the default 15 s setting (75 chars).
    confident_size = max(20, expected_overlap_chars // 3)
    if match.size >= confident_size:
        # Splice at the match: keep prev THROUGH the end of the match (so
        # we preserve the matching substring exactly once), and skip past
        # the match in next so we don't duplicate it.
        prev_keep = len(prev_text) - len(prev_tail) + match.a + match.size
        next_skip = match.b + match.size
        return prev_text[:prev_keep].rstrip() + " " + next_text[next_skip:].lstrip()

    # Fallback: drop expected overlap from the head of next_text.
    skip = min(expected_overlap_chars, len(next_text))
    return prev_text.rstrip() + " " + next_text[skip:].lstrip()


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def combine_existing_and_new_transcript(existing: str, new_text: str) -> str:
    """Join an existing meeting transcript with freshly transcribed text.

    Used by the orchestrator so that uploading a second audio (or text) file
    *adds to* the transcript rather than replacing it — matching the mental
    model of "Continue transcription" on the live button. Blank inputs are
    handled so the empty-transcript case behaves exactly like the pre-append
    code path (output equals ``new_text``).
    """
    prev = (existing or "").rstrip()
    nxt = (new_text or "").lstrip()
    if prev and nxt:
        return prev + "\n\n" + nxt
    return prev or nxt


def orchestrate_upload_transcription(
    meeting_id: int,
    temp_path: Path,
    model_id: str,
    user_id: int | None,
    *,
    service=None,
    target_chunk_seconds: int = DEFAULT_TARGET_CHUNK_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
) -> str:
    """Transcribe an uploaded meeting audio file end-to-end.

    Sequence:
      1. Load the Meeting and build the meta prompt.
      2. Split the audio into overlapping chunks.
      3. If single chunk: one transcription call, persist, return.
      4. Else: set chunks_total/done atomically, then iterate sequentially.
         For each chunk, build a prompt with the running tail, call the
         service with a small transient-error retry, and stitch.
      5. On chunk failure: persist any partial transcript, set FAILED,
         reset progress fields, re-raise.

    When the meeting already has a transcript (re-upload case), the newly
    transcribed text is appended to the existing transcript — never
    replacing it. The running tail used for prompt carryover includes the
    prior transcript too, so the first new chunk benefits from the existing
    transcript's proper nouns / jargon continuity.

    Returns the combined transcript (existing + new) text.
    """
    from meetings.models import Meeting

    if service is None:
        from llm.service.transcription_service import get_transcription_service
        service = get_transcription_service()

    info = get_transcription_model_info(model_id)
    if info is None:
        raise ValueError(f"Unknown transcription model: {model_id}")

    meeting = Meeting.objects.get(pk=meeting_id)
    ctx = RunContext.create(user_id=user_id)

    # Snapshot the existing transcript BEFORE any chunking work. If the user
    # is re-uploading to an existing meeting, new text is appended to this
    # snapshot rather than overwriting it.
    existing_transcript = meeting.transcript or ""

    # Build the chunks before touching any meeting state — if pydub blows up
    # we don't want partially-set progress fields lying around.
    # Wrap in a timeout so a hung ffmpeg process cannot burn the entire Celery
    # time_limit (30 min). The timeout is deliberately generous — normal splits
    # finish in single-digit seconds even for hour-long files.
    import concurrent.futures

    t0 = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                split_audio_with_overlap,
                temp_path,
                target_chunk_seconds=target_chunk_seconds,
                overlap_seconds=overlap_seconds,
                max_bytes=info.max_file_size_bytes,
                max_seconds=info.max_duration_seconds,
            )
            chunk_specs = future.result(timeout=AUDIO_SPLIT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        split_duration = time.perf_counter() - t0
        logger.error(
            "audio split timed out after %.1fs (limit=%ds) for meeting=%s file=%s",
            split_duration, AUDIO_SPLIT_TIMEOUT_SECONDS, meeting_id, temp_path,
        )
        raise AudioSplitTimeoutError(
            f"Audio splitting timed out after {AUDIO_SPLIT_TIMEOUT_SECONDS}s. "
            f"The file may be corrupted or too complex for ffmpeg to process."
        )
    split_duration = time.perf_counter() - t0
    logger.info(
        "audio split completed in %.1fs (%d chunks) for meeting=%s file=%s",
        split_duration, len(chunk_specs), meeting_id, temp_path,
    )

    if not chunk_specs:
        raise ValueError(f"Audio file produced no chunks: {temp_path}")

    expected_overlap_chars = overlap_seconds * CHARS_PER_OVERLAP_SECOND

    # Single-chunk fast path: skip overlap/progress entirely.
    if len(chunk_specs) == 1:
        spec = chunk_specs[0]
        try:
            # Seed the prompt with the tail of any existing transcript so the
            # model has continuity for proper nouns across the append boundary.
            prior_tail = (
                existing_transcript[-DEFAULT_PRIOR_TAIL_CHARS:]
                if existing_transcript
                else None
            )
            prompt = build_transcription_prompt(meeting, prior_tail=prior_tail)
            result = _transcribe_with_transient_retry(
                service, spec.path, model_id, ctx, prompt,
            )
            text = result.text or ""
            combined = combine_existing_and_new_transcript(existing_transcript, text)
            _finalize_meeting_success(meeting_id, combined, model_id)
            return combined
        finally:
            spec.path.unlink(missing_ok=True)

    # Multi-chunk path.
    n = len(chunk_specs)
    Meeting.objects.filter(pk=meeting_id).update(
        transcription_chunks_total=n,
        transcription_chunks_done=0,
        status=Meeting.Status.LIVE_TRANSCRIBING,
        transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
        transcription_error="",
        updated_at=timezone.now(),
    )

    running_new_transcript = ""  # only the *new* upload's stitched text
    try:
        for i, spec in enumerate(chunk_specs):
            # Cancellation check: if the user clicked Stop, the cancel view
            # flipped status to FAILED. Bail out with the partial transcript
            # we've stitched so far. We can't interrupt the in-flight chunk,
            # but we won't start a new one.
            current_status = Meeting.objects.filter(pk=meeting_id).values_list("status", flat=True).first()
            if current_status != Meeting.Status.LIVE_TRANSCRIBING:
                logger.info(
                    "orchestrator: upload transcription cancelled for meeting=%s after %d/%d chunks",
                    meeting_id, i, n,
                )
                partial_combined = combine_existing_and_new_transcript(
                    existing_transcript, running_new_transcript,
                )
                Meeting.objects.filter(pk=meeting_id).update(
                    transcript=partial_combined,
                    transcription_chunks_total=0,
                    transcription_chunks_done=0,
                    transcript_updated_at=timezone.now() if partial_combined else None,
                    updated_at=timezone.now(),
                )
                return partial_combined

            # Prior tail seeds prompt carryover across chunks. For the first
            # chunk we pull from the existing transcript (if any); for later
            # chunks we pull from the running new transcript. Either way we
            # never stitch the new text into the old text — we just want the
            # model to see prior context for jargon continuity.
            if i == 0:
                prior_tail = (
                    existing_transcript[-DEFAULT_PRIOR_TAIL_CHARS:]
                    if existing_transcript
                    else None
                )
            else:
                prior_tail = running_new_transcript[-DEFAULT_PRIOR_TAIL_CHARS:]
            prompt = build_transcription_prompt(meeting, prior_tail=prior_tail)

            try:
                result = _transcribe_with_transient_retry(
                    service, spec.path, model_id, ctx, prompt,
                )
            except Exception as exc:
                # Per-chunk failure: persist combined partial, mark FAILED, re-raise.
                partial_combined = combine_existing_and_new_transcript(
                    existing_transcript, running_new_transcript,
                )
                _finalize_meeting_failure(
                    meeting_id, partial_combined, i, n, exc,
                )
                raise

            chunk_text = result.text or ""
            if i == 0:
                running_new_transcript = chunk_text
            else:
                running_new_transcript = stitch_transcripts(
                    running_new_transcript,
                    chunk_text,
                    expected_overlap_chars=expected_overlap_chars,
                )

            Meeting.objects.filter(pk=meeting_id).update(
                transcript=combine_existing_and_new_transcript(
                    existing_transcript, running_new_transcript,
                ),
                transcription_chunks_done=i + 1,
                transcript_updated_at=timezone.now(),
                updated_at=timezone.now(),
            )

            spec.path.unlink(missing_ok=True)
    finally:
        for spec in chunk_specs:
            spec.path.unlink(missing_ok=True)

    combined = combine_existing_and_new_transcript(existing_transcript, running_new_transcript)
    _finalize_meeting_success(meeting_id, combined, model_id)
    return combined


def _transcribe_with_transient_retry(service, file_path: Path, model_id: str, context: RunContext, prompt: str):
    """Call ``service.transcribe`` with retries on transient OpenAI errors only.

    Retries on ``APIConnectionError`` / ``RateLimitError`` per the
    ``TRANSIENT_RETRY_BACKOFFS`` schedule. Does NOT retry ``BadRequestError``
    (those are input errors and the service layer already strips a bad
    prompt and retries inside ``_call_create``).
    """
    transient_excs: tuple[type[Exception], ...] = ()
    try:
        from openai import APIConnectionError, RateLimitError
        transient_excs = (APIConnectionError, RateLimitError)
    except ImportError:  # pragma: no cover
        pass

    attempt = 0
    last_exc: Optional[Exception] = None
    while True:
        try:
            return service.transcribe(file_path, model_id, context=context, prompt=prompt)
        except Exception as exc:
            last_exc = exc
            if transient_excs and isinstance(exc, transient_excs) and attempt < len(TRANSIENT_RETRY_BACKOFFS):
                backoff = TRANSIENT_RETRY_BACKOFFS[attempt]
                logger.warning(
                    "orchestrator: transient transcription error on attempt %d (%s); "
                    "sleeping %.1fs before retry",
                    attempt + 1, exc, backoff,
                )
                time.sleep(backoff)
                attempt += 1
                continue
            raise
    # Unreachable, but keeps type checkers happy.
    if last_exc is not None:  # pragma: no cover
        raise last_exc


def _finalize_meeting_success(meeting_id: int, text: str, model_id: str) -> None:
    from meetings.models import Meeting

    ended = timezone.now()
    meeting = Meeting.objects.filter(pk=meeting_id).only("started_at").first()
    duration = None
    if meeting and meeting.started_at:
        duration = max(0, int((ended - meeting.started_at).total_seconds()))
    Meeting.objects.filter(pk=meeting_id).update(
        transcript=text or "",
        transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
        transcription_model=model_id,
        transcription_error="",
        status=Meeting.Status.READY,
        ended_at=ended,
        duration_seconds=duration,
        transcription_chunks_total=0,
        transcription_chunks_done=0,
        transcript_updated_at=ended,
        updated_at=ended,
    )


def _finalize_meeting_failure(
    meeting_id: int,
    partial_transcript: str,
    failed_index: int,
    total_chunks: int,
    exc: BaseException,
) -> None:
    from meetings.models import Meeting

    err_msg = str(exc)[:500]
    note = f"Failed on chunk {failed_index + 1}/{total_chunks}: {err_msg}"
    if partial_transcript:
        note += " (partial transcript saved)"
    ended = timezone.now()
    Meeting.objects.filter(pk=meeting_id).update(
        transcript=partial_transcript or "",
        transcript_source=Meeting.TranscriptSource.AUDIO_UPLOAD,
        transcription_error=note,
        status=Meeting.Status.FAILED,
        ended_at=ended,
        transcription_chunks_total=0,
        transcription_chunks_done=0,
        transcript_updated_at=ended if partial_transcript else None,
        updated_at=ended,
    )
