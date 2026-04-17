"""Transcription service with cost tracking and observability.

Handles audio file transcription via the OpenAI API, with automatic
splitting for files that exceed per-request size or duration limits.
Logs all calls to LLMCallLog for unified cost tracking.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

from django.conf import settings

from llm.service.errors import LLMProviderError
from llm.service.logger import log_transcription, log_transcription_error
from llm.service.pricing import calculate_transcription_cost
from llm.transcription_registry import get_transcription_model_info
from llm.types.context import RunContext

logger = logging.getLogger(__name__)

DeltaCallback = Callable[[str], None]


@dataclass
class TranscriptionResult:
    """Result of a transcription call."""

    text: str
    model: str
    audio_duration_seconds: float
    cost_usd: Optional[Decimal]
    segments: int


class TranscriptionService:
    """Transcription service with cost tracking and observability."""

    def transcribe(
        self,
        file_path: Path,
        model_id: str,
        context: RunContext | None = None,
        prompt: str | None = None,
        language: str | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file, splitting if needed.

        Returns a TranscriptionResult with text, duration, cost, and segment count.
        Logs each API call to LLMCallLog.

        ``prompt`` is forwarded to the OpenAI transcription API and biases the
        model toward proper nouns / vocabulary in the prompt. Empty / None means
        no prompt is sent.

        ``language`` is an optional ISO-639-1 code (e.g. ``"en"``, ``"no"``). When
        set, skips the model's language detection pass and forces output in that
        language. When ``None``, the model auto-detects per request.

        ``on_delta`` enables output streaming for models with
        ``supports_output_streaming=True``. Each text delta is passed to the
        callback as it arrives from the server. The final text is still
        returned in the result. Ignored for models that don't support
        streaming (whisper-1, diarize).
        """
        if context is None:
            context = RunContext.create()

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        info = get_transcription_model_info(model_id)
        if info is None:
            raise ValueError(f"Unknown transcription model: {model_id}")

        file_size = file_path.stat().st_size
        max_size = getattr(settings, "AUDIO_UPLOAD_MAX_SIZE_BYTES", info.max_file_size_bytes)
        if file_size > max_size:
            raise ValueError(
                f"Audio file too large ({file_size:,} bytes). "
                f"Maximum allowed: {max_size:,} bytes."
            )

        from openai import OpenAI

        client = OpenAI()
        started_at = time.perf_counter()
        api_limit = info.max_file_size_bytes
        max_duration = info.max_duration_seconds
        needs_split = file_size > api_limit or _audio_exceeds_duration(file_path, max_duration)

        run_id = context.run_id
        logger.info(
            "TranscriptionService.transcribe: run_id=%s model=%s file_size=%d prompt_len=%d lang=%s stream=%s",
            run_id, model_id, file_size, len(prompt) if prompt else 0,
            language or "-", bool(on_delta and info.supports_output_streaming),
        )

        try:
            if needs_split:
                result = self._transcribe_chunked(
                    client, file_path, model_id, info, context, file_size,
                    api_limit, max_duration, prompt=prompt,
                    language=language, on_delta=on_delta,
                )
            else:
                result = self._transcribe_single(
                    client, file_path, model_id, info, context, file_size,
                    prompt=prompt, language=language, on_delta=on_delta,
                )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            log_transcription_error(model_id, context, exc, duration_ms, file_size)
            raise LLMProviderError(f"Transcription failed: {exc}") from exc

        total_duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "TranscriptionService.transcribe: run_id=%s model=%s segments=%d "
            "audio_duration=%.1fs transcript_len=%d wall_time_ms=%d",
            run_id, model_id, result.segments,
            result.audio_duration_seconds, len(result.text), total_duration_ms,
        )

        return result

    def _transcribe_single(
        self, client, file_path, model_id, info, context, file_size,
        prompt: str | None = None,
        language: str | None = None,
        on_delta: DeltaCallback | None = None,
    ):
        """Transcribe a single file directly."""
        t0 = time.perf_counter()
        response = self._call_create(client, file_path, info, prompt, language=language, on_delta=on_delta)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        text = _extract_text_from_response(response, info)
        audio_duration = _get_audio_duration_seconds(file_path)

        # Prefer token counts reported by the API over per-minute estimates.
        # gpt-4o-transcribe / gpt-4o-mini-transcribe return a UsageTokens; the
        # legacy whisper-1 path returns a UsageDuration (no token counts) and
        # we fall through to the per-minute formula.
        input_tokens, output_tokens, total_tokens, audio_tokens = _extract_transcription_usage(response)
        cost = calculate_transcription_cost(
            model_id, audio_duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        log_transcription(
            model=model_id,
            context=context,
            audio_duration_seconds=audio_duration,
            transcript_len=len(text),
            cost_usd=cost,
            duration_ms=duration_ms,
            file_size=file_size,
            segments=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            audio_tokens=audio_tokens,
        )

        return TranscriptionResult(
            text=text,
            model=model_id,
            audio_duration_seconds=audio_duration,
            cost_usd=cost,
            segments=1,
        )

    def _transcribe_chunked(
        self, client, file_path, model_id, info, context, file_size, api_limit, max_duration,
        prompt: str | None = None,
        language: str | None = None,
        on_delta: DeltaCallback | None = None,
    ):
        """Split file into segments and transcribe each.

        This is the legacy fallback path: it's only entered when a caller hands
        the service a file that exceeds the API's per-request limits without
        pre-splitting. The meetings upload orchestrator pre-splits with overlap
        and stitches results, so it should never enter this branch. If it does
        WITH a prompt set, log a warning — that signals the orchestrator's
        chunking math was wrong.
        """
        if prompt:
            logger.warning(
                "TranscriptionService._transcribe_chunked: entered with prompt set "
                "(file_size=%d, model=%s) — caller should pre-split for prompt-aware chunking",
                file_size, model_id,
            )
        segment_paths: list[Path] = []
        try:
            segment_paths = _split_audio_file(file_path, api_limit, max_duration)
            logger.info(
                "TranscriptionService: splitting file_size=%d into %d segments",
                file_size, len(segment_paths),
            )

            transcripts = []
            total_audio_duration = 0.0
            total_cost = Decimal("0")

            for i, seg_path in enumerate(segment_paths):
                seg_size = seg_path.stat().st_size
                t0 = time.perf_counter()

                logger.info(
                    "TranscriptionService: transcribing segment %d/%d size=%d",
                    i + 1, len(segment_paths), seg_size,
                )

                response = self._call_create(
                    client, seg_path, info, prompt,
                    language=language, on_delta=on_delta,
                )

                seg_duration_ms = int((time.perf_counter() - t0) * 1000)
                text = _extract_text_from_response(response, info)
                audio_duration = _get_audio_duration_seconds(seg_path)

                input_tokens, output_tokens, total_tokens, audio_tokens = _extract_transcription_usage(response)
                cost = calculate_transcription_cost(
                    model_id, audio_duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

                total_audio_duration += audio_duration
                if cost is not None:
                    total_cost += cost

                log_transcription(
                    model=model_id,
                    context=context,
                    audio_duration_seconds=audio_duration,
                    transcript_len=len(text),
                    cost_usd=cost,
                    duration_ms=seg_duration_ms,
                    file_size=seg_size,
                    segments=len(segment_paths),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    audio_tokens=audio_tokens,
                )

                transcripts.append(text)

            return TranscriptionResult(
                text=" ".join(transcripts),
                model=model_id,
                audio_duration_seconds=total_audio_duration,
                cost_usd=total_cost if total_cost > 0 else None,
                segments=len(segment_paths),
            )
        finally:
            for p in segment_paths:
                p.unlink(missing_ok=True)

    def _call_create(
        self,
        client,
        file_path: Path,
        info,
        prompt: str | None,
        *,
        language: str | None = None,
        on_delta: DeltaCallback | None = None,
    ):
        """Call the OpenAI transcription API with optional prompt + graceful fallback.

        Branches on the model's capability flags (set in the registry):

        * ``supports_diarization=True`` — uses ``response_format="diarized_json"``
          (skips streaming since the diarize endpoint only returns a full
          formatted response).
        * ``supports_output_streaming=True`` AND ``on_delta`` set — opens a
          streaming response, forwards each ``transcript.text.delta`` event to
          the callback, returns the final ``.done`` event.
        * Otherwise — plain ``response_format="json"`` as before.

        Other kwargs added for newer models:

        * ``chunking_strategy="auto"`` — required by diarize on inputs >30s,
          recommended by OpenAI for ``gpt-4o-transcribe`` / ``-mini`` to avoid
          the ~8-minute output truncation bug. Skipped for whisper-1.
        * ``language`` — optional ISO-639-1 code. When set, skips the 30-second
          language detection pass.

        If the API rejects the request because the prompt is invalid (BadRequestError
        whose message mentions 'prompt'), retry once with prompt stripped.
        """
        kwargs: dict = {"model": info.api_model}

        if info.supports_diarization:
            # Diarize endpoint only supports diarized_json; streaming not available.
            kwargs["response_format"] = "diarized_json"
        else:
            # `verbose_json` is rejected by gpt-4o-transcribe / gpt-4o-mini-transcribe
            # (only whisper-1 supports it). We use plain `json` and compute audio
            # duration locally for cost tracking.
            kwargs["response_format"] = "json"

        # chunking_strategy is only supported by the gpt-4o transcription family;
        # whisper-1 rejects it. Required for diarize on >30s input; for 4o/4o-mini
        # it prevents output truncation on segments >8 minutes.
        if info.api_model.startswith("gpt-4o"):
            kwargs["chunking_strategy"] = "auto"

        if prompt and not info.supports_diarization:
            # Diarize rejects the prompt parameter per OpenAI docs.
            kwargs["prompt"] = prompt

        if language:
            kwargs["language"] = language

        use_stream = bool(on_delta) and info.supports_output_streaming and not info.supports_diarization
        if use_stream:
            kwargs["stream"] = True

        def _do_call():
            with open(file_path, "rb") as f:
                if use_stream:
                    return self._consume_streaming(
                        client.audio.transcriptions.create(file=f, **kwargs),
                        on_delta,
                    )
                return client.audio.transcriptions.create(file=f, **kwargs)

        try:
            return _do_call()
        except Exception as exc:
            if not prompt:
                raise
            from openai import BadRequestError  # local import to keep top of file clean
            if isinstance(exc, BadRequestError) and "prompt" in str(exc).lower():
                logger.warning(
                    "TranscriptionService: API rejected prompt (%s); retrying without prompt",
                    exc,
                )
                kwargs.pop("prompt", None)
                return _do_call()
            raise

    def _consume_streaming(self, event_stream, on_delta: DeltaCallback | None):
        """Drain a streaming transcription response, forwarding deltas.

        Returns the final response object from the ``transcript.text.done``
        event, which carries the full text and (on newer SDKs) usage. The
        event objects are typed by the SDK; we access them defensively in
        case OpenAI tweaks the shape.
        """
        final_event = None
        full_text_parts: list[str] = []
        for event in event_stream:
            etype = getattr(event, "type", None)
            if etype == "transcript.text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    full_text_parts.append(delta)
                    if on_delta is not None:
                        try:
                            on_delta(delta)
                        except Exception:
                            # A faulty callback must not break transcription.
                            logger.exception("on_delta callback raised")
            elif etype == "transcript.text.done":
                final_event = event
                # Some SDK versions carry the full text here; keep both for safety.
                final_text = getattr(event, "text", None)
                if final_text and not full_text_parts:
                    full_text_parts.append(final_text)

        if final_event is not None:
            # Reuse the done event as the "response" — it has .text and .usage
            # on the SDKs we care about. If .text is missing/short, fall back
            # to the assembled delta buffer.
            assembled = "".join(full_text_parts)
            if not getattr(final_event, "text", None):
                # Attach assembled text as a dynamic attribute for downstream.
                try:
                    setattr(final_event, "text", assembled)
                except Exception:
                    pass
            return final_event

        # No done event — synthesize a minimal response object.
        assembled = "".join(full_text_parts)
        return _SynthesizedResponse(text=assembled)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


@dataclass
class _SynthesizedResponse:
    """Fallback response shape when the streaming path produces no `.done` event."""
    text: str


def _extract_text_from_response(response, info) -> str:
    """Pull the transcript text from a response, handling all branches.

    For diarized responses we flatten ``response.segments`` into
    ``Speaker A: ...`` lines; for plain responses we return ``response.text``.
    """
    if info.supports_diarization:
        segments = getattr(response, "segments", None)
        if segments:
            lines: list[str] = []
            for seg in segments:
                speaker = getattr(seg, "speaker", None)
                text = (getattr(seg, "text", "") or "").strip()
                if not text:
                    continue
                if speaker:
                    lines.append(f"Speaker {speaker}: {text}")
                else:
                    lines.append(text)
            if lines:
                return "\n".join(lines)
        # Fall through — diarize responses usually also carry a flat .text.
    if hasattr(response, "text"):
        return response.text or ""
    return str(response)


def _extract_transcription_usage(
    response,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Pull ``(input_tokens, output_tokens, total_tokens, audio_tokens)`` from
    a transcription response, or a 4-tuple of ``None`` when usage is absent.

    The OpenAI SDK returns a ``response.usage`` discriminated union:
      - ``UsageTokens`` (``type="tokens"``) — carries input/output/total token
        counts and an optional ``input_token_details.audio_tokens`` breakdown.
        Emitted by ``gpt-4o-transcribe`` and ``gpt-4o-mini-transcribe``.
      - ``UsageDuration`` (``type="duration"``) — whisper-1 / legacy shape,
        only carries audio seconds. Returned as all-``None`` here so the
        caller falls back to per-minute billing.
    Missing usage (e.g. a MagicMock in tests) also yields all-``None``.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None, None
    # `type` distinguishes the two variants. Require a real str match so a
    # MagicMock (which auto-vivifies attributes in tests) falls through.
    utype = getattr(usage, "type", None)
    if not isinstance(utype, str) or utype != "tokens":
        return None, None, None, None

    def _int_or_none(value) -> Optional[int]:
        return value if isinstance(value, int) else None

    details = getattr(usage, "input_token_details", None)
    audio_tokens = (
        _int_or_none(getattr(details, "audio_tokens", None))
        if details is not None
        else None
    )
    return (
        _int_or_none(getattr(usage, "input_tokens", None)),
        _int_or_none(getattr(usage, "output_tokens", None)),
        _int_or_none(getattr(usage, "total_tokens", None)),
        audio_tokens,
    )


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Return True if ffmpeg and ffprobe are on PATH."""
    from llm.service._audio_subprocess import ffmpeg_available
    return ffmpeg_available()


def _audio_exceeds_duration(file_path: Path, max_seconds: int) -> bool:
    """Return True if the audio file is longer than *max_seconds*."""
    if max_seconds <= 0:
        return False
    if not _ffmpeg_available():
        return False
    try:
        from llm.service._audio_subprocess import ffprobe_duration_ms
        duration = ffprobe_duration_ms(file_path)
        if duration is None:
            return False
        return duration > max_seconds * 1000
    except Exception:
        return False


def _get_audio_duration_seconds(file_path: Path) -> float:
    """Best-effort audio duration in seconds, used for cost tracking.

    Returns 0.0 if ffprobe is unavailable or the file cannot be parsed.
    We accept a 0.0 fallback because the alternative (failing the whole
    transcription because we can't compute cost) is worse than logging a $0
    cost.
    """
    if not _ffmpeg_available():
        return 0.0
    try:
        from llm.service._audio_subprocess import ffprobe_duration_ms
        duration_ms = ffprobe_duration_ms(file_path)
        if duration_ms is None:
            return 0.0
        return duration_ms / 1000.0
    except Exception:
        logger.info("could not compute audio duration for %s", file_path)
        return 0.0


def _split_audio_file(
    file_path: Path,
    max_segment_bytes: int,
    max_segment_seconds: int = 0,
) -> list[Path]:
    """Split an audio file into MP3 segments respecting size and duration limits.

    Returns a list of Path objects to temporary files. Caller must delete them.
    """
    from llm.service._audio_subprocess import (
        ffmpeg_available,
        ffmpeg_extract_chunk,
        ffprobe_duration_ms,
    )

    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg/ffprobe is required for splitting large audio files but is "
            "not installed. Install ffmpeg to enable splitting files that exceed "
            f"the {max_segment_bytes / 1_000_000:.0f} MB API limit."
        )

    duration_ms = ffprobe_duration_ms(file_path)
    if duration_ms is None or duration_ms <= 0:
        raise RuntimeError(
            f"Could not determine audio duration for {file_path}. "
            "The file may be corrupted or in an unsupported format."
        )
    total_ms = duration_ms

    file_size = file_path.stat().st_size
    safe_limit = int(max_segment_bytes * 0.8)
    segments_by_size = math.ceil(file_size / safe_limit) if file_size > safe_limit else 1

    if max_segment_seconds > 0:
        max_ms = max_segment_seconds * 1000
        segments_by_duration = math.ceil(total_ms / max_ms)
    else:
        segments_by_duration = 1

    num_segments = max(segments_by_size, segments_by_duration)
    segment_ms = math.ceil(total_ms / num_segments)

    segment_paths: list[Path] = []
    try:
        for i in range(num_segments):
            start = i * segment_ms
            end = min((i + 1) * segment_ms, total_ms)
            seg_path = ffmpeg_extract_chunk(
                file_path, start, end, i,
                output_prefix="transcribe_seg",
            )
            segment_paths.append(seg_path)
    except Exception:
        for p in segment_paths:
            p.unlink(missing_ok=True)
        raise

    return segment_paths


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_service: TranscriptionService | None = None
_global_service_lock = threading.Lock()


def get_transcription_service() -> TranscriptionService:
    """Return the process-wide TranscriptionService singleton (thread-safe)."""
    global _global_service
    if _global_service is None:
        with _global_service_lock:
            if _global_service is None:
                _global_service = TranscriptionService()
    return _global_service
