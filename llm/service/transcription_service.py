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
# The shared SDK client lives in llm.service.openai_client (also used by the
# decisions service); the old private name stays for callers and tests.
from llm.service.openai_client import get_openai_client as _get_openai_client  # noqa: F401
from llm.service.pricing import calculate_transcription_cost
from llm.transcription_registry import (
    canonical_transcription_model_id,
    get_transcription_model_info,
)
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
        keywords: list[str] | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file, splitting if needed.

        Returns a TranscriptionResult with text, duration, cost, and segment count.
        Logs each API call to LLMCallLog.

        ``prompt`` is forwarded to the OpenAI transcription API and biases the
        model toward proper nouns / vocabulary in the prompt. Empty / None means
        no prompt is sent.

        ``keywords`` are literal terms expected in the audio (names, products),
        sent as the API's ``keywords`` list. Entries must be single-line and
        free of ``<`` / ``>`` (callers sanitize; see
        ``meetings.services.audio_transcription.build_transcription_keywords``).

        ``language`` is an optional ISO-639-1 code (e.g. ``"en"``, ``"no"``). When
        set, skips the model's language detection pass and forces output in that
        language. When ``None``, the model auto-detects per request.

        ``on_delta`` enables output streaming for models with
        ``supports_output_streaming=True``. Each text delta is passed to the
        callback as it arrives from the server. The final text is still
        returned in the result. Ignored for models that don't support
        streaming.
        """
        if context is None:
            context = RunContext.create()

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Resolve to the model that serves the file endpoint (a retired ID's
        # successor, a live model's counterpart); logs carry the resolved ID.
        model_id = canonical_transcription_model_id(model_id, path="file") or model_id
        info = get_transcription_model_info(model_id)
        if info is None:
            raise ValueError(f"Unknown transcription model: {model_id}")
        if not info.supports_file_transcription:
            raise ValueError(f"Model {model_id} does not support file transcription")

        file_size = file_path.stat().st_size
        max_size = getattr(settings, "AUDIO_UPLOAD_MAX_SIZE_BYTES", info.max_file_size_bytes)
        if file_size > max_size:
            raise ValueError(
                f"Audio file too large ({file_size:,} bytes). "
                f"Maximum allowed: {max_size:,} bytes."
            )

        client = _get_openai_client()
        started_at = time.perf_counter()
        api_limit = info.max_file_size_bytes
        max_duration = info.max_duration_seconds

        # Probe duration once: reject unreadable files early and decide
        # whether the file needs splitting — avoids sending corrupted
        # audio to the transcription API.
        probe_ms: int | None = None
        if _ffmpeg_available():
            from llm.service._audio_subprocess import ffprobe_duration_ms
            probe_ms = ffprobe_duration_ms(file_path)
            if probe_ms is None:
                raise ValueError(
                    f"Audio file is corrupted or unreadable: {file_path.name}"
                )

        needs_split = file_size > api_limit or (
            probe_ms is not None and max_duration > 0 and probe_ms > max_duration * 1000
        )

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
                    language=language, on_delta=on_delta, keywords=keywords,
                )
            else:
                result = self._transcribe_single(
                    client, file_path, model_id, info, context, file_size,
                    prompt=prompt, language=language, on_delta=on_delta, keywords=keywords,
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
        keywords: list[str] | None = None,
    ):
        """Transcribe a single file directly."""
        t0 = time.perf_counter()
        response = self._call_create(
            client, file_path, info, prompt, language=language, on_delta=on_delta, keywords=keywords,
        )
        duration_ms = int((time.perf_counter() - t0) * 1000)
        text = _extract_text_from_response(response)
        audio_duration = _get_audio_duration_seconds(file_path)

        # Bill the seconds the API reports (gpt-transcribe returns a duration
        # usage); fall back to our own ffprobe duration when it doesn't.
        input_tokens, output_tokens, total_tokens, audio_tokens = _extract_transcription_usage(response)
        cost = calculate_transcription_cost(
            model_id, _extract_billed_seconds(response) or audio_duration,
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
        keywords: list[str] | None = None,
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
                    language=language, on_delta=on_delta, keywords=keywords,
                )

                seg_duration_ms = int((time.perf_counter() - t0) * 1000)
                text = _extract_text_from_response(response)
                audio_duration = _get_audio_duration_seconds(seg_path)

                input_tokens, output_tokens, total_tokens, audio_tokens = _extract_transcription_usage(response)
                cost = calculate_transcription_cost(
                    model_id, _extract_billed_seconds(response) or audio_duration,
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
        keywords: list[str] | None = None,
    ):
        """Call the OpenAI transcription API with optional prompt + graceful fallback.

        Branches on the model's capability flags (set in the registry):

        * ``supports_output_streaming=True`` AND ``on_delta`` set — opens a
          streaming response, forwards each ``transcript.text.delta`` event to
          the callback, returns the final ``.done`` event.
        * Otherwise — plain ``response_format="json"``.

        Other kwargs:

        * ``chunking_strategy="auto"`` — server-side segmentation of long
          files (avoids output truncation), for models that accept it.
        * ``languages`` — the expected spoken language as a one-item list
          (the current models replaced the singular ``language`` field, and
          reject a request carrying both). Omitted for auto-detect.
        * ``keywords`` — literal terms expected in the audio.

        If the API rejects the request because the prompt or the keywords are
        invalid (BadRequestError whose message mentions 'prompt' / 'keyword'),
        retry once with that field stripped.
        """
        kwargs: dict = {"model": info.api_model, "response_format": "json"}

        # Driven by the registry capability flag so a new model can't silently
        # miss it (or be sent one it rejects).
        if info.supports_chunking_strategy:
            kwargs["chunking_strategy"] = "auto"

        if prompt:
            kwargs["prompt"] = prompt

        if language:
            kwargs["languages"] = [language]

        if keywords:
            kwargs["keywords"] = list(keywords)

        use_stream = bool(on_delta) and info.supports_output_streaming
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
            if not (prompt or keywords):
                raise
            from openai import BadRequestError  # local import to keep top of file clean
            if not isinstance(exc, BadRequestError):
                raise
            message = str(exc).lower()
            for marker, kwarg in (("prompt", "prompt"), ("keyword", "keywords")):
                if kwarg in kwargs and marker in message:
                    logger.warning(
                        "TranscriptionService: API rejected %s (%s); retrying without it",
                        kwarg, exc,
                    )
                    kwargs.pop(kwarg)
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
            # Build a stable response object instead of mutating the SDK event:
            # some SDK events are frozen/slotted, so the old setattr(text)
            # silently failed and the whole transcript was lost. Prefer the done
            # event's own text, falling back to the assembled deltas, and carry
            # over the usage the event exposes.
            assembled = "".join(full_text_parts)
            return _SynthesizedResponse(
                text=getattr(final_event, "text", None) or assembled,
                usage=getattr(final_event, "usage", None),
            )

        # No done event — synthesize a minimal response object.
        assembled = "".join(full_text_parts)
        return _SynthesizedResponse(text=assembled)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


@dataclass
class _SynthesizedResponse:
    """Stable response shape for the streaming path.

    Used both when there is no ``transcript.text.done`` event and when there is
    one — we never mutate the SDK event object (it may be frozen). Mirrors the
    ``.text`` / ``.usage`` attributes the callers read.
    """
    text: str
    usage: dict | None = None


def _extract_text_from_response(response) -> str:
    """Pull the transcript text from a (plain or synthesized) response."""
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
        counts and an optional ``input_token_details.audio_tokens`` breakdown
        (the retired gpt-4o transcription models).
      - ``UsageDuration`` (``type="duration"``) — only carries audio seconds
        (``gpt-transcribe``; read by ``_extract_billed_seconds``). Returned
        as all-``None`` here.
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


def _extract_billed_seconds(response) -> Optional[float]:
    """Seconds of audio a ``UsageDuration`` response says were billed.

    ``None`` for token usage, missing usage, or a MagicMock in tests, so the
    caller falls back to its own (ffprobe) duration.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        utype, seconds = usage.get("type"), usage.get("seconds")
    else:
        utype, seconds = getattr(usage, "type", None), getattr(usage, "seconds", None)
    if utype != "duration" or isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    return float(seconds) if seconds > 0 else None


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Return True if ffmpeg and ffprobe are on PATH."""
    from llm.service._audio_subprocess import ffmpeg_available
    return ffmpeg_available()



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
        logger.warning("could not compute audio duration for %s", file_path, exc_info=True)
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
