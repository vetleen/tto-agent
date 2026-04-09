"""Transcription service with cost tracking and observability.

Handles audio file transcription via the OpenAI API, with automatic
splitting for files that exceed per-request size or duration limits.
Logs all calls to LLMCallLog for unified cost tracking.
"""

from __future__ import annotations

import logging
import math
import tempfile
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

from django.conf import settings

from llm.service.errors import LLMProviderError
from llm.service.logger import log_transcription, log_transcription_error
from llm.service.pricing import calculate_transcription_cost
from llm.transcription_registry import get_transcription_model_info
from llm.types.context import RunContext

logger = logging.getLogger(__name__)


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
    ) -> TranscriptionResult:
        """Transcribe an audio file, splitting if needed.

        Returns a TranscriptionResult with text, duration, cost, and segment count.
        Logs each API call to LLMCallLog.

        ``prompt`` is forwarded to the OpenAI transcription API and biases the
        model toward proper nouns / vocabulary in the prompt. Empty / None means
        no prompt is sent.
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
            "TranscriptionService.transcribe: run_id=%s model=%s file_size=%d prompt_len=%d",
            run_id, model_id, file_size, len(prompt) if prompt else 0,
        )

        try:
            if needs_split:
                result = self._transcribe_chunked(
                    client, file_path, model_id, info, context, file_size,
                    api_limit, max_duration, prompt=prompt,
                )
            else:
                result = self._transcribe_single(
                    client, file_path, model_id, info, context, file_size, prompt=prompt,
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

    def _transcribe_single(self, client, file_path, model_id, info, context, file_size, prompt: str | None = None):
        """Transcribe a single file directly."""
        t0 = time.perf_counter()
        response = self._call_create(client, file_path, info, prompt)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        text = response.text if hasattr(response, "text") else str(response)
        audio_duration = _get_audio_duration_seconds(file_path)
        cost = calculate_transcription_cost(model_id, audio_duration)

        log_transcription(
            model=model_id,
            context=context,
            audio_duration_seconds=audio_duration,
            transcript_len=len(text),
            cost_usd=cost,
            duration_ms=duration_ms,
            file_size=file_size,
            segments=1,
        )

        return TranscriptionResult(
            text=text,
            model=model_id,
            audio_duration_seconds=audio_duration,
            cost_usd=cost,
            segments=1,
        )

    def _transcribe_chunked(self, client, file_path, model_id, info, context, file_size, api_limit, max_duration, prompt: str | None = None):
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

                response = self._call_create(client, seg_path, info, prompt)

                seg_duration_ms = int((time.perf_counter() - t0) * 1000)
                text = response.text if hasattr(response, "text") else str(response)
                audio_duration = _get_audio_duration_seconds(seg_path)
                cost = calculate_transcription_cost(model_id, audio_duration)

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

    def _call_create(self, client, file_path: Path, info, prompt: str | None):
        """Call the OpenAI transcription API with optional prompt + graceful fallback.

        If the API rejects the request because the prompt is invalid (BadRequestError
        whose message mentions 'prompt'), retry once with prompt stripped. This avoids
        preemptive truncation and lets the API tell us when it can't accept the prompt.
        """
        kwargs = {
            "model": info.api_model,
            # `verbose_json` is rejected by gpt-4o-transcribe / gpt-4o-mini-transcribe
            # (only whisper-1 supports it). We use plain `json` and compute audio
            # duration locally for cost tracking.
            "response_format": "json",
        }
        if prompt:
            kwargs["prompt"] = prompt

        try:
            with open(file_path, "rb") as f:
                return client.audio.transcriptions.create(file=f, **kwargs)
        except Exception as exc:
            # Detect prompt-related BadRequest and retry without the prompt.
            if not prompt:
                raise
            from openai import BadRequestError  # local import to keep top of file clean
            if isinstance(exc, BadRequestError) and "prompt" in str(exc).lower():
                logger.warning(
                    "TranscriptionService: API rejected prompt (%s); retrying without prompt",
                    exc,
                )
                kwargs.pop("prompt", None)
                with open(file_path, "rb") as f:
                    return client.audio.transcriptions.create(file=f, **kwargs)
            raise


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _audio_exceeds_duration(file_path: Path, max_seconds: int) -> bool:
    """Return True if the audio file is longer than *max_seconds*."""
    if max_seconds <= 0:
        return False
    try:
        from pydub import AudioSegment
    except ImportError:
        return False
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) > max_seconds * 1000
    except Exception:
        return False


def _get_audio_duration_seconds(file_path: Path) -> float:
    """Best-effort audio duration in seconds, used for cost tracking.

    Returns 0.0 if pydub is unavailable or the file cannot be parsed. We accept
    a 0.0 fallback because the alternative (failing the whole transcription
    because we can't compute cost) is worse than logging a $0 cost.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        return 0.0
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        logger.warning("could not compute audio duration for %s", file_path)
        return 0.0


def _split_audio_file(
    file_path: Path,
    max_segment_bytes: int,
    max_segment_seconds: int = 0,
) -> list[Path]:
    """Split an audio file into MP3 segments respecting size and duration limits.

    Returns a list of Path objects to temporary files. Caller must delete them.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is required for splitting long or large audio files. "
            "Install it with: pip install pydub"
        )

    audio = AudioSegment.from_file(file_path)
    total_ms = len(audio)

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
    for i in range(num_segments):
        start = i * segment_ms
        end = min((i + 1) * segment_ms, total_ms)
        chunk = audio[start:end]

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False, prefix=f"transcribe_seg{i}_"
        )
        tmp.close()
        seg_path = Path(tmp.name)
        chunk.export(str(seg_path), format="mp3")
        segment_paths.append(seg_path)

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
