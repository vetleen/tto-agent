"""Classify transcription failures into friendly, user-facing messages.

Mirrors the LLM provider pattern in ``llm/core/providers/base.py`` (the
``ClassifiedError`` / ``classify_api_error`` / ``_highlight_if_unmapped``
trio). The goal is the same: never surface a raw ``str(exc)`` to the client,
keep a curated message per known failure mode, and emit a distinct,
Sentry-actionable log line whenever an error hits the ``unknown`` catch-all so
we can incrementally grow the mapping.

Used by the meetings views, the live WebSocket consumer, the per-chunk Celery
task, and the upload orchestrator — every place that previously interpolated an
exception into a user-visible string or stored it on ``Meeting.transcription_error``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassifiedTranscriptionError:
    """Result of classifying a transcription error."""

    error_code: str
    user_message: str
    log_level: str  # "warning" or "error"


# Generic fallback message reused by the catch-all branch.
_GENERIC_MESSAGE = "Transcription failed unexpectedly. Please try again."


def classify_transcription_error(exc: Exception) -> ClassifiedTranscriptionError:
    """Inspect an exception and return a classified, user-facing error.

    Branches are ordered most-specific first. Unknown errors fall through to
    the ``unknown`` catch-all — callers should pass the result to
    :func:`log_unmapped` so each new failure mode is flagged in Sentry.
    """
    status = getattr(exc, "status_code", None)
    msg = str(exc)
    msg_lower = msg.lower()
    exc_name = type(exc).__name__

    # Transient: rate limiting. Match the OpenAI exception class by name (so we
    # don't hard-depend on the import) plus the HTTP 429 status.
    if exc_name == "RateLimitError" or status == 429:
        return ClassifiedTranscriptionError(
            error_code="rate_limited",
            user_message="Transcription is rate-limited right now. Please wait a moment and try again.",
            log_level="warning",
        )

    # Transient: cannot reach the provider.
    if exc_name == "APIConnectionError" or isinstance(exc, (ConnectionError, TimeoutError)):
        return ClassifiedTranscriptionError(
            error_code="connection_error",
            user_message="Couldn't reach the transcription service. Please try again.",
            log_level="warning",
        )

    # Undecodable / unreadable audio. The downstream transcription helper raises
    # ValueError("...corrupted or unreadable...") for chunks ffmpeg/ffprobe
    # can't parse; treat as a benign input error, not an alert.
    if "corrupted or unreadable" in msg_lower or "could not be read" in msg_lower:
        return ClassifiedTranscriptionError(
            error_code="undecodable_audio",
            user_message="That audio couldn't be read. Try a different file or format.",
            log_level="warning",
        )

    # ffmpeg split hung and was killed by the orchestrator's watchdog.
    if exc_name == "AudioSplitTimeoutError":
        return ClassifiedTranscriptionError(
            error_code="split_timeout",
            user_message="The audio took too long to process. Try a shorter file.",
            log_level="warning",
        )

    # File too large for the transcription API (raised by the splitter when
    # ffmpeg is unavailable, or when a chunk exceeds the safe byte limit).
    if "api limit" in msg_lower or "too large" in msg_lower:
        return ClassifiedTranscriptionError(
            error_code="too_large",
            user_message="That audio file is too large to transcribe.",
            log_level="warning",
        )

    # Catch-all: a failure we have no curated message for yet.
    return ClassifiedTranscriptionError(
        error_code="unknown",
        user_message=_GENERIC_MESSAGE,
        log_level="error",
    )


def log_unmapped(
    classified: ClassifiedTranscriptionError,
    exc: Exception,
    *,
    context: str,
) -> None:
    """Emit a Sentry-actionable line when an error hits the ``unknown`` catch-all.

    These are failures we have no curated user-facing message for yet. The
    dedicated ERROR line (grouped in Sentry by ``exc_type``) flags each new
    failure mode so we can add a branch to :func:`classify_transcription_error`
    with a friendly message and resolve the Sentry issue — incrementally growing
    the mapping. No-op for already-mapped errors.

    ``context`` is a short caller tag (e.g. ``"chunk_task"``, ``"upload"``) so
    the same unmapped exception type is distinguishable by code path.
    """
    if classified.error_code != "unknown":
        return
    logger.error(
        "Unmapped transcription error — add a branch to classify_transcription_error "
        "so the user gets a specific message. context=%s exc_type=%s",
        context,
        type(exc).__name__,
        exc_info=True,
    )


__all__ = [
    "ClassifiedTranscriptionError",
    "classify_transcription_error",
    "log_unmapped",
]
