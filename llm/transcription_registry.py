"""Transcription model registry.

Single source of truth for transcription model metadata: pricing, file size
limits, and supported audio formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class TranscriptionModelInfo:
    display_name: str
    provider: str  # "openai"
    api_model: str  # name sent to API
    price_per_minute: Decimal
    max_file_size_bytes: int = 25_000_000  # 25 MB (OpenAI API limit)
    max_duration_seconds: int = 1400  # OpenAI API limit
    supported_formats: frozenset[str] = frozenset(
        {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac", "ogg"}
    )


_TRANSCRIPTION_MODELS: dict[str, TranscriptionModelInfo] = {
    "openai/gpt-4o-transcribe": TranscriptionModelInfo(
        display_name="GPT-4o Transcribe",
        provider="openai",
        api_model="gpt-4o-transcribe",
        price_per_minute=Decimal("0.06"),
    ),
    "openai/gpt-4o-mini-transcribe": TranscriptionModelInfo(
        display_name="GPT-4o Mini Transcribe",
        provider="openai",
        api_model="gpt-4o-mini-transcribe",
        price_per_minute=Decimal("0.03"),
    ),
}

# Union of all supported audio formats across all transcription models.
AUDIO_EXTENSIONS: frozenset[str] = frozenset().union(
    *(m.supported_formats for m in _TRANSCRIPTION_MODELS.values())
)


def get_transcription_model_info(model_id: str) -> TranscriptionModelInfo | None:
    """Look up metadata for a transcription model. Returns None for unknown models."""
    return _TRANSCRIPTION_MODELS.get(model_id)


def get_all_transcription_models() -> dict[str, TranscriptionModelInfo]:
    """Return all registered transcription models."""
    return dict(_TRANSCRIPTION_MODELS)


__all__ = [
    "TranscriptionModelInfo",
    "get_transcription_model_info",
    "get_all_transcription_models",
    "AUDIO_EXTENSIONS",
]
