"""Transcription model registry.

Single source of truth for transcription model metadata: pricing, file size
limits, and supported audio formats.

Pricing notes
-------------
OpenAI's newer transcription models (``gpt-4o-transcribe`` /
``gpt-4o-mini-transcribe``) bill by token and return a populated
``response.usage`` object. When those token counts are available we compute
cost as ``input_tokens * input_price + output_tokens * output_price``. When
the provider returns no usage (older models, partial failures, etc.) we fall
back to ``audio_duration * price_per_minute`` — the per-minute rate here is
OpenAI's own documented *estimate* on their pricing page, so the two paths
stay within shouting distance of each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class TranscriptionModelInfo:
    display_name: str
    provider: str  # "openai"
    api_model: str  # name sent to API
    price_per_minute: Decimal  # duration-based fallback (also OpenAI's own estimate)
    # Token-based billing (per 1M tokens). Both must be set to enable token-based
    # cost; otherwise the cost helper falls back to price_per_minute. OpenAI does
    # not differentiate text vs audio input on the pricing page, so a single
    # input rate covers both.
    input_price_per_1m_tokens: Optional[Decimal] = None
    output_price_per_1m_tokens: Optional[Decimal] = None
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
        # OpenAI pricing page: $2.50 / 1M input tokens, $10.00 / 1M output tokens,
        # with a documented $0.006 / minute *estimate* (used only when a response
        # arrives without usage info — gpt-4o-transcribe normally returns usage).
        price_per_minute=Decimal("0.006"),
        input_price_per_1m_tokens=Decimal("2.50"),
        output_price_per_1m_tokens=Decimal("10.00"),
    ),
    "openai/gpt-4o-mini-transcribe": TranscriptionModelInfo(
        display_name="GPT-4o Mini Transcribe",
        provider="openai",
        api_model="gpt-4o-mini-transcribe",
        # OpenAI pricing page: $1.25 / 1M input tokens, $5.00 / 1M output tokens,
        # with a documented $0.003 / minute estimate.
        price_per_minute=Decimal("0.003"),
        input_price_per_1m_tokens=Decimal("1.25"),
        output_price_per_1m_tokens=Decimal("5.00"),
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
