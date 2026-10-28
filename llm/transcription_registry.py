"""Transcription model registry.

Single source of truth for transcription model metadata: pricing, file size
limits, supported audio formats and which API path a model serves.

Two API paths
-------------
* ``"file"`` — ``/v1/audio/transcriptions`` (uploads, the chunked live
  fallback, data-room audio).
* ``"live"`` — the Realtime transcription socket.

OpenAI's current generation splits them across two models: ``gpt-transcribe``
(file) and ``gpt-live-transcribe`` (live). Each names the other as its
counterpart, so a caller that knows its path can resolve whichever model ID
it was handed (a meeting's stored choice, an org default, a config var) with
``canonical_transcription_model_id(model_id, path=...)``.

Retired IDs
-----------
The gpt-4o transcription family shuts down on 2027-02-26. Retired IDs are not
registered; ``TRANSCRIPTION_MODEL_REPLACEMENTS`` maps each to the successors
that cover the paths it served, so stored preferences, meeting rows and
``TRANSCRIPTION_*`` config vars that still name an old ID keep working
(mirrors ``MODEL_REPLACEMENTS`` for chat and ``IMAGE_MODEL_REPLACEMENTS``).

Pricing
-------
Both models bill per minute of audio and report ``usage`` as a duration, so
cost is ``audio_duration * price_per_minute``. Token rates stay optional for
any future model that bills by token.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Literal, Optional

TranscriptionPath = Literal["file", "live"]


@dataclass(frozen=True)
class TranscriptionModelInfo:
    display_name: str
    provider: str  # "openai"
    api_model: str  # name sent to API
    price_per_minute: Decimal  # duration-based billing
    # Token-based billing (per 1M tokens). Both must be set to enable token-based
    # cost; otherwise the cost helper uses price_per_minute.
    input_price_per_1m_tokens: Optional[Decimal] = None
    output_price_per_1m_tokens: Optional[Decimal] = None
    max_file_size_bytes: int = 25_000_000  # 25 MB (OpenAI API limit)
    max_duration_seconds: int = 1400  # per-request cap we split uploads to
    supported_formats: frozenset[str] = frozenset(
        {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "flac", "ogg"}
    )
    # Capability flags drive UI gating (which buttons to enable) and runtime
    # routing (which API code path to take). All new registry entries must set
    # these explicitly; callers should never hardcode per-model behavior.
    supports_live_streaming: bool = False       # Realtime API / live WS path
    supports_file_transcription: bool = False   # /v1/audio/transcriptions
    supports_output_streaming: bool = False     # stream=True on batch transcription
    supports_diarization: bool = False          # returns speaker-labelled segments
    # chunking_strategy="auto" — server-side segmentation of long files
    # (prevents output truncation on long segments).
    supports_chunking_strategy: bool = False


_TRANSCRIPTION_MODELS: dict[str, TranscriptionModelInfo] = {
    "openai/gpt-transcribe": TranscriptionModelInfo(
        display_name="GPT Transcribe",
        provider="openai",
        api_model="gpt-transcribe",
        # OpenAI pricing page: $0.0045 / minute; usage arrives as a duration.
        price_per_minute=Decimal("0.0045"),
        supports_file_transcription=True,
        supports_output_streaming=True,
        supports_chunking_strategy=True,
    ),
    "openai/gpt-live-transcribe": TranscriptionModelInfo(
        display_name="GPT Live Transcribe",
        provider="openai",
        api_model="gpt-live-transcribe",
        # OpenAI pricing page: $0.017 / minute of realtime audio; each
        # completed turn reports its billed seconds.
        price_per_minute=Decimal("0.017"),
        supports_live_streaming=True,
    ),
}

# The model that serves the other path for the same provider generation.
_PATH_COUNTERPARTS: dict[str, dict[str, str]] = {
    "openai/gpt-transcribe": {"live": "openai/gpt-live-transcribe"},
    "openai/gpt-live-transcribe": {"file": "openai/gpt-transcribe"},
}

# Retired ID -> its successors, covering every path the retired model served.
# Rows are never repointed; lookups follow this map. A retired ID must not
# also be registered above. Shutdown 2027-02-26 (OpenAI deprecations page).
TRANSCRIPTION_MODEL_REPLACEMENTS: dict[str, tuple[str, ...]] = {
    "openai/gpt-4o-transcribe": ("openai/gpt-transcribe", "openai/gpt-live-transcribe"),
    "openai/gpt-4o-mini-transcribe": ("openai/gpt-transcribe", "openai/gpt-live-transcribe"),
    # Speaker labels have no successor; the file model transcribes without them.
    "openai/gpt-4o-transcribe-diarize": ("openai/gpt-transcribe",),
    # Never registered here, but config vars may still name it.
    "openai/whisper-1": ("openai/gpt-transcribe",),
}

_PATH_FLAGS = {
    "file": "supports_file_transcription",
    "live": "supports_live_streaming",
}


def _serves(model_id: str, path: TranscriptionPath) -> bool:
    info = _TRANSCRIPTION_MODELS.get(model_id)
    return bool(info and getattr(info, _PATH_FLAGS[path]))


def canonical_transcription_model_id(
    model_id: str | None, *, path: TranscriptionPath | None = None,
) -> str | None:
    """Return the registered ID to use for ``model_id``.

    With ``path``, the result serves that path: a retired ID resolves to the
    successor for it, and a registered model for the other path resolves to
    its counterpart. ``None`` when nothing serves the path (or the ID is
    unknown). Without ``path``, a retired ID resolves to its first successor
    (for display and pricing of stored rows).
    """
    if not model_id:
        return None
    candidate = model_id.strip()
    if candidate in TRANSCRIPTION_MODEL_REPLACEMENTS:
        successors = TRANSCRIPTION_MODEL_REPLACEMENTS[candidate]
        if path is None:
            return successors[0]
        for successor in successors:
            if _serves(successor, path):
                return successor
        candidate = successors[0]
    if candidate not in _TRANSCRIPTION_MODELS:
        return None
    if path is None or _serves(candidate, path):
        return candidate
    counterpart = _PATH_COUNTERPARTS.get(candidate, {}).get(path)
    return counterpart if counterpart and _serves(counterpart, path) else None


def normalize_transcription_model_ids(model_ids: Iterable[str]) -> list[str]:
    """Canonicalize an allow-list: registered IDs stay, a retired ID becomes
    every successor (so an old list naming gpt-4o-transcribe still allows both
    the file and the live model), unknown IDs are dropped; order kept,
    de-duplicated."""
    result: list[str] = []
    for model_id in model_ids:
        key = (model_id or "").strip()
        if key in TRANSCRIPTION_MODEL_REPLACEMENTS:
            candidates = TRANSCRIPTION_MODEL_REPLACEMENTS[key]
        elif key in _TRANSCRIPTION_MODELS:
            candidates = (key,)
        else:
            continue
        for candidate in candidates:
            if candidate not in result:
                result.append(candidate)
    return result


def partition_transcription_models(allowed: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split an allow-list into (live_capable, file_capable) pools.

    Each dropdown only offers options that work in its context. Retired IDs
    are canonicalized first; unknown entries are dropped.
    """
    normalized = normalize_transcription_model_ids(allowed)
    live = [m for m in normalized if _serves(m, "live")]
    file = [m for m in normalized if _serves(m, "file")]
    return live, file


def get_transcription_model_info(model_id: str) -> TranscriptionModelInfo | None:
    """Look up metadata for a transcription model (retired IDs resolve to
    their first successor). Returns None for unknown models."""
    key = canonical_transcription_model_id(model_id)
    return _TRANSCRIPTION_MODELS.get(key) if key else None


def get_all_transcription_models() -> dict[str, TranscriptionModelInfo]:
    """Return all registered transcription models."""
    return dict(_TRANSCRIPTION_MODELS)


# Union of all supported audio formats across all transcription models.
AUDIO_EXTENSIONS: frozenset[str] = frozenset().union(
    *(m.supported_formats for m in _TRANSCRIPTION_MODELS.values())
)


__all__ = [
    "TRANSCRIPTION_MODEL_REPLACEMENTS",
    "TranscriptionModelInfo",
    "TranscriptionPath",
    "canonical_transcription_model_id",
    "get_transcription_model_info",
    "get_all_transcription_models",
    "normalize_transcription_model_ids",
    "partition_transcription_models",
    "AUDIO_EXTENSIONS",
]
