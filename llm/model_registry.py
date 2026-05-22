"""Unified model registry.

Single source of truth for model metadata: pricing, context windows,
capabilities (thinking, vision), display names, and tier classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

TIER_CHEAP = "cheap"
TIER_MID = "mid"
TIER_STANDARD = "standard"
TIER_ORDER = {TIER_CHEAP: 0, TIER_MID: 1, TIER_STANDARD: 2}

# Maps preference slot keys to the set of model tiers allowed in that slot.
SLOT_ALLOWED_TIERS: dict[str, set[str]] = {
    "cheap": {TIER_CHEAP},
    "mid": {TIER_MID, TIER_STANDARD},
    "primary": {TIER_STANDARD},
}


@dataclass(frozen=True)
class ModelInfo:
    display_name: str
    provider: str  # "openai" | "anthropic" | "google_genai"
    api_model: str  # name sent to API (no prefix)
    tier: str = TIER_STANDARD
    supports_thinking: bool = False
    supports_vision: bool = False
    context_window: int = 128_000
    input_price: Decimal | None = None
    cached_input_price: Decimal | None = None
    cache_write_price: Decimal | None = None
    output_price: Decimal | None = None


# Keyed by full model ID (e.g. "openai/gpt-5.4")
_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-5.5": ModelInfo(
        display_name="GPT-5.5",
        provider="openai",
        api_model="gpt-5.5",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"),
        output_price=Decimal("30.00"),
    ),
    "openai/gpt-5.4": ModelInfo(
        display_name="GPT-5.4",
        provider="openai",
        api_model="gpt-5.4",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("1.75"),
        cached_input_price=Decimal("0.175"),
        output_price=Decimal("14.00"),
    ),
    "openai/gpt-5.4-mini": ModelInfo(
        display_name="GPT-5.4 Mini",
        provider="openai",
        api_model="gpt-5.4-mini",
        tier=TIER_MID,
        supports_thinking=False,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.25"),
        cached_input_price=Decimal("0.025"),
        output_price=Decimal("2.00"),
    ),
    "openai/gpt-5.4-nano": ModelInfo(
        display_name="GPT-5.4 Nano",
        provider="openai",
        api_model="gpt-5.4-nano",
        tier=TIER_CHEAP,
        supports_thinking=False,
        supports_vision=True,
        context_window=128_000,
        input_price=Decimal("0.05"),
        cached_input_price=Decimal("0.005"),
        output_price=Decimal("0.40"),
    ),
    # Anthropic
    "anthropic/claude-opus-4-7": ModelInfo(
        display_name="Claude Opus 4.7",
        provider="anthropic",
        api_model="claude-opus-4-7",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"),
        cache_write_price=Decimal("6.25"),
        output_price=Decimal("25.00"),
    ),
    "anthropic/claude-opus-4-6": ModelInfo(
        display_name="Claude Opus 4.6",
        provider="anthropic",
        api_model="claude-opus-4-6",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"),
        cache_write_price=Decimal("6.25"),
        output_price=Decimal("25.00"),
    ),
    "anthropic/claude-sonnet-4-6": ModelInfo(
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        api_model="claude-sonnet-4-6",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("3.00"),
        cached_input_price=Decimal("0.30"),
        cache_write_price=Decimal("3.75"),
        output_price=Decimal("15.00"),
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        display_name="Claude Haiku 4.5",
        provider="anthropic",
        api_model="claude-haiku-4-5",
        tier=TIER_MID,
        supports_thinking=True,
        supports_vision=True,
        context_window=200_000,
        input_price=Decimal("1.00"),
        cached_input_price=Decimal("0.10"),
        cache_write_price=Decimal("1.25"),
        output_price=Decimal("5.00"),
    ),
    # Google Gemini
    "gemini/gemini-2.5-pro": ModelInfo(
        display_name="Gemini 2.5 Pro",
        provider="google_genai",
        api_model="gemini-2.5-pro",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("1.25"),
        cached_input_price=Decimal("0.125"),
        output_price=Decimal("10.00"),
    ),
    "gemini/gemini-2.5-flash": ModelInfo(
        display_name="Gemini 2.5 Flash",
        provider="google_genai",
        api_model="gemini-2.5-flash",
        tier=TIER_MID,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.30"),
        cached_input_price=Decimal("0.03"),
        output_price=Decimal("2.50"),
    ),
    "gemini/gemini-2.5-flash-lite": ModelInfo(
        display_name="Gemini 2.5 Flash Lite",
        provider="google_genai",
        api_model="gemini-2.5-flash-lite",
        tier=TIER_CHEAP,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.10"),
        cached_input_price=Decimal("0.01"),
        output_price=Decimal("0.40"),
    ),
    "gemini/gemini-3.1-pro-preview": ModelInfo(
        display_name="Gemini 3.1 Pro",
        provider="google_genai",
        api_model="gemini-3.1-pro-preview",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"),
        output_price=Decimal("12.00"),
    ),
    "gemini/gemini-3.5-flash": ModelInfo(
        display_name="Gemini 3.5 Flash",
        provider="google_genai",
        api_model="gemini-3.5-flash",
        tier=TIER_STANDARD,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("1.50"),
        cached_input_price=Decimal("0.15"),
        output_price=Decimal("9.00"),
    ),
    "gemini/gemini-3-flash-preview": ModelInfo(
        display_name="Gemini 3 Flash",
        provider="google_genai",
        api_model="gemini-3-flash-preview",
        tier=TIER_MID,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.50"),
        cached_input_price=Decimal("0.05"),
        output_price=Decimal("3.00"),
    ),
    "gemini/gemini-3.1-flash-lite": ModelInfo(
        display_name="Gemini 3.1 Flash Lite",
        provider="google_genai",
        api_model="gemini-3.1-flash-lite",
        tier=TIER_CHEAP,
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.25"),
        cached_input_price=Decimal("0.025"),
        output_price=Decimal("1.50"),
    ),
}

# Provider prefix mapping for normalisation
_PROVIDER_PREFIXES = ("openai/", "anthropic/", "gemini/")


def _normalize(model_id: str) -> str | None:
    """Try to find a canonical key in _MODELS for *model_id*.

    Handles:
    - Exact match: "openai/gpt-5.4"
    - Bare model name: "gpt-5.4" -> "openai/gpt-5.4"
    """
    if model_id in _MODELS:
        return model_id
    # Try adding each provider prefix
    for prefix in _PROVIDER_PREFIXES:
        candidate = prefix + model_id
        if candidate in _MODELS:
            return candidate
    return None


def get_model_info(model_id: str) -> ModelInfo | None:
    """Look up metadata for a model. Returns None for unknown models."""
    key = _normalize(model_id)
    return _MODELS.get(key) if key else None


def get_model_tier(model_id: str) -> str | None:
    """Return the tier for a model, or None if unknown."""
    info = get_model_info(model_id)
    return info.tier if info else None


def get_models_by_tier(tier: str) -> list[str]:
    """Return all registered model IDs with the given tier."""
    return [mid for mid, info in _MODELS.items() if info.tier == tier]


def get_models_at_or_above_tier(tier: str) -> list[str]:
    """Return model IDs at the given tier or higher."""
    min_rank = TIER_ORDER.get(tier, 0)
    return [
        mid for mid, info in _MODELS.items()
        if TIER_ORDER.get(info.tier, 0) >= min_rank
    ]


def is_model_valid_for_slot(model_id: str, slot: str) -> bool:
    """Check if a model's tier is valid for a given preference slot."""
    info = get_model_info(model_id)
    if info is None:
        return False
    allowed = SLOT_ALLOWED_TIERS.get(slot)
    if allowed is None:
        return True
    return info.tier in allowed


def get_models_for_slot(slot: str, allowed_models: list[str] | None = None) -> list[str]:
    """Return model IDs valid for a preference slot, optionally filtered by an allow-list."""
    allowed_tiers = SLOT_ALLOWED_TIERS.get(slot)
    candidates = allowed_models if allowed_models else list(_MODELS.keys())
    if allowed_tiers is None:
        return list(candidates)
    return [m for m in candidates if (info := get_model_info(m)) and info.tier in allowed_tiers]


__all__ = [
    "ModelInfo",
    "TIER_CHEAP",
    "TIER_MID",
    "TIER_STANDARD",
    "TIER_ORDER",
    "SLOT_ALLOWED_TIERS",
    "get_model_info",
    "get_model_tier",
    "get_models_by_tier",
    "get_models_at_or_above_tier",
    "is_model_valid_for_slot",
    "get_models_for_slot",
]
