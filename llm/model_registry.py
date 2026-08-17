"""Curated LLM model registry.

This is the single source of truth for model IDs, capabilities, reasoning
controls, context limits, and token prices shown to users and used for billing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

TIER_CHEAP = "cheap"
TIER_MID = "mid"
TIER_STANDARD = "standard"
TIER_PREMIUM = "premium"
TIER_ORDER = {
    TIER_CHEAP: 0,
    TIER_MID: 1,
    TIER_STANDARD: 2,
    TIER_PREMIUM: 3,
}

SLOT_ALLOWED_TIERS: dict[str, set[str]] = {
    "cheap": {TIER_CHEAP},
    "mid": {TIER_MID, TIER_STANDARD, TIER_PREMIUM},
    "primary": {TIER_STANDARD, TIER_PREMIUM},
}


def get_performance_tier(input_price: Decimal | None) -> str:
    """Map standard input price per 1M tokens to the four performance tiers."""
    if input_price is None:
        return TIER_STANDARD
    if input_price <= Decimal("0.50"):
        return TIER_CHEAP
    if input_price <= Decimal("1.50"):
        return TIER_MID
    if input_price < Decimal("5.00"):
        return TIER_STANDARD
    return TIER_PREMIUM


@dataclass(frozen=True)
class PriceChange:
    """A complete set of token prices that takes effect on ``starts_on``."""

    starts_on: date
    input_price: Decimal
    cached_input_price: Decimal
    cache_write_price: Decimal | None
    cache_write_1h_price: Decimal | None
    output_price: Decimal


@dataclass(frozen=True)
class ModelInfo:
    display_name: str
    provider: str  # "openai" | "anthropic" | "google_genai"
    api_model: str
    # The first four performance stars derive from the standard input price.
    # Only the very best current models receive the manually curated fifth.
    flagship: bool = False
    # Exact values accepted by the provider for this model, in UX order.
    reasoning_levels: tuple[str, ...] = ()
    default_reasoning_level: str | None = None
    # Anthropic transport: adaptive or extended. Other providers leave this None.
    thinking_mode: str | None = None
    uses_responses_api: bool = False
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    context_window: int = 128_000
    max_output_tokens: int = 16_384
    input_price: Decimal | None = None
    cached_input_price: Decimal | None = None
    cache_write_price: Decimal | None = None
    cache_write_1h_price: Decimal | None = None
    output_price: Decimal | None = None
    # Some providers price the entire request at a higher rate once its input
    # crosses a threshold. The corresponding long-context fields hold that band.
    long_context_threshold: int | None = None
    long_context_input_price: Decimal | None = None
    long_context_cached_input_price: Decimal | None = None
    long_context_cache_write_price: Decimal | None = None
    long_context_cache_write_1h_price: Decimal | None = None
    long_context_output_price: Decimal | None = None
    price_changes: tuple[PriceChange, ...] = ()

    @property
    def supports_thinking(self) -> bool:
        return bool(self.reasoning_levels)

    @property
    def supports_max_effort(self) -> bool:
        """Backward-compatible capability accessor."""
        return "max" in self.reasoning_levels

    @property
    def supports_vision(self) -> bool:
        return "image" in self.input_modalities

    @property
    def tier(self) -> str:
        return get_performance_tier(self.input_price)


_GPT56_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")
_MULTIMODAL = ("text", "image", "pdf")

# Keyed only by the models that may appear in the curated picker. Historical
# IDs are handled separately by MODEL_REPLACEMENTS so stored preferences migrate
# without putting retired models back into the allow-list.
_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-5.6-sol": ModelInfo(
        display_name="GPT-5.6 Sol", provider="openai", api_model="gpt-5.6-sol",
        flagship=True,
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("5.00"), cached_input_price=Decimal("0.50"),
        cache_write_price=Decimal("6.25"), output_price=Decimal("30.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("10.00"),
        long_context_cached_input_price=Decimal("1.00"),
        long_context_cache_write_price=Decimal("12.50"),
        long_context_output_price=Decimal("45.00"),
    ),
    "openai/gpt-5.6-terra": ModelInfo(
        display_name="GPT-5.6 Terra", provider="openai", api_model="gpt-5.6-terra",
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("2.50"), cached_input_price=Decimal("0.25"),
        cache_write_price=Decimal("3.125"), output_price=Decimal("15.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("5.00"),
        long_context_cached_input_price=Decimal("0.50"),
        long_context_cache_write_price=Decimal("6.25"),
        long_context_output_price=Decimal("22.50"),
    ),
    "openai/gpt-5.6-luna": ModelInfo(
        display_name="GPT-5.6 Luna", provider="openai", api_model="gpt-5.6-luna",
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("1.00"), cached_input_price=Decimal("0.10"),
        cache_write_price=Decimal("1.25"), output_price=Decimal("6.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("2.00"),
        long_context_cached_input_price=Decimal("0.20"),
        long_context_cache_write_price=Decimal("2.50"),
        long_context_output_price=Decimal("9.00"),
    ),
    "openai/gpt-5.4-nano": ModelInfo(
        display_name="GPT-5.4 Nano", provider="openai", api_model="gpt-5.4-nano",
        reasoning_levels=("none", "low", "medium", "high", "xhigh"),
        default_reasoning_level="none", uses_responses_api=True,
        input_modalities=_MULTIMODAL, context_window=400_000,
        max_output_tokens=128_000, input_price=Decimal("0.20"),
        cached_input_price=Decimal("0.02"), cache_write_price=Decimal("0.25"),
        output_price=Decimal("1.25"),
    ),
    # Anthropic
    "anthropic/claude-fable-5": ModelInfo(
        display_name="Claude Fable 5", provider="anthropic", api_model="claude-fable-5",
        flagship=True,
        reasoning_levels=("low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("10.00"),
        cached_input_price=Decimal("1.00"), cache_write_price=Decimal("12.50"),
        cache_write_1h_price=Decimal("20.00"), output_price=Decimal("50.00"),
    ),
    "anthropic/claude-opus-5": ModelInfo(
        display_name="Claude Opus 5", provider="anthropic", api_model="claude-opus-5",
        reasoning_levels=("off", "low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"), cache_write_price=Decimal("6.25"),
        cache_write_1h_price=Decimal("10.00"), output_price=Decimal("25.00"),
    ),
    "anthropic/claude-opus-4-6": ModelInfo(
        display_name="Claude Opus 4.6", provider="anthropic", api_model="claude-opus-4-6",
        reasoning_levels=("off", "low", "medium", "high", "max"),
        default_reasoning_level="off", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"), cache_write_price=Decimal("6.25"),
        cache_write_1h_price=Decimal("10.00"), output_price=Decimal("25.00"),
    ),
    "anthropic/claude-sonnet-5": ModelInfo(
        display_name="Claude Sonnet 5", provider="anthropic", api_model="claude-sonnet-5",
        reasoning_levels=("off", "low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"), cache_write_price=Decimal("2.50"),
        cache_write_1h_price=Decimal("4.00"), output_price=Decimal("10.00"),
        price_changes=(PriceChange(
            starts_on=date(2026, 9, 1), input_price=Decimal("3.00"),
            cached_input_price=Decimal("0.30"), cache_write_price=Decimal("3.75"),
            cache_write_1h_price=Decimal("6.00"), output_price=Decimal("15.00"),
        ),),
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        display_name="Claude Haiku 4.5", provider="anthropic", api_model="claude-haiku-4-5",
        reasoning_levels=("off", "low", "medium", "high"),
        default_reasoning_level="off", thinking_mode="extended",
        input_modalities=_MULTIMODAL, context_window=200_000,
        max_output_tokens=64_000, input_price=Decimal("1.00"),
        cached_input_price=Decimal("0.10"), cache_write_price=Decimal("1.25"),
        cache_write_1h_price=Decimal("2.00"), output_price=Decimal("5.00"),
    ),
    # Google Gemini
    "gemini/gemini-3.1-pro-preview": ModelInfo(
        display_name="Gemini 3.1 Pro Preview", provider="google_genai",
        api_model="gemini-3.1-pro-preview",
        reasoning_levels=("low", "medium", "high"), default_reasoning_level="high",
        input_modalities=_MULTIMODAL, context_window=1_048_576,
        max_output_tokens=65_536, input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"), output_price=Decimal("12.00"),
        long_context_threshold=200_000,
        long_context_input_price=Decimal("4.00"),
        long_context_cached_input_price=Decimal("0.40"),
        long_context_output_price=Decimal("18.00"),
    ),
    "gemini/gemini-3.7-flash": ModelInfo(
        display_name="Gemini 3.7 Flash", provider="google_genai",
        api_model="gemini-3.7-flash",
        reasoning_levels=("low", "medium", "high"), default_reasoning_level="medium",
        input_modalities=_MULTIMODAL, context_window=1_048_576,
        max_output_tokens=65_536, input_price=Decimal("0.75"),
        cached_input_price=Decimal("0.075"), output_price=Decimal("3.75"),
        price_changes=(PriceChange(
            starts_on=date(2027, 1, 1), input_price=Decimal("1.50"),
            cached_input_price=Decimal("0.15"), cache_write_price=None,
            cache_write_1h_price=None, output_price=Decimal("7.50"),
        ),),
    ),
    "gemini/gemini-3.5-flash-lite": ModelInfo(
        display_name="Gemini 3.5 Flash-Lite", provider="google_genai",
        api_model="gemini-3.5-flash-lite",
        reasoning_levels=("minimal", "low", "medium", "high"),
        default_reasoning_level="minimal", input_modalities=_MULTIMODAL,
        context_window=1_048_576, max_output_tokens=65_536,
        input_price=Decimal("0.30"), cached_input_price=Decimal("0.03"),
        output_price=Decimal("2.50"),
    ),
}


MODEL_REPLACEMENTS: dict[str, str] = {
    "openai/gpt-5.5": "openai/gpt-5.6-sol",
    "openai/gpt-5.4": "openai/gpt-5.6-terra",
    "openai/gpt-5.4-mini": "openai/gpt-5.6-luna",
    "anthropic/claude-opus-4-8": "anthropic/claude-opus-5",
    "anthropic/claude-opus-4-7": "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-5",
    "gemini/gemini-3.5-flash": "gemini/gemini-3.7-flash",
    "gemini/gemini-3.1-flash-lite": "gemini/gemini-3.5-flash-lite",
}

_PROVIDER_PREFIXES = ("openai/", "anthropic/", "gemini/")


def _with_inferred_prefix(model_id: str) -> str:
    if "/" in model_id:
        return model_id
    if model_id.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{model_id}"
    if model_id.startswith("claude-"):
        return f"anthropic/{model_id}"
    if model_id.startswith("gemini-"):
        return f"gemini/{model_id}"
    return model_id


def canonical_model_id(model_id: str | None) -> str | None:
    """Return the current canonical ID for a registered or retired model."""
    if not model_id:
        return None
    candidate = _with_inferred_prefix(model_id.strip())
    candidate = MODEL_REPLACEMENTS.get(candidate, candidate)
    return candidate if candidate in _MODELS else None


def normalize_model_ids(model_ids: list[str]) -> list[str]:
    """Canonicalize, de-duplicate, and drop unknown IDs while preserving order."""
    result: list[str] = []
    for model_id in model_ids:
        canonical = canonical_model_id(model_id)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def get_model_info(model_id: str) -> ModelInfo | None:
    key = canonical_model_id(model_id)
    return _MODELS.get(key) if key else None


def get_registered_model_ids() -> list[str]:
    return list(_MODELS)


def get_model_tier(model_id: str) -> str | None:
    info = get_model_info(model_id)
    return info.tier if info else None


def get_models_by_tier(tier: str) -> list[str]:
    return [mid for mid, info in _MODELS.items() if info.tier == tier]


def get_models_at_or_above_tier(tier: str) -> list[str]:
    min_rank = TIER_ORDER.get(tier, 0)
    return [
        mid for mid, info in _MODELS.items()
        if TIER_ORDER.get(info.tier, 0) >= min_rank
    ]


def is_model_valid_for_slot(model_id: str, slot: str) -> bool:
    info = get_model_info(model_id)
    if info is None:
        return False
    allowed = SLOT_ALLOWED_TIERS.get(slot)
    return allowed is None or info.tier in allowed


def get_models_for_slot(slot: str, allowed_models: list[str] | None = None) -> list[str]:
    allowed_tiers = SLOT_ALLOWED_TIERS.get(slot)
    candidates = normalize_model_ids(allowed_models) if allowed_models else list(_MODELS)
    if allowed_tiers is None:
        return candidates
    return [m for m in candidates if (info := get_model_info(m)) and info.tier in allowed_tiers]


__all__ = [
    "ModelInfo", "PriceChange", "MODEL_REPLACEMENTS",
    "TIER_CHEAP", "TIER_MID", "TIER_STANDARD", "TIER_PREMIUM", "TIER_ORDER",
    "SLOT_ALLOWED_TIERS", "canonical_model_id", "normalize_model_ids",
    "get_performance_tier",
    "get_model_info", "get_registered_model_ids", "get_model_tier",
    "get_models_by_tier", "get_models_at_or_above_tier",
    "is_model_valid_for_slot", "get_models_for_slot",
]
