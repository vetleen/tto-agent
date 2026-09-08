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
TIER_FLAGSHIP = "flagship"
TIER_ORDER = {
    TIER_CHEAP: 0,
    TIER_MID: 1,
    TIER_STANDARD: 2,
    TIER_FLAGSHIP: 3,
}

# Slot eligibility floors on capability stars. The cheap slot has no star
# floor — it is price-defined (a model qualifies via the cheap category).
SLOT_MIN_STARS: dict[str, int] = {
    "mid": 2,
    "primary": 3,
}

# Star rating → tier category. 1-star models categorize only as cheap (via
# price); any model may additionally be cheap when priced below $0.50.
_STARS_TO_TIER = {
    2: TIER_MID,
    3: TIER_MID,
    4: TIER_STANDARD,
    5: TIER_FLAGSHIP,
}


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
    # Manually curated 1-5 picker star rating. When None, capability_stars
    # falls back to an input-price rating (+1 for flagship) — set them
    # explicitly so promo pricing can't demote a model's standing (e.g. Sol's
    # 2026 promo price).
    stars: int | None = None
    # Only the very best current models are flagships (tooltip label; also the
    # fifth star in the price-derived fallback).
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
    def capability_stars(self) -> int:
        """Manual picker stars, falling back to a price-derived rating."""
        if self.stars is not None:
            return self.stars
        if self.input_price is None:
            base = 3
        elif self.input_price <= Decimal("0.50"):
            base = 1
        elif self.input_price <= Decimal("1.50"):
            base = 2
        elif self.input_price < Decimal("5.00"):
            base = 3
        else:
            base = 4
        return min(base + (1 if self.flagship else 0), 5)

    @property
    def tiers(self) -> frozenset[str]:
        """Every performance category the model belongs to (may overlap)."""
        result: set[str] = set()
        if self.input_price is not None and self.input_price < Decimal("0.50"):
            result.add(TIER_CHEAP)
        star_tier = _STARS_TO_TIER.get(self.capability_stars)
        if star_tier:
            result.add(star_tier)
        return frozenset(result)

    @property
    def tier(self) -> str:
        """Highest-ranked category, for ordinal consumers and display."""
        if not self.tiers:
            return TIER_STANDARD
        return max(self.tiers, key=TIER_ORDER.__getitem__)


_GPT56_LEVELS = ("none", "low", "medium", "high", "xhigh", "max")
_MULTIMODAL = ("text", "image", "pdf")

# Keyed only by the models that may appear in the curated picker. Historical
# IDs are handled separately by MODEL_REPLACEMENTS so stored preferences migrate
# without putting retired models back into the allow-list.
_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-6-astra": ModelInfo(
        display_name="GPT-6 Astra", provider="openai", api_model="gpt-6-astra",
        stars=5, flagship=True,
        # Astra dropped the "none" effort the GPT-5.6 family accepts.
        reasoning_levels=("low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("10.00"), cached_input_price=Decimal("1.00"),
        cache_write_price=Decimal("12.50"), output_price=Decimal("50.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("20.00"),
        long_context_cached_input_price=Decimal("2.00"),
        long_context_cache_write_price=Decimal("25.00"),
        long_context_output_price=Decimal("75.00"),
    ),
    "openai/gpt-5.6-sol": ModelInfo(
        display_name="GPT-5.6 Sol", provider="openai", api_model="gpt-5.6-sol",
        stars=4,
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        # Promotional pricing through at least 2026-11-21; list price is
        # $5.00/$30.00 (LC $10/$45) — restore it when OpenAI ends the promo.
        input_price=Decimal("4.00"), cached_input_price=Decimal("0.40"),
        cache_write_price=Decimal("5.00"), output_price=Decimal("20.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("8.00"),
        long_context_cached_input_price=Decimal("0.80"),
        long_context_cache_write_price=Decimal("10.00"),
        long_context_output_price=Decimal("30.00"),
    ),
    "openai/gpt-5.6-terra": ModelInfo(
        display_name="GPT-5.6 Terra", provider="openai", api_model="gpt-5.6-terra",
        stars=3,
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("2.00"), cached_input_price=Decimal("0.20"),
        cache_write_price=Decimal("2.50"), output_price=Decimal("12.00"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("4.00"),
        long_context_cached_input_price=Decimal("0.40"),
        long_context_cache_write_price=Decimal("5.00"),
        long_context_output_price=Decimal("18.00"),
    ),
    "openai/gpt-5.6-luna": ModelInfo(
        display_name="GPT-5.6 Luna", provider="openai", api_model="gpt-5.6-luna",
        stars=2,
        reasoning_levels=_GPT56_LEVELS, default_reasoning_level="medium",
        uses_responses_api=True, input_modalities=_MULTIMODAL,
        context_window=1_050_000, max_output_tokens=128_000,
        input_price=Decimal("0.20"), cached_input_price=Decimal("0.02"),
        cache_write_price=Decimal("0.25"), output_price=Decimal("1.20"),
        long_context_threshold=272_000,
        long_context_input_price=Decimal("0.40"),
        long_context_cached_input_price=Decimal("0.04"),
        long_context_cache_write_price=Decimal("0.50"),
        long_context_output_price=Decimal("1.80"),
    ),
    "openai/gpt-5.4-nano": ModelInfo(
        display_name="GPT-5.4 Nano", provider="openai", api_model="gpt-5.4-nano",
        stars=1,
        reasoning_levels=("none", "low", "medium", "high", "xhigh"),
        default_reasoning_level="none", uses_responses_api=True,
        input_modalities=_MULTIMODAL, context_window=400_000,
        max_output_tokens=128_000, input_price=Decimal("0.20"),
        cached_input_price=Decimal("0.02"), cache_write_price=Decimal("0.25"),
        output_price=Decimal("1.25"),
    ),
    # Anthropic
    "anthropic/claude-fable-5-1": ModelInfo(
        display_name="Claude Fable 5.1", provider="anthropic",
        api_model="claude-fable-5-1", stars=5, flagship=True,
        reasoning_levels=("low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("10.00"),
        # Anthropic prices Fable 5.1 cache reads at 0.025x, not the usual 0.1x.
        cached_input_price=Decimal("0.25"), cache_write_price=Decimal("12.50"),
        cache_write_1h_price=Decimal("20.00"), output_price=Decimal("50.00"),
    ),
    "anthropic/claude-fable-5": ModelInfo(
        display_name="Claude Fable 5", provider="anthropic", api_model="claude-fable-5",
        stars=5, flagship=True,
        reasoning_levels=("low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("10.00"),
        cached_input_price=Decimal("1.00"), cache_write_price=Decimal("12.50"),
        cache_write_1h_price=Decimal("20.00"), output_price=Decimal("50.00"),
    ),
    "anthropic/claude-opus-5": ModelInfo(
        display_name="Claude Opus 5", provider="anthropic", api_model="claude-opus-5",
        stars=4,
        reasoning_levels=("off", "low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"), cache_write_price=Decimal("6.25"),
        cache_write_1h_price=Decimal("10.00"), output_price=Decimal("25.00"),
    ),
    "anthropic/claude-opus-4-8": ModelInfo(
        display_name="Claude Opus 4.8", provider="anthropic", api_model="claude-opus-4-8",
        stars=4,
        reasoning_levels=("off", "low", "medium", "high", "max"),
        default_reasoning_level="off", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"), cache_write_price=Decimal("6.25"),
        cache_write_1h_price=Decimal("10.00"), output_price=Decimal("25.00"),
    ),
    "anthropic/claude-opus-4-6": ModelInfo(
        display_name="Claude Opus 4.6", provider="anthropic", api_model="claude-opus-4-6",
        stars=4,
        reasoning_levels=("off", "low", "medium", "high", "max"),
        default_reasoning_level="off", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"), cache_write_price=Decimal("6.25"),
        cache_write_1h_price=Decimal("10.00"), output_price=Decimal("25.00"),
    ),
    "anthropic/claude-sonnet-5": ModelInfo(
        display_name="Claude Sonnet 5", provider="anthropic", api_model="claude-sonnet-5",
        stars=3,
        reasoning_levels=("off", "low", "medium", "high", "xhigh", "max"),
        default_reasoning_level="high", thinking_mode="adaptive",
        input_modalities=_MULTIMODAL, context_window=1_000_000,
        max_output_tokens=128_000, input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"), cache_write_price=Decimal("2.50"),
        cache_write_1h_price=Decimal("4.00"), output_price=Decimal("10.00"),
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        display_name="Claude Haiku 4.5", provider="anthropic", api_model="claude-haiku-4-5",
        stars=2,
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
        api_model="gemini-3.1-pro-preview", stars=3,
        reasoning_levels=("low", "medium", "high"), default_reasoning_level="high",
        input_modalities=_MULTIMODAL, context_window=1_048_576,
        max_output_tokens=65_536, input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"), output_price=Decimal("12.00"),
        long_context_threshold=200_000,
        long_context_input_price=Decimal("4.00"),
        long_context_cached_input_price=Decimal("0.40"),
        long_context_output_price=Decimal("18.00"),
    ),
    "gemini/gemini-3.8-flash": ModelInfo(
        display_name="Gemini 3.8 Flash", provider="google_genai",
        api_model="gemini-3.8-flash", stars=2,
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
    "gemini/gemini-3.7-flash": ModelInfo(
        display_name="Gemini 3.7 Flash", provider="google_genai",
        api_model="gemini-3.7-flash", stars=2,
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
        api_model="gemini-3.5-flash-lite", stars=1,
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
    return [mid for mid, info in _MODELS.items() if tier in info.tiers]


def get_models_with_min_stars(min_stars: int) -> list[str]:
    return [mid for mid, info in _MODELS.items() if info.capability_stars >= min_stars]


def is_model_valid_for_slot(model_id: str, slot: str) -> bool:
    info = get_model_info(model_id)
    if info is None:
        return False
    if slot == "cheap":
        return TIER_CHEAP in info.tiers
    floor = SLOT_MIN_STARS.get(slot)
    return floor is None or info.capability_stars >= floor


def get_models_for_slot(slot: str, allowed_models: list[str] | None = None) -> list[str]:
    candidates = normalize_model_ids(allowed_models) if allowed_models else list(_MODELS)
    return [m for m in candidates if is_model_valid_for_slot(m, slot)]


__all__ = [
    "ModelInfo", "PriceChange", "MODEL_REPLACEMENTS",
    "TIER_CHEAP", "TIER_MID", "TIER_STANDARD", "TIER_FLAGSHIP", "TIER_ORDER",
    "SLOT_MIN_STARS", "canonical_model_id", "normalize_model_ids",
    "get_model_info", "get_registered_model_ids", "get_model_tier",
    "get_models_by_tier", "get_models_with_min_stars",
    "is_model_valid_for_slot", "get_models_for_slot",
]
