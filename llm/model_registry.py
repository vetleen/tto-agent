"""Unified model registry.

Single source of truth for model metadata: pricing, context windows,
capabilities (thinking, vision), and display names.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelInfo:
    display_name: str
    provider: str  # "openai" | "anthropic" | "google_genai"
    api_model: str  # name sent to API (no prefix)
    supports_thinking: bool = False
    supports_vision: bool = False
    context_window: int = 128_000
    input_price: Decimal | None = None
    cached_input_price: Decimal | None = None
    output_price: Decimal | None = None


# Keyed by full model ID (e.g. "openai/gpt-5.4")
_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-5.4": ModelInfo(
        display_name="GPT-5.4",
        provider="openai",
        api_model="gpt-5.4",
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
        supports_thinking=False,
        supports_vision=True,
        context_window=128_000,
        input_price=Decimal("0.05"),
        cached_input_price=Decimal("0.005"),
        output_price=Decimal("0.40"),
    ),
    # Anthropic
    "anthropic/claude-opus-4-6": ModelInfo(
        display_name="Claude Opus 4.6",
        provider="anthropic",
        api_model="claude-opus-4-6",
        supports_thinking=True,
        supports_vision=True,
        context_window=200_000,
        input_price=Decimal("5.00"),
        cached_input_price=Decimal("0.50"),
        output_price=Decimal("25.00"),
    ),
    "anthropic/claude-sonnet-4-6": ModelInfo(
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        api_model="claude-sonnet-4-6",
        supports_thinking=True,
        supports_vision=True,
        context_window=200_000,
        input_price=Decimal("3.00"),
        cached_input_price=Decimal("0.30"),
        output_price=Decimal("15.00"),
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        display_name="Claude Haiku 4.5",
        provider="anthropic",
        api_model="claude-haiku-4-5",
        supports_thinking=True,
        supports_vision=True,
        context_window=200_000,
        input_price=Decimal("1.00"),
        cached_input_price=Decimal("0.10"),
        output_price=Decimal("5.00"),
    ),
    # Google Gemini
    "gemini/gemini-2.5-pro": ModelInfo(
        display_name="Gemini 2.5 Pro",
        provider="google_genai",
        api_model="gemini-2.5-pro",
        supports_thinking=False,
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
        supports_thinking=False,
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
        supports_thinking=False,
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
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("2.00"),
        cached_input_price=Decimal("0.20"),
        output_price=Decimal("12.00"),
    ),
    "gemini/gemini-3-flash-preview": ModelInfo(
        display_name="Gemini 3 Flash",
        provider="google_genai",
        api_model="gemini-3-flash-preview",
        supports_thinking=True,
        supports_vision=True,
        context_window=1_000_000,
        input_price=Decimal("0.50"),
        cached_input_price=Decimal("0.05"),
        output_price=Decimal("3.00"),
    ),
    "gemini/gemini-3.1-flash-lite-preview": ModelInfo(
        display_name="Gemini 3.1 Flash Lite",
        provider="google_genai",
        api_model="gemini-3.1-flash-lite-preview",
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


__all__ = ["ModelInfo", "get_model_info"]
