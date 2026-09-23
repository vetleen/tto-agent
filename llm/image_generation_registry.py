"""Image generation model registry.

Single source of truth for image-generation model metadata: pricing,
capabilities, and the provider/model strings sent to the API.

Pricing notes
-------------
Image-generation providers bill in two different shapes, so the registry
supports both and the cost helper picks per model:

* **Per image (flat).** Google's Gemini image models ("Nano Banana") bill image
  output by tokens, but the token count per image is fixed for a given
  resolution. We never request a resolution (so we always get the 1K
  default), which makes the price per image effectively flat. We hardcode a
  ``price_per_image`` that is the 1K output price *rounded up* to also cover
  the (small) prompt-input and always-on "minimal" thinking tokens — an
  estimate, not an exact bill.
* **Per token.** OpenAI's ``gpt-image-*`` family bills text/image *input*
  tokens plus *image output* tokens and returns a populated ``usage`` object.
  When such a model is added, set ``input_price_per_1m_tokens`` /
  ``output_image_price_per_1m_tokens`` and the cost helper computes from the
  reported token counts.

When both are configured, per-image wins. Verify the numbers on the live
pricing page before trusting them — these models version quickly.

Retired models
--------------
Retired IDs are not registered; ``IMAGE_MODEL_REPLACEMENTS`` maps each to its
successor so stored org preferences and ``IMAGE_*`` config vars that still
name the old ID keep resolving (mirrors ``MODEL_REPLACEMENTS`` for chat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


# Gemini 3.1 image models. Vertex also lists 9:21 for 3.1 Flash Image, but the
# Developer API doesn't, so it's left out to keep one set for both models.
_GEMINI_31_ASPECT_RATIOS = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
    "1:4", "4:1", "1:8", "8:1",
)


@dataclass(frozen=True)
class ImageGenModelInfo:
    display_name: str
    provider: str  # "google_genai" | "openai"
    api_model: str  # name sent to the API (no "gemini/" prefix)
    # Flat per-image price (Gemini). When set, the cost helper uses
    # ``price_per_image * n_images`` and ignores the token rates below.
    price_per_image: Optional[Decimal] = None
    # Token-based billing (per 1M tokens) for providers that return usage
    # (e.g. OpenAI gpt-image-*). ``output_image_price_per_1m_tokens`` is the
    # rate for generated image tokens; the input rate covers text + image in.
    input_price_per_1m_tokens: Optional[Decimal] = None
    output_image_price_per_1m_tokens: Optional[Decimal] = None
    # Capability flags drive UI gating and runtime routing. New entries must
    # set these explicitly rather than relying on callers to hardcode behavior.
    supports_editing: bool = True  # accepts input image(s) for edit / reference
    max_input_images: int = 3
    output_format: str = "png"
    supported_aspect_ratios: tuple[str, ...] = field(
        default_factory=lambda: ("1:1", "3:4", "4:3", "9:16", "16:9")
    )


_IMAGE_GENERATION_MODELS: dict[str, ImageGenModelInfo] = {
    # Gemini 3.1 Flash Lite Image ("Nano Banana 2 Lite") — GA; Google's named
    # replacement for 2.5 Flash Image. 1K only: 1,120 output tokens at $30 / 1M
    # = $0.0336, rounded up to cover input + minimal-thinking tokens. Docs
    # note it isn't optimized for many reference images or multi-turn edits.
    # Vertex: global location only.
    "gemini/gemini-3.1-flash-lite-image": ImageGenModelInfo(
        display_name="Gemini 3.1 Flash Lite Image (Nano Banana 2 Lite)",
        provider="google_genai",
        api_model="gemini-3.1-flash-lite-image",
        price_per_image=Decimal("0.035"),
        supports_editing=True,
        max_input_images=14,
        output_format="png",
        supported_aspect_ratios=_GEMINI_31_ASPECT_RATIOS,
    ),
    # Gemini 3.1 Flash Image ("Nano Banana 2") — GA; higher quality. At the 1K
    # default: 1,120 output tokens at $60 / 1M = $0.067, rounded up. Supports
    # 512/2K/4K too, but we never request them — pricing assumes 1K.
    "gemini/gemini-3.1-flash-image": ImageGenModelInfo(
        display_name="Gemini 3.1 Flash Image (Nano Banana 2)",
        provider="google_genai",
        api_model="gemini-3.1-flash-image",
        price_per_image=Decimal("0.07"),
        supports_editing=True,
        max_input_images=14,
        output_format="png",
        supported_aspect_ratios=_GEMINI_31_ASPECT_RATIOS,
    ),
}

# Retired image model ID -> current replacement. The retired ID must NOT also
# be registered above, or it would keep resolving to itself.
IMAGE_MODEL_REPLACEMENTS: dict[str, str] = {
    # Developer API shutdown 2026-10-02; Vertex retirement 2027-03-15.
    "gemini/gemini-2.5-flash-image": "gemini/gemini-3.1-flash-lite-image",
}


def canonical_image_model_id(model_id: str | None) -> str | None:
    """Return the current registered ID for a registered or retired image model."""
    if not model_id:
        return None
    candidate = model_id.strip()
    candidate = IMAGE_MODEL_REPLACEMENTS.get(candidate, candidate)
    return candidate if candidate in _IMAGE_GENERATION_MODELS else None


def normalize_image_model_ids(model_ids: list[str]) -> list[str]:
    """Canonicalize, de-duplicate, and drop unknown IDs while preserving order."""
    result: list[str] = []
    for model_id in model_ids:
        canonical = canonical_image_model_id(model_id)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def get_image_generation_model_info(model_id: str) -> ImageGenModelInfo | None:
    """Look up metadata for an image-generation model (retired IDs resolve to
    their replacement). None for unknown models."""
    key = canonical_image_model_id(model_id)
    return _IMAGE_GENERATION_MODELS.get(key) if key else None


def get_image_generation_models() -> dict[str, ImageGenModelInfo]:
    """Return all registered image-generation models."""
    return dict(_IMAGE_GENERATION_MODELS)


def get_system_image_allowed_models() -> list[str]:
    """``settings.IMAGE_ALLOWED_MODELS`` canonicalized against the registry.

    Retired IDs map to their replacement; unknown IDs are dropped.
    """
    from django.conf import settings

    return normalize_image_model_ids(list(getattr(settings, "IMAGE_ALLOWED_MODELS", [])))


def get_system_image_default_model() -> str:
    """``settings.IMAGE_DEFAULT_MODEL`` canonicalized ("" when unset/unknown)."""
    from django.conf import settings

    return canonical_image_model_id(getattr(settings, "IMAGE_DEFAULT_MODEL", "")) or ""


__all__ = [
    "IMAGE_MODEL_REPLACEMENTS",
    "ImageGenModelInfo",
    "canonical_image_model_id",
    "get_image_generation_model_info",
    "get_image_generation_models",
    "get_system_image_allowed_models",
    "get_system_image_default_model",
    "normalize_image_model_ids",
]
