"""Image generation model registry.

Single source of truth for image-generation model metadata: pricing,
capabilities, and the provider/model strings sent to the API.

Pricing notes
-------------
Image-generation providers bill in two different shapes, so the registry
supports both and the cost helper picks per model:

* **Per image (flat).** Google's Gemini image models ("Nano Banana") bill an
  effectively fixed price per generated image (a constant token count per
  image at the model's resolution). We hardcode that flat ``price_per_image``.
* **Per token.** OpenAI's ``gpt-image-*`` family bills text/image *input*
  tokens plus *image output* tokens and returns a populated ``usage`` object.
  When such a model is added, set ``input_price_per_1m_tokens`` /
  ``output_image_price_per_1m_tokens`` and the cost helper computes from the
  reported token counts.

When both are configured, per-image wins (it's exact for the flat-priced
models). Verify the numbers on the live pricing page before trusting them —
these models version quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


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
    output_format: str = "png"
    supported_aspect_ratios: tuple[str, ...] = field(
        default_factory=lambda: ("1:1", "3:4", "4:3", "9:16", "16:9")
    )


_IMAGE_GENERATION_MODELS: dict[str, ImageGenModelInfo] = {
    # Gemini 2.5 Flash Image ("Nano Banana") — fast, cheap, editing-capable.
    # Google bills it at a fixed ~$0.039 per 1024px image (1,290 output tokens
    # at $30 / 1M). VERIFY on the live pricing page before relying on this.
    "gemini/gemini-2.5-flash-image": ImageGenModelInfo(
        display_name="Gemini 2.5 Flash Image (Nano Banana)",
        provider="google_genai",
        api_model="gemini-2.5-flash-image",
        price_per_image=Decimal("0.039"),
        supports_editing=True,
        output_format="png",
        supported_aspect_ratios=("1:1", "3:4", "4:3", "9:16", "16:9"),
    ),
}


def get_image_generation_model_info(model_id: str) -> ImageGenModelInfo | None:
    """Look up metadata for an image-generation model. None for unknown models."""
    return _IMAGE_GENERATION_MODELS.get(model_id)


def get_image_generation_models() -> dict[str, ImageGenModelInfo]:
    """Return all registered image-generation models."""
    return dict(_IMAGE_GENERATION_MODELS)


__all__ = [
    "ImageGenModelInfo",
    "get_image_generation_model_info",
    "get_image_generation_models",
]
