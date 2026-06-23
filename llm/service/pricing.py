"""Model pricing data and cost calculation.

Maps model names to per-1M-token prices and provides helpers to compute
the USD cost of a single LLM call.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Tuple

from llm.model_registry import get_model_info

_ONE_MILLION = Decimal("1000000")


def _normalize_model_name(model: str) -> str:
    """Strip provider prefixes like ``openai/``, ``anthropic/``, ``gemini/``."""
    for prefix in ("openai/", "anthropic/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def get_model_pricing(model: str) -> Optional[Tuple[Decimal, Decimal, Decimal, Decimal]]:
    """Return ``(input, cached_input, cache_write, output)`` per-1M-token prices, or *None*."""
    info = get_model_info(model)
    if info and info.input_price is not None:
        return (
            info.input_price,
            info.cached_input_price or Decimal("0"),
            info.cache_write_price or info.input_price,
            info.output_price or Decimal("0"),
        )
    return None


def calculate_cost(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_input_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    *,
    cache_write_1h_tokens: Optional[int] = None,
) -> Optional[Decimal]:
    """Compute the USD cost of a call, or *None* if pricing is unknown.

    Handles partial data gracefully: missing token counts are treated as 0.

    ``cache_write_1h_tokens`` is the 1h-TTL portion of ``cache_write_tokens``
    (Anthropic bills 1h writes at 2x input vs 1.25x for 5m writes). When the
    breakdown is absent — or the model has no 1h rate configured — all cache
    writes bill at the 5m rate, matching the pre-breakdown behavior.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    inp_price, cached_price, write_price, out_price = pricing
    inp = Decimal(input_tokens or 0)
    out = Decimal(output_tokens or 0)
    cached = Decimal(cached_input_tokens or 0)
    written = Decimal(cache_write_tokens or 0)

    written_1h = Decimal(0)
    write_1h_price = Decimal(0)
    if cache_write_1h_tokens:
        info = get_model_info(model)
        if info and info.cache_write_1h_price is not None:
            # Clamp to the total write count in case of inconsistent data.
            written_1h = min(Decimal(cache_write_1h_tokens), written)
            write_1h_price = info.cache_write_1h_price
    written_5m = written - written_1h

    # Cache reads and writes are subsets of input tokens, each billed at
    # their own rate.  The remainder is regular input.
    regular_input = inp - cached - written
    cost = (
        regular_input * inp_price / _ONE_MILLION
        + written_5m * write_price / _ONE_MILLION
        + written_1h * write_1h_price / _ONE_MILLION
        + cached * cached_price / _ONE_MILLION
        + out * out_price / _ONE_MILLION
    )
    return cost


def calculate_transcription_cost(
    model: str,
    audio_duration_seconds: float,
    *,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> Optional[Decimal]:
    """Compute the USD cost of a transcription call.

    Prefers token-based billing when the caller supplies both ``input_tokens``
    and ``output_tokens`` AND the model has per-token prices configured —
    this is the accurate path for ``gpt-4o-transcribe`` /
    ``gpt-4o-mini-transcribe``, which return a populated ``response.usage``
    from the OpenAI API.

    Falls back to ``audio_duration_seconds * price_per_minute`` when:
      - the caller did not (or could not) pass tokens (e.g. whisper-1's
        ``UsageDuration`` response, or a response missing ``usage`` entirely);
      - the registered model has no per-token pricing.

    Returns *None* for unknown models or when neither pricing path is
    configured.
    """
    from llm.transcription_registry import get_transcription_model_info

    info = get_transcription_model_info(model)
    if info is None:
        return None

    # Token-based path: both tokens and both rates present.
    if (
        input_tokens is not None
        and output_tokens is not None
        and info.input_price_per_1m_tokens is not None
        and info.output_price_per_1m_tokens is not None
    ):
        inp = Decimal(input_tokens) * info.input_price_per_1m_tokens / _ONE_MILLION
        out = Decimal(output_tokens) * info.output_price_per_1m_tokens / _ONE_MILLION
        return inp + out

    # Duration fallback.
    if info.price_per_minute is None:
        return None
    minutes = Decimal(str(audio_duration_seconds)) / Decimal("60")
    return info.price_per_minute * minutes


def calculate_image_generation_cost(
    model: str,
    n_images: int = 1,
    *,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> Optional[Decimal]:
    """Compute the USD cost of an image-generation call, or *None* if unknown.

    Prefers the flat per-image price when the model defines one — Gemini's
    "Nano Banana" family bills a fixed amount per generated image, so
    ``price_per_image * n_images`` is exact. Falls back to token-based billing
    (input tokens + image-output tokens) for models that report ``usage``
    (e.g. OpenAI ``gpt-image-*``).

    Returns *None* for unknown models or when neither pricing path is
    configured (so the call is still logged, just without a cost).
    """
    from llm.image_generation_registry import get_image_generation_model_info

    info = get_image_generation_model_info(model)
    if info is None:
        return None

    # Per-image path (exact for flat-priced models).
    if info.price_per_image is not None:
        return info.price_per_image * Decimal(n_images)

    # Token-based path: both token counts and both rates present.
    if (
        input_tokens is not None
        and output_tokens is not None
        and info.input_price_per_1m_tokens is not None
        and info.output_image_price_per_1m_tokens is not None
    ):
        inp = Decimal(input_tokens) * info.input_price_per_1m_tokens / _ONE_MILLION
        out = Decimal(output_tokens) * info.output_image_price_per_1m_tokens / _ONE_MILLION
        return inp + out

    return None


__all__ = [
    "get_model_pricing",
    "calculate_cost",
    "calculate_transcription_cost",
    "calculate_image_generation_cost",
]
