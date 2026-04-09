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


def get_model_pricing(model: str) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
    """Return ``(input, cached_input, output)`` per-1M-token prices, or *None*."""
    info = get_model_info(model)
    if info and info.input_price is not None:
        return (info.input_price, info.cached_input_price or Decimal("0"), info.output_price or Decimal("0"))
    return None


def calculate_cost(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_input_tokens: Optional[int] = None,
) -> Optional[Decimal]:
    """Compute the USD cost of a call, or *None* if pricing is unknown.

    Handles partial data gracefully: missing token counts are treated as 0.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    inp_price, cached_price, out_price = pricing
    inp = Decimal(input_tokens or 0)
    out = Decimal(output_tokens or 0)
    cached = Decimal(cached_input_tokens or 0)

    # Cached tokens are a subset of input tokens billed at the lower rate.
    billable_input = inp - cached
    cost = (
        billable_input * inp_price / _ONE_MILLION
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


__all__ = ["get_model_pricing", "calculate_cost", "calculate_transcription_cost"]
