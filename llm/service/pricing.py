"""Model pricing data and cost calculation.

Maps model names to per-1M-token prices and provides helpers to compute
the USD cost of a single LLM call.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional, Tuple

# (input_per_1M, cached_input_per_1M, output_per_1M)
_PRICING: dict[str, Tuple[Decimal, Decimal, Decimal]] = {
    # OpenAI
    "gpt-5.2": (Decimal("1.75"), Decimal("0.175"), Decimal("14.00")),
    "gpt-5-mini": (Decimal("0.25"), Decimal("0.025"), Decimal("2.00")),
    "gpt-5-nano": (Decimal("0.05"), Decimal("0.005"), Decimal("0.40")),
    # Anthropic
    "claude-opus-4-6": (Decimal("5.00"), Decimal("0.50"), Decimal("25.00")),
    "claude-sonnet-4-6": (Decimal("3.00"), Decimal("0.30"), Decimal("15.00")),
    "claude-haiku-4-5-20251001": (Decimal("1.00"), Decimal("0.10"), Decimal("5.00")),
    # Google
    "gemini-2.5-pro": (Decimal("1.25"), Decimal("0.125"), Decimal("10.00")),
    "gemini-2.5-flash": (Decimal("0.30"), Decimal("0.03"), Decimal("2.50")),
    "gemini-2.5-flash-lite": (Decimal("0.10"), Decimal("0.01"), Decimal("0.40")),
}

_ONE_MILLION = Decimal("1000000")


def _normalize_model_name(model: str) -> str:
    """Strip provider prefixes like ``openai/``, ``anthropic/``, ``gemini/``."""
    for prefix in ("openai/", "anthropic/", "gemini/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def get_model_pricing(model: str) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
    """Return ``(input, cached_input, output)`` per-1M-token prices, or *None*."""
    return _PRICING.get(_normalize_model_name(model))


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


__all__ = ["get_model_pricing", "calculate_cost"]
