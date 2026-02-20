"""
Fallback pricing when LiteLLM does not return cost (e.g. new or custom models).
USD per 1M tokens; used only for allowed models when cost is missing.
"""
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# model string or price_plan -> { "input": USD per 1M tokens, "output": USD per 1M tokens }
# Add entries for LLM_ALLOWED_MODELS as needed. Sources: provider pricing pages (standard tier).
FALLBACK_PRICING: dict[str, dict[str, Decimal]] = {
    # OpenAI (Standard tier) https://platform.openai.com/docs/pricing
    "openai/gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
    "openai/gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "openai/gpt-5.2": {"input": Decimal("1.75"), "output": Decimal("14.00")},
    "openai/gpt-5-mini": {"input": Decimal("0.25"), "output": Decimal("2.00")},
    "openai/gpt-5-nano": {"input": Decimal("0.05"), "output": Decimal("0.40")},
    # Anthropic (≤200K prompts) https://www.anthropic.com/pricing
    "anthropic/claude-sonnet-4-5-20250929": {"input": Decimal("3.00"), "output": Decimal("15.00")},
    "anthropic/claude-opus-4-1-20250805": {"input": Decimal("15.00"), "output": Decimal("75.00")},
    "anthropic/claude-3-5-haiku-20241022": {"input": Decimal("0.25"), "output": Decimal("1.25")},
    "anthropic/claude-3-5-sonnet-20241022": {"input": Decimal("3.00"), "output": Decimal("15.00")},
    # Gemini (Paid tier, ≤200K where applicable) https://ai.google.dev/pricing
    "gemini/gemini-3-pro-preview": {"input": Decimal("2.00"), "output": Decimal("12.00")},
    "gemini/gemini-3-flash-preview": {"input": Decimal("0.50"), "output": Decimal("3.00")},
    "gemini/gemini-2.5-pro": {"input": Decimal("1.25"), "output": Decimal("10.00")},
    "gemini/gemini-2.5-flash": {"input": Decimal("0.30"), "output": Decimal("2.50")},
    "gemini/gemini-2.5-flash-lite": {"input": Decimal("0.10"), "output": Decimal("0.40")},
    "gemini/gemini-2.0-flash": {"input": Decimal("0.10"), "output": Decimal("0.40")},
    # Moonshot / KIMI (input = cache-miss rate) https://platform.moonshot.ai/docs/pricing/chat
    "moonshot/kimi-k2.5": {"input": Decimal("0.60"), "output": Decimal("3.00")},
    "moonshot/kimi-k2-thinking": {"input": Decimal("0.60"), "output": Decimal("2.50")},
}


def get_fallback_cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Decimal | None:
    """Compute cost from fallback table if model is present. Returns None if no price."""
    pricing = FALLBACK_PRICING.get(model)
    if not pricing:
        return None
    input_per_m = Decimal(input_tokens) / Decimal(1_000_000)
    output_per_m = Decimal(output_tokens) / Decimal(1_000_000)
    cost = input_per_m * pricing["input"] + output_per_m * pricing["output"]
    return cost.quantize(Decimal("0.00000001"))
