"""Display utilities for LLM model IDs."""

from __future__ import annotations

import re
from decimal import Decimal

from llm.model_registry import TIER_ORDER, get_model_info


def get_display_name(model_id: str) -> str:
    """Auto-generate a human-friendly display name from a model ID.

    Examples:
        anthropic/claude-sonnet-4-5-20250929 -> Claude Sonnet 4.5
        openai/gpt-5-mini -> GPT-5 Mini
        gemini/gemini-3.5-flash -> Gemini 3.5 Flash
    """
    name = model_id

    # Strip provider prefix
    if "/" in name:
        name = name.split("/", 1)[1]

    # Strip date suffixes (e.g. -20250929, -20251001)
    name = re.sub(r"-\d{8}$", "", name)

    # Special-case: claude version patterns like claude-sonnet-4-5 -> Claude Sonnet 4.5
    m = re.match(
        r"^claude-(\w+)-(\d+)-(\d+)(.*)$", name
    )
    if m:
        variant, major, minor, rest = m.groups()
        rest_clean = rest.strip("-") if rest else ""
        parts = ["Claude", variant.title(), f"{major}.{minor}"]
        if rest_clean:
            parts.append(rest_clean.replace("-", " ").title())
        return " ".join(parts)

    # Single-version claude: claude-sonnet-4
    m = re.match(r"^claude-(\w+)-(\d+)$", name)
    if m:
        variant, version = m.groups()
        return f"Claude {variant.title()} {version}"

    # GPT models: uppercase GPT
    if name.lower().startswith("gpt-"):
        return "GPT-" + name[4:].replace("-", " ").title()

    # o-series models (o1, o3, o4-mini, etc.)
    m = re.match(r"^(o\d+)(.*)$", name)
    if m:
        model, rest = m.groups()
        if rest:
            return model.upper() + rest.replace("-", " ").title()
        return model.upper()

    # Gemini models: strip duplicate "gemini-" prefix
    if name.lower().startswith("gemini-"):
        return "Gemini " + name[7:].replace("-", " ").title()

    # Fallback: replace hyphens, title-case
    return name.replace("-", " ").title()


def supports_thinking(model_id: str) -> bool:
    """Return True if the model supports extended thinking / reasoning."""
    info = get_model_info(model_id)
    if info is not None:
        return info.supports_thinking

    # Fallback heuristics for models not in the registry
    lower = model_id.lower()

    # All Anthropic models support extended thinking
    if lower.startswith("anthropic/"):
        return True

    # OpenAI reasoning models: o1, o3, o4 series and GPT-5.4+
    name = lower.split("/", 1)[-1] if "/" in lower else lower
    if re.match(r"^o[134]\b", name):
        return True
    if name.startswith("gpt-5.5") or name.startswith("gpt-5.4") or name.startswith("gpt-5.2-pro"):
        return True

    # Models with "thinking" in their name
    if "thinking" in lower:
        return True

    return False


_MAX_EFFORT_MODELS = {"claude-opus-4-7", "claude-opus-4-8"}


def get_thinking_levels(model_id: str) -> list[str]:
    """Return the thinking levels available for a model."""
    if not supports_thinking(model_id):
        return []
    info = get_model_info(model_id)
    api_model = info.api_model if info else ""
    if api_model in _MAX_EFFORT_MODELS:
        return ["low", "medium", "high", "max"]
    return ["low", "medium", "high"]


def supports_vision(model_id: str) -> bool:
    """Return True if the model supports vision (image) inputs."""
    info = get_model_info(model_id)
    if info is not None:
        return info.supports_vision

    # Fallback heuristics for models not in the registry
    lower = model_id.lower()

    # Strip provider prefix for name matching
    name = lower.split("/", 1)[-1] if "/" in lower else lower

    # Anthropic Claude models
    if lower.startswith("anthropic/") and name.startswith("claude-"):
        return True

    # OpenAI GPT-4+ and GPT-5+ models
    if lower.startswith("openai/") and re.match(r"^gpt-[45]", name):
        return True

    # Gemini models
    if lower.startswith("gemini/") and name.startswith("gemini-"):
        return True

    return False


def input_modalities(model_id: str) -> tuple[str, ...]:
    """Return the input modalities a model accepts (e.g. ("text", "image", "pdf")).

    Registry models report their declared modalities; unknown models fall back
    to heuristics (every current-gen vision model also accepts native PDF).
    """
    info = get_model_info(model_id)
    if info is not None:
        return tuple(info.input_modalities)

    mods = ["text"]
    if supports_vision(model_id):
        mods += ["image", "pdf"]
    return tuple(mods)


def supports_modality(model_id: str, modality: str) -> bool:
    """Return True if the model accepts *modality* (e.g. "image", "pdf") as input."""
    if modality == "text":
        return True
    return modality in input_modalities(model_id)


# Output-price buckets (USD per 1M output tokens) -> "$" count (1-5).
# Upper-inclusive: $ <= $1, $$ <= $5, $$$ <= $15, $$$$ <= $50, $$$$$ > $50.
_PRICE_THRESHOLDS = (
    (Decimal("1"), 1),
    (Decimal("5"), 2),
    (Decimal("15"), 3),
    (Decimal("50"), 4),
)


def get_price_level(model_id: str) -> int:
    """Return a 1-5 cost rating from a model's output price (0 if unknown).

    Buckets are upper-inclusive on USD per 1M output tokens:
    ``<=1 -> 1``, ``<=5 -> 2``, ``<=15 -> 3``, ``<=50 -> 4``, ``>50 -> 5``.
    Drives the ``$``-``$$$$$`` glyphs in the chat model picker.
    """
    info = get_model_info(model_id)
    if info is None or info.output_price is None:
        return 0
    for threshold, level in _PRICE_THRESHOLDS:
        if info.output_price <= threshold:
            return level
    return 5


def get_capability_level(model_id: str) -> int:
    """Return a 1-4 capability rating (0 if unknown).

    The first three stars come from tier (``cheap -> 1``, ``mid -> 2``,
    ``standard -> 3``); the manually-curated ``cutting_edge`` registry flag
    awards a 4th. Drives the star glyphs in the chat model picker.
    """
    info = get_model_info(model_id)
    if info is None:
        return 0
    return TIER_ORDER.get(info.tier, 0) + 1 + (1 if info.cutting_edge else 0)


def _format_output_price(price: Decimal) -> str:
    """Format an output price as ``$25`` (whole) or ``$0.40`` (fractional)."""
    if price == price.to_integral_value():
        return f"${int(price)}"
    return f"${price:.2f}"


def get_model_meta_tooltip(model_id: str) -> str | None:
    """Hover text combining standing and output price, or None if unknown.

    Cutting-edge models lead with "Cutting edge" (explaining their 4th star);
    others show their tier. Example: ``"Cutting edge · $30 / 1M output tokens"``.
    """
    info = get_model_info(model_id)
    if info is None or info.output_price is None:
        return None
    label = "Cutting edge" if info.cutting_edge else info.tier.capitalize()
    return (
        f"{label} · "
        f"{_format_output_price(info.output_price)} / 1M output tokens"
    )
