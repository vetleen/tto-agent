"""Display utilities for LLM model IDs."""

from __future__ import annotations

import re


def get_display_name(model_id: str) -> str:
    """Auto-generate a human-friendly display name from a model ID.

    Examples:
        anthropic/claude-sonnet-4-5-20250929 -> Claude Sonnet 4.5
        openai/gpt-5-mini -> GPT-5 Mini
        gemini/gemini-2.5-flash -> Gemini 2.5 Flash
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
    lower = model_id.lower()

    # All Anthropic models support extended thinking
    if lower.startswith("anthropic/"):
        return True

    # OpenAI reasoning models: o1, o3, o4 series
    name = lower.split("/", 1)[-1] if "/" in lower else lower
    if re.match(r"^o[134]\b", name):
        return True

    # Models with "thinking" in their name
    if "thinking" in lower:
        return True

    return False


def supports_vision(model_id: str) -> bool:
    """Return True if the model supports vision (image) inputs."""
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
