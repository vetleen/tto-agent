"""Model context window registry and history budget calculation."""

from __future__ import annotations

from llm.service.pricing import _normalize_model_name

_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.4": 1_000_000,
    "gpt-5-mini": 1_000_000,
    "gpt-5-nano": 128_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
}

_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model: str | None) -> int:
    """Return the context window size for *model*, or the default for unknown models."""
    if not model:
        return _DEFAULT_CONTEXT_WINDOW
    return _CONTEXT_WINDOWS.get(_normalize_model_name(model), _DEFAULT_CONTEXT_WINDOW)


def get_history_budget(model: str | None, max_context_tokens: int | None = None) -> int:
    """75% of context window, capped at 150k, further capped by max_context_tokens."""
    context = get_context_window(model)
    if max_context_tokens is not None:
        context = min(context, max_context_tokens)
    return min(int(context * 0.75), 150_000)


__all__ = ["get_context_window", "get_history_budget"]
