"""Model context window registry and history budget calculation."""

from __future__ import annotations

from llm.model_registry import get_model_info

_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model: str | None) -> int:
    """Return the context window size for *model*, or the default for unknown models."""
    if not model:
        return _DEFAULT_CONTEXT_WINDOW
    info = get_model_info(model)
    if info:
        return info.context_window
    return _DEFAULT_CONTEXT_WINDOW


def get_history_budget(model: str | None, max_context_tokens: int | None = None) -> int:
    """75% of context window, capped at 150k, further capped by max_context_tokens."""
    context = get_context_window(model)
    if max_context_tokens is not None:
        context = min(context, max_context_tokens)
    return min(int(context * 0.75), 150_000)


__all__ = ["get_context_window", "get_history_budget"]
