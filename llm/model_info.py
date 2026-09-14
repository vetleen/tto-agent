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


def get_history_budget(
    model: str | None,
    max_context_tokens: int | None = None,
    *,
    effort: str | None = None,
    reserved_tokens: int | None = None,
) -> int:
    """Tokens available for conversation history.

    Delegates to the measured budget in :mod:`llm.context_budget`: the request
    input ceiling (``min(setting, model window) − output reservation − margin``)
    minus the system-prompt + tool-schema + current-message overhead. Replaces
    the former blind ``context * 0.75`` capped at 150k — that fraction both
    over-reserved on large windows and, crucially, failed to reserve output on
    small ones. ``max_context_tokens`` is the aim; the model window is the hard cap.
    """
    from llm.context_budget import history_budget

    return history_budget(
        model, max_context_tokens, effort=effort, reserved_tokens=reserved_tokens
    )


__all__ = ["get_context_window", "get_history_budget"]
