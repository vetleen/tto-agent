"""Measured per-request token budget.

Replaces the old blind ``min(window, setting) * 0.75, capped 150k`` fraction with
an explicit reservation model:

    request_input_ceiling = min(max_context_tokens, model_window)
                            - output_reservation      # output counts against the window
                            - safety_margin           # tokenizer estimate slack
    history_budget        = request_input_ceiling - input_overhead(system + tools + current)

Two facts drive this (verified against the model registry + provider docs):
- **Output counts against the context window** on every provider, so the tokens
  we're about to generate must be reserved or a long input + a long answer
  overflows the model's hard window (a 400).
- ``max_context_tokens`` is the *aim* (org/user cost knob); the model's registry
  window is the *hard* cap. We take the min and keep a margin below it.
"""

from __future__ import annotations

from django.conf import settings

from llm.model_info import get_context_window
from llm.model_registry import get_model_info

# Output reservation by effort — roughly the max_tokens we send (see
# llm/core/providers/anthropic.py for the Anthropic thinking map). Capped per
# request by the model's registry max_output_tokens.
_EFFORT_OUTPUT_RESERVATION = {
    "low": 16_384,
    "medium": 16_384,
    "high": 40_000,
    "xhigh": 48_000,
    "max": 64_000,
}
# Used when effort is unknown — a conservative middle so a big answer never
# overflows the window.
_DEFAULT_OUTPUT_RESERVATION = 32_000


def output_reservation(model: str | None, effort: str | None = None) -> int:
    """Tokens to reserve for the model's response (including thinking).

    Scales with effort, capped by the model's registry ``max_output_tokens``.
    """
    base = _EFFORT_OUTPUT_RESERVATION.get(
        (effort or "").lower(), _DEFAULT_OUTPUT_RESERVATION
    )
    info = get_model_info(model) if model else None
    cap = getattr(info, "max_output_tokens", None) if info else None
    return min(base, cap) if cap else base


def _margin() -> int:
    return int(getattr(settings, "CONTEXT_SAFETY_MARGIN_TOKENS", 8_000))


def request_input_ceiling(
    model: str | None, max_context_tokens: int | None = None, effort: str | None = None
) -> int:
    """Max INPUT tokens for one request: ``min(aim, model window)`` minus the
    output reservation and a safety margin. Never exceeds the model's window."""
    window = get_context_window(model)
    aim = min(window, max_context_tokens) if max_context_tokens else window
    ceiling = aim - output_reservation(model, effort) - _margin()
    return max(ceiling, 1_000)


def history_budget(
    model: str | None,
    max_context_tokens: int | None = None,
    *,
    effort: str | None = None,
    reserved_tokens: int | None = None,
) -> int:
    """Tokens available for conversation history.

    ``reserved_tokens`` is the measured system-prompt + tool-schema + current-
    message footprint when the caller has it; otherwise a conservative flat
    overhead is used (the per-round pruner is the exact backstop within a turn).
    """
    ceiling = request_input_ceiling(model, max_context_tokens, effort)
    overhead = (
        reserved_tokens
        if reserved_tokens is not None
        else int(getattr(settings, "CONTEXT_INPUT_OVERHEAD_TOKENS", 24_000))
    )
    floor = int(getattr(settings, "MIN_HISTORY_BUDGET_TOKENS", 4_000))
    return max(ceiling - overhead, floor)


__all__ = ["output_reservation", "request_input_ceiling", "history_budget"]
