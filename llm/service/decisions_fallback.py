"""Decisions first, the caller's legacy classifier only when Decisions cannot answer.

``decide_with_fallback`` is the one entry point the production classification jobs
(chunk scan, the text classifiers, the PII scan) use for their Layer-1 verdict:

1. ``get_decisions_service().decide(...)`` — the org's pathway (quick, which itself
   falls back to structured inside the service).
2. ``interpret(result)`` turns the answers into the job's own verdict object. It
   raises :class:`DecisionsRefused` when a required answer is a refusal.
3. Only when step 1 or 2 fails does ``fallback(meta)`` run — the job's legacy
   classifier. ``meta`` is the ``_log_metadata`` dict the legacy call passes on, so
   every fallback is a countable ``LLMCallLog`` row:
   ``LLMCallLog.objects.filter(response_metadata__kind="decisions_fallback")``.

Bad questions or input (``ValueError``/``ValidationError``) are validated up front
and propagate: they are bugs, and the legacy path would not fix them. Anything that
fails after that — provider errors, configuration errors, a refusal, an unexpected
exception from the structured pathway — takes the fallback, because these jobs gate
documents and messages and must still produce a verdict.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence, TypeVar

from django.core.cache import cache

from llm.service.errors import LLMError
from llm.types.context import RunContext
from llm.types.decisions import DecisionInput, DecisionsResult, Question, normalize_input, validate_questions

logger = logging.getLogger(__name__)

FALLBACK_KIND = "decisions_fallback"
# One WARNING (a Sentry event) per feature per window; the rest log at INFO. Every
# fallback still writes its own tagged LLMCallLog row.
WARN_WINDOW_SECONDS = 60

T = TypeVar("T")


class DecisionsRefused(LLMError):
    """A required answer came back as a refusal."""


def _get_decisions_service():
    from llm.service.decisions_service import get_decisions_service

    return get_decisions_service()


def _log_fallback(feature_key: str, exc: BaseException, run_id: str) -> None:
    try:
        first = cache.add(f"decisions_fallback_warned:{feature_key}", 1, WARN_WINDOW_SECONDS)
    except Exception:
        first = True
    log = logger.warning if first else logger.info
    log(
        "decisions fallback: feature=%s error=%s code=%s run_id=%s — the legacy classifier answers",
        feature_key, type(exc).__name__, getattr(exc, "error_code", ""), run_id,
        exc_info=first,
    )


def decide_with_fallback(
    questions: Sequence[Question | dict],
    input: DecisionInput,
    *,
    feature_key: str,
    org_id: int | None,
    interpret: Callable[[DecisionsResult], T],
    fallback: Callable[[dict[str, Any]], T],
    context: RunContext | None = None,
    prompt_cache_key: str | None = None,
) -> T:
    """The job's verdict from Decisions, or from ``fallback(meta)`` when Decisions fails."""
    questions = validate_questions(questions)
    normalize_input(input)  # raises on bad input before any call
    context = context or RunContext.create()
    try:
        result = _get_decisions_service().decide(
            questions, input, feature_key=feature_key, org_id=org_id,
            context=context, prompt_cache_key=prompt_cache_key,
        )
        return interpret(result)
    except Exception as exc:  # noqa: BLE001 — a gating job must still get a verdict
        _log_fallback(feature_key, exc, context.run_id)
        return fallback({
            "kind": FALLBACK_KIND,
            "feature": feature_key,
            "decide_error": type(exc).__name__,
            "error_code": str(getattr(exc, "error_code", "") or ""),
        })


__all__ = ["FALLBACK_KIND", "DecisionsRefused", "decide_with_fallback"]
