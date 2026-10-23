"""Decisions: structured yes/no, choice and score questions through one call.

    from llm import get_decisions_service
    from llm.types.decisions import Choice, Question

    result = get_decisions_service().decide(
        [
            Question(type="predicate", name="disclosed",
                     instructions="Has the invention already been publicly disclosed?"),
            Question(type="choice", name="field", instructions="Which technology field?",
                     choices=[Choice(value="chemistry"), Choice(value="software"), Choice(value="other")]),
        ],
        input=disclosure_text,
        feature_key="invention_triage",
        org_id=org.id,
    )
    if not result.refused("disclosed") and result["disclosed"].probability > 0.8:
        ...

Two pathways, identical answers (``llm.types.decisions``) — a caller never sees
which one answered:

* **quick** — OpenAI's Decisions endpoint (``llm.service.decisions_quick``),
  models with ``ModelInfo.supports_decisions_api``; ~2 s, input-only billing.
* **structured** — an ordinary structured-output call on any chat model
  (``llm.service.decisions_structured``).

Which runs is the org's business (``core.preferences.resolve_decisions_settings``):
the "Quick decisions" toggle, the quick and structured feature models, and a
pinned reasoning level for the structured model. Quick is used when the toggle
is on, a capable model is allowed, ``OPENAI_API_KEY`` is set and no skip window
is open; otherwise — and whenever a quick call fails after the SDK's retries —
the structured pathway answers (the failed batch is requeued, nothing is lost,
and every process skips quick for ``DECISIONS_QUICK_SKIP_SECONDS``). Programmer
errors (bad questions, an input over ``DECISIONS_MAX_INPUT_TOKENS``, a 400 from
the endpoint) raise instead of falling back.

Writing an application:

* Put the long, shared framing in ``input`` (a developer message for rules; the
  endpoint caches the input prefix) and keep per-question ``instructions`` short.
* Give choice questions an ``other``/``none`` option so nothing is forced.
* Questions in one call are independent; run a decision that depends on another
  answer as a second call.
* Pass a stable ``feature_key`` — it tags every ``LLMCallLog`` row
  (``response_metadata.kind == "decisions"``) and groups the endpoint's cache.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import Counter
from typing import Sequence

from django.conf import settings

from llm.service.decisions_quick import MAX_QUESTIONS, QuickAvailability, QuickUnavailable, call_quick
from llm.service.decisions_structured import run_structured_batch
from llm.service.errors import LLMConfigurationError, LLMProviderError, LLMRequestTooLargeError
from llm.service.logger import log_decisions, log_decisions_error
from llm.service.pricing import calculate_cost, calculate_decisions_cost
from llm.types.context import RunContext
from llm.types.decisions import (
    Answer,
    CallRecord,
    DecisionInput,
    DecisionsResult,
    Question,
    chunk_questions,
    estimate_input_tokens,
    normalize_input,
    sum_usage,
    validate_questions,
)

logger = logging.getLogger(__name__)

QUICK_MAX_QUESTIONS = MAX_QUESTIONS


class DecisionsService:
    """Answers decision questions through the org's configured pathway(s)."""

    def __init__(self, availability: QuickAvailability | None = None):
        self._availability = availability or QuickAvailability()

    def decide(
        self,
        questions: Sequence[Question | dict],
        input: DecisionInput,
        *,
        feature_key: str,
        org_id: int | None,
        context: RunContext | None = None,
        prompt_cache_key: str | None = None,
    ) -> DecisionsResult:
        """Answer ``questions`` about ``input``; answers come back in question order.

        Raises ``ValueError``/``ValidationError`` for bad questions or input,
        ``LLMRequestTooLargeError`` when the input is over the cap (or the
        endpoint says so), ``LLMConfigurationError`` when no model resolves for
        either pathway, ``LLMProviderError(error_code="invalid_request")`` for a
        400 from the endpoint, and lets the structured pathway's errors other
        than a refusal propagate.
        """
        from core.preferences import resolve_decisions_settings

        if not feature_key:
            raise ValueError("feature_key is required")
        questions = validate_questions(questions)
        messages = normalize_input(input)
        cap = int(settings.DECISIONS_MAX_INPUT_TOKENS)
        estimate = estimate_input_tokens(messages)
        if cap and estimate > cap:
            raise LLMRequestTooLargeError(
                f"decisions input is ~{estimate} tokens; the cap is {cap} (DECISIONS_MAX_INPUT_TOKENS)"
            )
        if context is None:
            context = RunContext.create()

        prefs = resolve_decisions_settings(org_id)
        cache_key = prompt_cache_key or f"decisions:{feature_key}"
        quick_ok, quick_reason = self._quick_available(prefs)
        if not quick_ok and not prefs.structured_model:
            raise LLMConfigurationError(
                f"No model is available for decisions (quick pathway: {quick_reason}; "
                "no structured model is allowed)"
            )
        if not quick_ok:
            logger.debug(
                "decisions: structured pathway (quick %s) feature=%s run_id=%s",
                quick_reason, feature_key, context.run_id,
            )

        remaining = list(questions)
        by_name: dict[str, Answer] = {}
        calls: list[CallRecord] = []
        fell_back = False
        quick_batches = len(chunk_questions(questions, QUICK_MAX_QUESTIONS))
        quick_index = 0

        while remaining:
            if quick_ok:
                batch, remaining = remaining[:QUICK_MAX_QUESTIONS], remaining[QUICK_MAX_QUESTIONS:]
                quick_index += 1
                t0 = time.perf_counter()
                try:
                    answers, usage, provider_model = call_quick(
                        batch, messages, model=prefs.quick_model, prompt_cache_key=cache_key
                    )
                except QuickUnavailable as exc:
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    log_decisions_error(
                        prefs.quick_model, context, exc, duration_ms,
                        feature_key=feature_key, questions=len(batch),
                        batch=quick_index, batches=quick_batches, error_code=exc.error_code,
                    )
                    calls.append(CallRecord(
                        pathway="quick", model=prefs.quick_model, questions=len(batch),
                        duration_ms=duration_ms, error_code=exc.error_code,
                    ))
                    self._availability.suspend(exc.error_code, exc.log_level)
                    if not prefs.structured_model:
                        from llm.core.providers.base import exception_for_error_code

                        # The typed subclass for known codes; the code itself is
                        # passed so an unmapped one (malformed_response) survives.
                        raise exception_for_error_code(exc.error_code)(
                            f"Decisions endpoint failed ({exc.error_code}) and no structured model is allowed",
                            error_code=exc.error_code,
                        ) from exc
                    logger.info(
                        "decisions: quick failed (%s) after %d ms; structured pathway takes over "
                        "feature=%s questions=%d run_id=%s",
                        exc.error_code, duration_ms, feature_key, len(batch) + len(remaining), context.run_id,
                    )
                    quick_ok = False
                    fell_back = True
                    remaining = batch + remaining
                    continue
                except (LLMRequestTooLargeError, LLMProviderError) as exc:
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    log_decisions_error(
                        prefs.quick_model, context, exc, duration_ms,
                        feature_key=feature_key, questions=len(batch),
                        batch=quick_index, batches=quick_batches, error_code=exc.error_code,
                    )
                    raise

                duration_ms = int((time.perf_counter() - t0) * 1000)
                cost = calculate_decisions_cost(
                    prefs.quick_model, usage.prompt_tokens, usage.cached_tokens, usage.cache_write_tokens
                )
                if cost is not None:
                    usage.cost_usd = float(cost)
                refusals = sum(1 for a in answers if a.type == "refusal")
                log_decisions(
                    prefs.quick_model, context,
                    feature_key=feature_key, questions=len(batch), refusals=refusals,
                    types=dict(Counter(q.type for q in batch)),
                    batch=quick_index, batches=quick_batches,
                    input_tokens=usage.prompt_tokens, cached_tokens=usage.cached_tokens,
                    cache_write_tokens=usage.cache_write_tokens, total_tokens=usage.total_tokens,
                    cost_usd=cost, duration_ms=duration_ms,
                    provider_model_id=provider_model, prompt_cache_key=cache_key,
                )
                calls.append(CallRecord(
                    pathway="quick", model=prefs.quick_model, questions=len(batch),
                    duration_ms=duration_ms, usage=usage,
                ))
                logger.info(
                    "decisions quick: feature=%s questions=%d refusals=%d input_tokens=%s cached=%s "
                    "wall_ms=%d run_id=%s",
                    feature_key, len(batch), refusals, usage.prompt_tokens, usage.cached_tokens,
                    duration_ms, context.run_id,
                )
                for answer in answers:
                    by_name[answer.name] = answer
            else:
                batches = chunk_questions(remaining, max(1, int(settings.DECISIONS_STRUCTURED_BATCH_SIZE)))
                remaining = []
                for index, batch in enumerate(batches, 1):
                    answers, usage, duration_ms = run_structured_batch(
                        batch, messages,
                        model=prefs.structured_model,
                        thinking_level=prefs.structured_thinking_level,
                        feature_key=feature_key, context=context,
                        batch_index=index, batch_count=len(batches),
                    )
                    if usage is not None and usage.cost_usd is None:
                        cost = calculate_cost(
                            prefs.structured_model, usage.prompt_tokens, usage.completion_tokens,
                            usage.cached_tokens, usage.cache_write_tokens,
                        )
                        if cost is not None:
                            usage.cost_usd = float(cost)
                    calls.append(CallRecord(
                        pathway="structured", model=prefs.structured_model, questions=len(batch),
                        duration_ms=duration_ms, usage=usage,
                    ))
                    for answer in answers:
                        by_name[answer.name] = answer

        return DecisionsResult(
            answers=[by_name[q.name] for q in questions],
            usage=sum_usage(c.usage for c in calls if not c.error_code),
            calls=calls,
            fell_back=fell_back,
        )

    async def adecide(self, questions, input, **kwargs) -> DecisionsResult:
        """``decide`` off the event loop (the quick call and the structured call
        are both synchronous)."""
        return await asyncio.to_thread(self.decide, questions, input, **kwargs)

    def _quick_available(self, prefs) -> tuple[bool, str]:
        """Whether the quick pathway may run now, else why not."""
        if not prefs.quick_enabled:
            return False, "off"
        if not prefs.quick_model:
            return False, "no_capable_model"
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "no_openai_key"
        if self._availability.is_suspended():
            return False, "skipped"
        return True, ""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_service: DecisionsService | None = None
_global_service_lock = threading.Lock()


def get_decisions_service() -> DecisionsService:
    """Return the process-wide DecisionsService singleton (thread-safe)."""
    global _global_service
    if _global_service is None:
        with _global_service_lock:
            if _global_service is None:
                _global_service = DecisionsService()
    return _global_service


__all__ = ["DecisionsService", "QUICK_MAX_QUESTIONS", "get_decisions_service"]
