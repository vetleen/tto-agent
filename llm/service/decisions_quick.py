"""Quick pathway of the decisions service: OpenAI's Decisions endpoint.

``POST /v1/decisions`` takes a shared input plus up to ``MAX_QUESTIONS`` typed
questions and returns typed answers (probabilities, a choice, a score, or a
refusal per question) in a couple of seconds, billing input tokens only. Our
pinned ``openai`` SDK has no ``decisions`` resource yet, so the call goes through
the SDK's generic ``client.post`` — auth, base URL, retries with exponential
backoff (408/409/429/5xx/connection errors) and per-request timeouts included.

Failure handling splits in two (see ``classify_quick_error``):

* ``QuickUnavailable`` — the endpoint could not answer this batch (timeout, rate
  limit, 5xx, connection, auth, unexpected status, malformed body). The service
  falls back to the structured pathway for the request and opens the skip window
  (``QuickAvailability``) so every process stops trying the endpoint for a while.
* Programmer errors raise straight to the caller: ``LLMRequestTooLargeError`` for
  an input the model cannot take, ``LLMProviderError(error_code="invalid_request")``
  for any other 400/422 (a bad question is our bug, and the structured pathway
  would not fix it).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Sequence

import httpx
from django.conf import settings
from pydantic import TypeAdapter, ValidationError

from llm.model_registry import get_model_info
from llm.service.errors import LLMProviderError, LLMRequestTooLargeError
from llm.service.openai_client import get_openai_client
from llm.types.decisions import Answer, InputMessage, Question
from llm.types.responses import Usage

logger = logging.getLogger(__name__)

API_PATH = "/decisions"
# The endpoint's per-request question cap.
MAX_QUESTIONS = 400
# Skip-window episode marker, shared by web + worker through the Django cache.
_SKIP_KEY = "decisions_quick_skip:v1"

_answer_adapter: TypeAdapter = TypeAdapter(Answer)


class QuickUnavailable(Exception):
    """The quick pathway could not answer this batch; fall back and skip it for a while.

    ``error_code`` is the classified cause (``timeout``, ``rate_limited``,
    ``overloaded``, ``server_error``, ``connection_error``, ``auth_error``,
    ``unknown``, ``malformed_response``); ``log_level`` says how loud the start
    of a skip episode should be (``auth_error`` and a body we cannot read are
    errors — they will not fix themselves).
    """

    def __init__(self, message: str, *, error_code: str, log_level: str = "warning"):
        super().__init__(message)
        self.error_code = error_code
        self.log_level = log_level


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def api_model_id(model: str) -> str:
    """The provider's model id for a registry id (``openai/gpt-6-luna`` → ``gpt-6-luna``)."""
    info = get_model_info(model)
    if info is not None:
        return info.api_model
    return model.split("/", 1)[-1]


def build_body(
    batch: Sequence[Question],
    input: Sequence[InputMessage],
    *,
    model: str,
    prompt_cache_key: str | None = None,
) -> dict:
    """The request body for one batch (the types dump straight to the wire shape)."""
    body = {
        "model": api_model_id(model),
        "input": [message.model_dump() for message in input],
        "questions": [q.to_api() for q in batch],
    }
    if prompt_cache_key:
        body["prompt_cache_key"] = prompt_cache_key
    return body


def _int_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_usage(raw) -> Usage:
    if not isinstance(raw, dict):
        return Usage(completion_tokens=0)  # the endpoint never bills output
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    return Usage(
        prompt_tokens=_int_or_none(raw.get("input_tokens")),
        completion_tokens=_int_or_none(raw.get("output_tokens")) or 0,
        total_tokens=_int_or_none(raw.get("total_tokens")),
        cached_tokens=_int_or_none(input_details.get("cached_tokens")),
        cache_write_tokens=_int_or_none(input_details.get("cache_write_tokens")),
        reasoning_tokens=_int_or_none(output_details.get("reasoning_tokens")),
    )


def parse_response(batch: Sequence[Question], raw) -> tuple[list[Answer], Usage, str]:
    """Typed answers in batch order, usage and the provider's model id.

    Any shape surprise — not a dict, no ``answers`` list, an answer that does not
    validate, or answer names that do not match the batch — is
    ``QuickUnavailable("malformed_response")`` (error level: schema drift).
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), list):
        raise QuickUnavailable(
            "Decisions endpoint returned a body without an answers list",
            error_code="malformed_response", log_level="error",
        )
    by_name: dict[str, Answer] = {}
    try:
        for item in raw["answers"]:
            answer = _answer_adapter.validate_python(item)
            by_name[answer.name] = answer
    except ValidationError as exc:
        raise QuickUnavailable(
            f"Decisions endpoint returned an answer we cannot read: {exc.error_count()} validation error(s)",
            error_code="malformed_response", log_level="error",
        ) from None
    expected = [q.name for q in batch]
    if set(by_name) != set(expected) or len(raw["answers"]) != len(expected):
        raise QuickUnavailable(
            f"Decisions endpoint answers do not match the questions "
            f"(got {len(raw['answers'])}, expected {len(expected)})",
            error_code="malformed_response", log_level="error",
        )
    return [by_name[name] for name in expected], _parse_usage(raw.get("usage")), str(raw.get("model") or "")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def classify_quick_error(exc: Exception) -> Exception:
    """The exception to raise for a failed ``client.post``: a programmer error
    (``LLMRequestTooLargeError`` / ``LLMProviderError(invalid_request)``) or a
    ``QuickUnavailable`` carrying the classified code and log level."""
    from llm.core.providers.base import classify_api_error

    classified = classify_api_error(exc, "OpenAI")
    if classified.error_code == "request_too_large":
        return LLMRequestTooLargeError(f"Decisions endpoint: {exc}")
    status = getattr(exc, "status_code", None)
    if status in (400, 422):
        return LLMProviderError(
            f"Decisions endpoint rejected the request: {exc}", error_code="invalid_request"
        )
    return QuickUnavailable(str(exc), error_code=classified.error_code, log_level=classified.log_level)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def call_quick(
    batch: Sequence[Question],
    input: Sequence[InputMessage],
    *,
    model: str,
    prompt_cache_key: str | None = None,
) -> tuple[list[Answer], Usage, str]:
    """One ``POST /v1/decisions`` for one batch → (answers, usage, provider model id).

    The SDK retries transient failures with exponential backoff
    (``DECISIONS_QUICK_MAX_RETRIES``); each attempt waits at most
    ``DECISIONS_QUICK_TIMEOUT_SECONDS`` for the response. Raises per
    ``classify_quick_error`` / ``parse_response``.
    """
    body = build_body(batch, input, model=model, prompt_cache_key=prompt_cache_key)
    options = {
        "timeout": httpx.Timeout(
            connect=10.0,
            read=float(settings.DECISIONS_QUICK_TIMEOUT_SECONDS),
            write=30.0,
            pool=10.0,
        ),
        "max_retries": int(settings.DECISIONS_QUICK_MAX_RETRIES),
    }
    try:
        raw = get_openai_client().post(API_PATH, cast_to=object, body=body, options=options)
    except Exception as exc:
        raise classify_quick_error(exc) from exc
    return parse_response(batch, raw)


# ---------------------------------------------------------------------------
# Skip window
# ---------------------------------------------------------------------------


class QuickAvailability:
    """After the endpoint fails, skip it for ``DECISIONS_QUICK_SKIP_SECONDS``.

    The episode is shared by every process (web + worker) through the Django
    cache — ``cache.add`` starts it, so an outage costs one failed call per window
    system-wide and is one Sentry event, not one per request. A local monotonic
    mirror keeps the hold in this process when the cache is down. Pattern:
    ``llm.tools.epo_ops._ServiceThrottle``.
    """

    def __init__(self, key: str = _SKIP_KEY):
        self._key = key
        self._lock = threading.Lock()
        self._skipped_until = 0.0

    @staticmethod
    def window() -> float:
        return float(settings.DECISIONS_QUICK_SKIP_SECONDS)

    def suspend(self, error_code: str, log_level: str = "warning") -> bool:
        """Open (or extend) the skip window; True when this starts an episode.

        A new episode logs once at ``log_level`` (WARNING/ERROR → a Sentry event);
        a failure during a held episode only extends it, at INFO. Never raises.
        """
        window = self.window()
        if window <= 0:
            return False
        with self._lock:
            self._skipped_until = time.monotonic() + window
        from django.core.cache import cache

        try:
            new = bool(cache.add(self._key, 1, window))
            if not new:
                cache.set(self._key, 1, window)
        except Exception:
            new = True  # can't tell without the cache: report it
        if new:
            log = logger.error if log_level == "error" else logger.warning
            log(
                "Decisions endpoint unavailable (%s); using the structured pathway for %.0f s",
                error_code, window,
            )
        else:
            logger.info("Decisions endpoint still unavailable (%s); skip window extended", error_code)
        return new

    def is_suspended(self) -> bool:
        """Whether a skip window is open (opened by any process)."""
        with self._lock:
            if self._skipped_until > time.monotonic():
                return True
        from django.core.cache import cache

        try:
            return bool(cache.get(self._key))
        except Exception:
            return False

    def reset(self) -> None:
        """Close the window (tests)."""
        with self._lock:
            self._skipped_until = 0.0
        from django.core.cache import cache

        try:
            cache.delete(self._key)
        except Exception:
            pass


__all__ = [
    "API_PATH",
    "MAX_QUESTIONS",
    "QuickAvailability",
    "QuickUnavailable",
    "api_model_id",
    "build_body",
    "call_quick",
    "classify_quick_error",
    "parse_response",
]
