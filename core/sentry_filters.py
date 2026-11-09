"""Sentry event filters applied in ``before_send``.

Sentry's LangChain integration (enabled automatically) captures every exception a
chat model raises as an *unhandled* event, before our own code sees it. For
provider API errors that is pure noise: ``llm/core/providers/base.py`` retries
them (rate limits, overloads, Gemini's bare 400), and when the retries run out
or the error is not retryable ``_report_to_sentry`` files one clean,
fingerprinted "LLM provider error error_code=…" event. Without this filter each
rate-limited turn produced up to four "unhandled ClientError" events, even when
the retry recovered (WILFRED-A0, WILFRED-AP, 2026-11).

Only provider-SDK exceptions are dropped: a bug inside LangChain or our own
callback code still reaches Sentry through the integration.
"""

from __future__ import annotations

from typing import Any

# Packages whose exceptions are classified, retried and reported by
# llm/core/providers/base.py. Matched against the exception class's module.
_PROVIDER_SDK_MODULE_PREFIXES = (
    "openai",
    "anthropic",
    "google.genai",
    "google.api_core",
    "langchain_google_genai",
    "langchain_openai",
    "langchain_anthropic",
    "httpx",
    "httpcore",
)


def _is_langchain_capture(event: dict) -> bool:
    values = (event.get("exception") or {}).get("values") or []
    for value in values:
        mechanism = value.get("mechanism") if isinstance(value, dict) else None
        if isinstance(mechanism, dict) and mechanism.get("type") == "langchain":
            return True
    return False


def _is_provider_sdk_exception(exc: BaseException | None) -> bool:
    # Walk the cause chain: langchain_google_genai wraps google.genai errors.
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        module = type(exc).__module__ or ""
        if module.startswith(_PROVIDER_SDK_MODULE_PREFIXES):
            return True
        exc = exc.__cause__
    return False


def drop_duplicate_provider_error(event: dict | None, hint: dict | None) -> dict | None:
    """Return None for a LangChain-integration capture of a provider API error."""
    if not event or not _is_langchain_capture(event):
        return event
    exc_info = (hint or {}).get("exc_info")
    exc = exc_info[1] if exc_info and len(exc_info) > 1 else None
    if _is_provider_sdk_exception(exc):
        return None
    return event


__all__ = ["drop_duplicate_provider_error"]
