"""PII scrubbing for Sentry events.

Strips personal data from Sentry event payloads before transmission:
request headers (Authorization, Cookie), request cookies, request body,
event.user (keeps id only), values under sensitive keys in extra/contexts,
and breadcrumb message bodies for SQL / HTTP categories.
"""

from __future__ import annotations

from typing import Any

_DENY_KEYS = frozenset({
    "prompt",
    "messages",
    "content",
    "raw_output",
    "email",
    "password",
    "passwd",
    "token",
    "authorization",
    "cookie",
    "cookies",
    "api_key",
    "apikey",
    "secret",
    "session",
    "csrftoken",
})

_SENSITIVE_HEADER_KEYS = frozenset({
    "authorization",
    "cookie",
    "x-api-key",
    "x-auth-token",
    "proxy-authorization",
})

_SCRUB_BREADCRUMB_CATEGORIES = frozenset({"query", "httplib", "http"})

REDACTED = "[redacted]"


def _scrub_mapping(obj: Any) -> Any:
    """Walk obj recursively, replacing values whose key matches the deny list."""
    if isinstance(obj, dict):
        return {
            k: (
                REDACTED
                if isinstance(k, str) and k.lower() in _DENY_KEYS
                else _scrub_mapping(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_scrub_mapping(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_mapping(item) for item in obj)
    return obj


def _scrub_request(request: dict) -> dict:
    headers = request.get("headers")
    if isinstance(headers, dict):
        for key in list(headers.keys()):
            if isinstance(key, str) and key.lower() in _SENSITIVE_HEADER_KEYS:
                headers[key] = REDACTED
    if "cookies" in request:
        request["cookies"] = REDACTED
    if "data" in request:
        request["data"] = REDACTED
    query_string = request.get("query_string")
    if isinstance(query_string, str):
        lower = query_string.lower()
        if any(k in lower for k in _DENY_KEYS):
            request["query_string"] = REDACTED
    return request


def scrub_event(event: dict | None) -> dict | None:
    """Remove PII from a Sentry event in place. Returns the event (or None)."""
    if not event:
        return event

    user = event.get("user")
    if isinstance(user, dict):
        event["user"] = {k: v for k, v in user.items() if k == "id"}

    request = event.get("request")
    if isinstance(request, dict):
        event["request"] = _scrub_request(request)

    for key in ("extra", "contexts"):
        if key in event:
            event[key] = _scrub_mapping(event[key])

    breadcrumbs = event.get("breadcrumbs")
    values = None
    if isinstance(breadcrumbs, dict):
        values = breadcrumbs.get("values")
    elif isinstance(breadcrumbs, list):
        values = breadcrumbs
    if isinstance(values, list):
        for crumb in values:
            if not isinstance(crumb, dict):
                continue
            category = crumb.get("category", "")
            if (
                isinstance(category, str)
                and category.lower() in _SCRUB_BREADCRUMB_CATEGORIES
            ):
                if "message" in crumb:
                    crumb["message"] = REDACTED
                if "data" in crumb:
                    crumb["data"] = REDACTED

    return event


__all__ = ["scrub_event", "REDACTED"]
