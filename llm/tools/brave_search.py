"""Brave Search tool — web search via Brave Search API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time

import requests
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)


class _TokenBucketRateLimiter:
    """Process-wide token bucket that gates outgoing requests."""

    def __init__(self, requests_per_second: float, burst: int = 1):
        self._rps = requests_per_second
        self._max_tokens = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + (now - self._last_refill) * self._rps,
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rps
            time.sleep(wait)


_BRAVE_SEARCH_RPM = int(os.environ.get("BRAVE_SEARCH_RPM", "45"))
_brave_rate_limiter = _TokenBucketRateLimiter(
    requests_per_second=_BRAVE_SEARCH_RPM / 60.0, burst=1
)

_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5
_RATE_LIMIT_BACKOFF_SCHEDULE: list[float] = [5.0, 15.0, 30.0, 60.0]

# Brave freshness presets (pd/pw/pm/py) or a custom YYYY-MM-DDtoYYYY-MM-DD range.
_FRESHNESS_PRESET_RE = re.compile(r"^p[dwmy]$")
_FRESHNESS_RANGE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}to\d{4}-\d{2}-\d{2}$")

# Result categories we know how to parse out of the response. Brave also
# supports infobox/locations/summarizer in result_filter, but those have
# different schemas / extra API flows and are deliberately not exposed.
_ALLOWED_CATEGORIES = ("web", "news", "videos", "discussions", "faq")


def _validate_freshness(value: str | None) -> str:
    """Return a valid freshness value or empty string (invalid values ignored)."""
    if not value:
        return ""
    v = value.strip()
    if _FRESHNESS_PRESET_RE.match(v) or _FRESHNESS_RANGE_RE.match(v):
        return v
    return ""


def _normalize_categories(raw: str | list[str] | None) -> list[str]:
    """Normalize a comma-separated string (or list) of categories.

    Unknown categories are dropped; an empty/invalid input falls back to web.
    Order follows _ALLOWED_CATEGORIES for a stable cache key and output.
    """
    if raw is None:
        parts: list[str] = []
    elif isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    requested = {p.strip().lower() for p in parts if p.strip()}
    result = [c for c in _ALLOWED_CATEGORIES if c in requested]
    return result or ["web"]


def _get_api_key() -> str:
    from django.conf import settings

    key = getattr(settings, "BRAVE_SEARCH_API_KEY", "")
    if not key:
        raise ValueError("BRAVE_SEARCH_API_KEY is not configured")
    return key


def _parse_retry_after(response: requests.Response) -> float | None:
    """Parse Retry-After header as float seconds. Returns None if absent/invalid.

    Brave doesn't document Retry-After; kept as a legacy fallback behind
    X-RateLimit-Reset.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_rate_limit_reset(response: requests.Response) -> float | None:
    """Parse X-RateLimit-Reset header (seconds until the quota window resets)."""
    value = response.headers.get("X-RateLimit-Reset")
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_field(text: str) -> str:
    """Clean one Brave text field for LLM consumption.

    Unescape HTML entities first (Brave returns e.g. ``&#x27;``/``&quot;``),
    then strip zero-width chars (incl. any reintroduced via entities), then
    convert ``<strong>`` highlight markers to markdown bold.
    """
    import html

    from llm.tools._text_cleaning import normalize_text, strip_html_bold

    return strip_html_bold(normalize_text(html.unescape(text)))


def _parse_result_item(item: dict, category: str) -> dict:
    """Flatten one Brave result item to a uniform shape."""
    if category == "faq":
        title = item.get("question", "")
        description = item.get("answer", "")
    else:
        title = item.get("title", "")
        description = item.get("description", "")
    age = item.get("age") or item.get("page_age") or ""
    return {
        "type": category,
        "title": _clean_field(title),
        "url": item.get("url", ""),
        "description": _clean_field(description),
        "age": age,
        "extra_snippets": [
            _clean_field(s) for s in item.get("extra_snippets", []) if s
        ],
    }


def _search_core(
    query: str,
    count: int = 5,
    freshness: str = "",
    categories: list[str] | None = None,
    context=None,
) -> dict:
    """Run a Brave web search and return a result dict (no formatting).

    Returns ``{"query", "results": [...], "count"}`` on success or
    ``{"error": "...", "results": []}`` on failure. Each result item:
    ``{type, title, url, description, age, extra_snippets}``.
    """
    categories = categories or ["web"]
    api_key = _get_api_key()

    params: dict = {
        "q": query,
        "count": count,
        "extra_snippets": 1,
        "result_filter": ",".join(categories),
    }
    if freshness:
        params["freshness"] = freshness
    # NOTE: Brave also supports `country` and `search_lang` (defaulting to
    # US/en). Not exposed yet — candidates for org-locale settings or model
    # inputs if US-biased results become a problem.

    headers = {
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "X-Subscription-Token": api_key,
    }
    # Optionally pin the Brave API version (YYYY-MM-DD) to avoid silent
    # behavior changes on their side.
    api_version = os.environ.get("BRAVE_API_VERSION", "")
    if api_version:
        headers["Api-Version"] = api_version

    last_exc = None
    for attempt in range(1 + _MAX_RETRIES):
        try:
            _brave_rate_limiter.acquire()
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for category in categories:
                section = data.get(category) or {}
                for item in section.get("results", []) or []:
                    results.append(_parse_result_item(item, category))

            # Scan for prompt injection (log only, never blocks)
            try:
                combined = "\n".join(
                    " ".join(
                        [r.get("title", ""), r.get("description", "")]
                        + r.get("extra_snippets", [])
                    )
                    for r in results
                )
                if combined.strip():
                    from guardrails.web_content import scan_web_content

                    scan_web_content(
                        combined,
                        user_id=context.user_id if context else None,
                        thread_id=context.conversation_id if context else None,
                        org_id=None,
                        source_label="brave_search",
                    )
            except Exception:
                logger.debug("brave_search: web content scan failed (non-fatal)")

            return {"query": query, "results": results, "count": len(results)}

        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning("Brave Search timeout (attempt %d) query=%r", attempt + 1, query)
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if response.status_code == 429:
                header_wait = _parse_rate_limit_reset(response)
                if header_wait is None:
                    header_wait = _parse_retry_after(response)
                schedule_wait = _RATE_LIMIT_BACKOFF_SCHEDULE[
                    min(attempt, len(_RATE_LIMIT_BACKOFF_SCHEDULE) - 1)
                ]
                wait = min(
                    header_wait if header_wait is not None else schedule_wait,
                    _RATE_LIMIT_BACKOFF_SCHEDULE[-1],
                )
                logger.warning("Brave Search rate limited (attempt %d), waiting %.1fs", attempt + 1, wait)
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    continue
            elif response.status_code < 500:
                body = ""
                detail = ""
                try:
                    body = response.text[:500]
                    detail = response.json().get("error", {}).get("detail", "")
                except Exception:
                    pass
                logger.warning(
                    "Brave Search client error %d query=%r body=%s",
                    response.status_code, query, body,
                )
                error_msg = f"Brave Search API error {response.status_code}"
                if detail:
                    error_msg += f": {detail}"
                error_msg += (
                    ". This is a client-side error that will not resolve by retrying"
                    " with a different query. Consider reporting this issue to the"
                    " user rather than continuing to search."
                )
                return {"error": error_msg, "results": []}
            logger.warning("Brave Search HTTP %d (attempt %d) query=%r", response.status_code, attempt + 1, query)
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("Brave Search request error (attempt %d) query=%r: %s", attempt + 1, query, e)

        if attempt < _MAX_RETRIES:
            time.sleep(_BACKOFF_BASE * (2 ** attempt))

    if isinstance(last_exc, requests.exceptions.HTTPError) and getattr(
        getattr(last_exc, "response", None), "status_code", None
    ) == 429:
        return {
            "error": "Brave Search rate limited after retries."
            " Web search is temporarily unavailable."
            " Consider reporting this to the user rather than continuing to search.",
            "results": [],
        }
    logger.error("Brave Search failed after retries query=%r: %s", query, last_exc)
    return {
        "error": "Brave Search failed after retries."
        " Web search is currently unavailable."
        " Consider reporting this to the user rather than continuing to search.",
        "results": [],
    }


def _format_search_results(data: dict) -> str:
    """Format a _search_core result dict as markdown for the LLM."""
    from llm.tools._text_cleaning import (
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_END,
        EXTERNAL_CONTENT_NOTE,
    )

    if "error" in data:
        return f"Search error: {data['error']}"
    results = data.get("results", [])
    if not results:
        return "No results found."

    lines = [EXTERNAL_CONTENT_BEGIN, EXTERNAL_CONTENT_NOTE, ""]
    for i, r in enumerate(results, 1):
        age_str = f" ({r['age']})" if r.get("age") else ""
        type_str = f" [{r['type']}]" if r.get("type") and r["type"] != "web" else ""
        lines.append(f"**[{i}] {r.get('title') or '(no title)'}**{age_str}{type_str}")
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        if r.get("description"):
            lines.append(r["description"])
        for snippet in r.get("extra_snippets", []):
            lines.append(f"> {snippet}")
        lines.append("")
    lines.append(EXTERNAL_CONTENT_END)
    return "\n".join(lines)


class BraveSearchInput(ReasonBaseModel):
    query: str = Field(description="The search query.")
    count: int = Field(default=5, description="Number of results to return (1-20, default 5).")
    freshness: str = Field(
        default="",
        description=(
            "Filter results by page age: pd (past day), pw (past week), "
            "pm (past month), py (past year), or a date range "
            "YYYY-MM-DDtoYYYY-MM-DD. Empty means no filter. "
            "Invalid values are ignored."
        ),
    )
    categories: str = Field(
        default="web",
        description=(
            "Result categories to include, comma-separated. Default: web. "
            "Add news for recent journalism, discussions for forum threads, "
            "videos for video results, faq for Q&A extracts. "
            "Example: 'web,news'."
        ),
    )


class BraveSearchTool(ContextAwareTool):
    """Search the web using the Brave Search API."""

    name: str = "web_search"
    description: str = (
        "Search the web for current information using Brave Search. "
        "Use this when you need up-to-date information, facts, or data "
        "that may not be in your training data. "
        "Use freshness to restrict results to a recent period, and "
        "categories to include news, discussions, videos or FAQ results."
    )
    args_schema: type[BaseModel] = BraveSearchInput

    def _run(
        self,
        query: str,
        count: int = 5,
        freshness: str = "",
        categories: str = "web",
        **kwargs,
    ) -> str:
        from django.core.cache import cache

        query = query.strip()
        if not query:
            return "Search error: empty search query."

        count = max(1, min(count, 20))
        freshness = _validate_freshness(freshness)
        category_list = _normalize_categories(categories)

        cache_key = "brave_search_v2:" + hashlib.sha256(
            f"{query}:{count}:{freshness}:{','.join(category_list)}".encode()
        ).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            logger.debug("brave_search: cache read failed, proceeding without cache")
            cached = None
        if cached is not None:
            logger.debug("Brave Search cache hit for query=%r count=%d", query, count)
            return _format_search_results(json.loads(cached))

        data = _search_core(
            query,
            count=count,
            freshness=freshness,
            categories=category_list,
            context=self.context,
        )
        if "error" not in data:
            try:
                cache.set(cache_key, json.dumps(data), timeout=900)
            except Exception:
                logger.debug("brave_search: cache write failed, continuing")
        return _format_search_results(data)


__all__ = ["BraveSearchTool"]
