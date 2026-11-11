"""Brave Search tool — web search via Brave Search API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

import requests
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel
from llm.tools._throttle import (
    BACKOFF_BASE as _BACKOFF_BASE,
    MAX_RETRIES as _MAX_RETRIES,
    RATE_LIMIT_BACKOFF_SCHEDULE as _RATE_LIMIT_BACKOFF_SCHEDULE,
    TokenBucketRateLimiter as _TokenBucketRateLimiter,
    deadline_capped_wait as _deadline_capped_wait,
    parse_rpm as _parse_rpm,
    wait_budget as _wait_budget,
)

# A retry is not started with less than this left before the deadline: it could
# not finish, and would only send a request from a call the round has abandoned.
_MIN_RETRY_SECONDS = 2.0

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_RPM = _parse_rpm(os.environ.get("BRAVE_SEARCH_RPM"), 45)
_brave_rate_limiter = _TokenBucketRateLimiter(
    requests_per_second=_BRAVE_SEARCH_RPM / 60.0, burst=1
)

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


_SEARCH_CACHE_TTL = 900  # seconds a web_search result is served from the cache

_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"


def _brave_get_json(
    url: str,
    params: dict,
    *,
    context=None,
    timeout: float = 10,
    query: str = "",
) -> dict:
    """GET one Brave Search API endpoint and return its parsed JSON body.

    Shared by every Brave endpoint (one subscription → one rate limiter): the
    ``Api-Version`` pin, the response size cap, 429 backoff on
    ``X-RateLimit-Reset``/``Retry-After``, exponential backoff on timeouts/5xx
    (both capped by the run deadline) and the "don't retry" message on other
    4xx. Returns the JSON dict on success or ``{"error": "..."}`` on failure.
    """
    api_key = _get_api_key()

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
        # Each attempt fits in what is left of the deadline (the run's and, for a
        # sub-agent, the tool call's timeout): a 20 s attempt started 3 s before a
        # 45 s timeout ran on for 17 s after the round had moved on.
        attempt_timeout = timeout
        budget = _wait_budget(context)
        if budget is not None:
            if attempt and budget < _MIN_RETRY_SECONDS:
                break
            attempt_timeout = max(1.0, min(timeout, budget))
        try:
            _brave_rate_limiter.acquire()
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=attempt_timeout,
                stream=True,
            )
            response.raise_for_status()
            # Brave returns modest JSON, but cap the read anyway (on par with
            # the fetch tool) so a malformed/huge body can't OOM the worker.
            from llm.tools.web_fetch import (
                _enforce_size_and_buffer,
                _max_response_bytes,
            )

            _enforce_size_and_buffer(response, _max_response_bytes())
            return response.json()

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
                    nap, may_retry = _deadline_capped_wait(wait, context)
                    if nap > 0:
                        time.sleep(nap)
                    if may_retry:
                        continue
                    break  # run deadline reached during backoff — stop retrying
            elif response.status_code < 500:
                detail = ""
                try:
                    detail = response.json().get("error", {}).get("detail", "")
                except Exception:
                    pass
                logger.warning(
                    "Brave Search client error %d query=%r", response.status_code, query
                )
                error_msg = f"Brave Search API error {response.status_code}"
                if detail:
                    error_msg += f": {detail}"
                error_msg += (
                    ". This is a client-side error that will not resolve by retrying"
                    " with a different query. Consider reporting this issue to the"
                    " user rather than continuing to search."
                )
                return {"error": error_msg}
            logger.warning("Brave Search HTTP %d (attempt %d) query=%r", response.status_code, attempt + 1, query)
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning("Brave Search request error (attempt %d) query=%r: %s", attempt + 1, query, e)
        except Exception as e:
            # _ResponseTooLarge (size cap) and any other non-request error: don't
            # retry — surface a graceful result rather than crashing the tool call.
            from llm.tools.web_fetch import _ResponseTooLarge

            if isinstance(e, _ResponseTooLarge):
                logger.warning("Brave Search response too large query=%r: %s", query, e)
                return {"error": "Brave Search returned an unexpectedly large response."}
            raise

        if attempt < _MAX_RETRIES:
            nap, may_retry = _deadline_capped_wait(_BACKOFF_BASE * (2 ** attempt), context)
            if nap > 0:
                time.sleep(nap)
            if not may_retry:
                break  # run deadline reached during backoff — stop retrying

    if isinstance(last_exc, requests.exceptions.HTTPError) and getattr(
        getattr(last_exc, "response", None), "status_code", None
    ) == 429:
        return {
            "error": "Brave Search rate limited after retries."
            " Web search is temporarily unavailable."
            " Consider reporting this to the user rather than continuing to search.",
        }
    logger.error("Brave Search failed after retries query=%r", query, exc_info=last_exc)
    return {
        "error": "Brave Search failed after retries."
        " Web search is currently unavailable."
        " Consider reporting this to the user rather than continuing to search.",
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

    data = _brave_get_json(_WEB_SEARCH_URL, params, context=context, query=query)
    if "error" in data:
        return {"error": data["error"], "results": []}

    results = []
    for category in categories:
        section = data.get(category) or {}
        for item in section.get("results", []) or []:
            results.append(_parse_result_item(item, category))

    _scan_search_results(results, context)

    return {"query": query, "results": results, "count": len(results)}


def _scan_search_results(results: list[dict], context) -> None:
    """Scan web_search results for prompt injection (log only, never blocks)."""
    from guardrails.web_content import scan_web_content_from_tool

    combined = "\n".join(
        " ".join(
            [r.get("title", ""), r.get("description", "")]
            + (r.get("extra_snippets") or [])
        )
        for r in results
    )
    scan_web_content_from_tool(combined, context, source_label="brave_search")


def _scan_llm_context_results(results: list[dict], context) -> None:
    """Scan LLM Context extracts for prompt injection (log only, never blocks)."""
    from guardrails.web_content import scan_web_content_from_tool

    combined = "\n".join(
        " ".join([r.get("title", "")] + (r.get("snippets") or [])) for r in results
    )
    scan_web_content_from_tool(combined, context, source_label="brave_llm_context")


# Brave LLM Context timeout: one search + server-side extraction (their docs
# suggest a 30 s client timeout; typical responses are ~1 s).
_LLM_CONTEXT_TIMEOUT_SECONDS = 20


def _context_age(source: dict | None) -> str:
    """Readable age from a ``sources[url].age`` array.

    Brave returns ``[full date, YYYY-MM-DD, relative age, ISO timestamp]`` (empty
    when the page has no known date). Prefer the relative age, like ``web_search``.
    """
    age = (source or {}).get("age") or []
    if isinstance(age, str):
        return age
    if not isinstance(age, list):
        return ""
    for idx in (2, 1, 0):
        if len(age) > idx and isinstance(age[idx], str) and age[idx].strip():
            return age[idx].strip()
    return ""


def _llm_context_core(
    query: str,
    *,
    max_urls: int,
    max_tokens: int,
    freshness: str = "",
    context=None,
) -> dict:
    """Run a Brave LLM Context search and return a result dict (no formatting).

    One call returns query-relevant extracts ("snippets") from the top pages,
    sized to ``max_tokens`` (Brave's approximate tokens). Returns
    ``{"query", "results": [...], "count"}`` on success or
    ``{"error": "...", "results": []}``. Each result item:
    ``{title, url, age, snippets: [str, ...]}``.
    """
    from llm.tools._text_cleaning import normalize_text

    params: dict = {
        "q": query,
        "maximum_number_of_urls": max_urls,
        "maximum_number_of_tokens": max_tokens,
    }
    if freshness:
        params["freshness"] = freshness

    data = _brave_get_json(
        _LLM_CONTEXT_URL,
        params,
        context=context,
        timeout=_LLM_CONTEXT_TIMEOUT_SECONDS,
        query=query,
    )
    if "error" in data:
        return {"error": data["error"], "results": []}

    sources = data.get("sources") or {}
    results = []
    for item in (data.get("grounding") or {}).get("generic") or []:
        url = item.get("url", "")
        snippets = []
        for s in item.get("snippets") or []:
            if not isinstance(s, str):
                # Tables/schemas can arrive structured; keep them as JSON text.
                s = json.dumps(s, ensure_ascii=False)
            s = normalize_text(s).strip()
            if s:
                snippets.append(s)
        if not snippets:
            continue
        source = sources.get(url) if isinstance(sources, dict) else None
        results.append({
            "title": _clean_field(item.get("title") or (source or {}).get("title") or ""),
            "url": url,
            "age": _context_age(source),
            "snippets": snippets,
        })

    _scan_llm_context_results(results, context)

    return {"query": query, "results": results, "count": len(results)}


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
    # Sub-agent per-call timeout: 1.5x the 14-day p99 (19 s).
    timeout_seconds: float | None = 30
    section: str = "skills"
    subagent_section: str = "chat"
    start_label: str = "Searching the web..."
    end_label: str = "Searched the web"
    description: str = (
        "Search the web for current information using Brave Search. "
        "Use this when you need up-to-date information, facts, or data "
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
        from guardrails.web_content import claim_thread_scan

        if cached is not None:
            logger.debug("Brave Search cache hit for query=%r count=%d", query, count)
            data = json.loads(cached)
            # The cache is shared across threads: a thread served from it still
            # gets its own scan (once per entry).
            if claim_thread_scan(cache, cache_key, self.context, _SEARCH_CACHE_TTL):
                _scan_search_results(data.get("results") or [], self.context)
            return _format_search_results(data)

        data = _search_core(
            query,
            count=count,
            freshness=freshness,
            categories=category_list,
            context=self.context,
        )
        if "error" not in data:
            claim_thread_scan(cache, cache_key, self.context, _SEARCH_CACHE_TTL)  # scanned above
            try:
                cache.set(cache_key, json.dumps(data), timeout=_SEARCH_CACHE_TTL)
            except Exception:
                logger.debug("brave_search: cache write failed, continuing")
        return _format_search_results(data)


__all__ = ["BraveSearchTool"]
