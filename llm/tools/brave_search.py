"""Brave Search tool — web search via Brave Search API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time

import requests
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool

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


class BraveSearchInput(BaseModel):
    query: str = Field(description="The search query.")
    count: int = Field(default=5, description="Number of results to return (1-10, default 5).")


class BraveSearchTool(ContextAwareTool):
    """Search the web using the Brave Search API."""

    name: str = "brave_search"
    description: str = (
        "Search the web for current information using Brave Search. "
        "Use this when you need up-to-date information, facts, or data "
        "that may not be in your training data."
    )
    args_schema: type[BaseModel] = BraveSearchInput

    _MAX_RETRIES: int = 3
    _BACKOFF_BASE: float = 0.5
    _RATE_LIMIT_BACKOFF_SCHEDULE: list[float] = [5.0, 15.0, 30.0, 60.0]

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float | None:
        """Parse Retry-After header as float seconds. Returns None if absent/invalid."""
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _get_api_key(self) -> str:
        from django.conf import settings
        key = getattr(settings, "BRAVE_SEARCH_API_KEY", "")
        if not key:
            raise ValueError("BRAVE_SEARCH_API_KEY is not configured")
        return key

    def _run(self, query: str, count: int = 5) -> str:
        from django.core.cache import cache

        query = query.strip()
        if not query:
            return json.dumps({"error": "Empty search query", "results": []})

        count = max(1, min(count, 10))

        cache_key = "brave_search:" + hashlib.sha256(f"{query}:{count}".encode()).hexdigest()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Brave Search cache hit for query=%r count=%d", query, count)
            return cached

        api_key = self._get_api_key()

        last_exc = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                _brave_rate_limiter.acquire()
                response = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                    params={"q": query, "count": count},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                web_results = data.get("web", {}).get("results", [])

                from llm.tools._text_cleaning import normalize_text

                results = []
                for item in web_results:
                    results.append({
                        "title": normalize_text(item.get("title", "")),
                        "url": item.get("url", ""),
                        "description": normalize_text(item.get("description", "")),
                    })

                # Scan for prompt injection (log only, never blocks)
                try:
                    combined = "\n".join(
                        f"{r.get('title', '')} {r.get('description', '')}"
                        for r in results
                    )
                    if combined.strip():
                        from guardrails.web_content import scan_web_content

                        scan_web_content(
                            combined,
                            user_id=self.context.user_id if self.context else None,
                            thread_id=self.context.conversation_id if self.context else None,
                            org_id=None,
                            source_label="brave_search",
                        )
                except Exception:
                    logger.debug("brave_search: web content scan failed (non-fatal)")

                result = json.dumps({"results": results, "count": len(results)})
                cache.set(cache_key, result, timeout=900)
                return result

            except requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning("Brave Search timeout (attempt %d)", attempt + 1)
            except requests.exceptions.HTTPError as e:
                last_exc = e
                if response.status_code == 429:
                    retry_after = self._parse_retry_after(response)
                    schedule_wait = self._RATE_LIMIT_BACKOFF_SCHEDULE[
                        min(attempt, len(self._RATE_LIMIT_BACKOFF_SCHEDULE) - 1)
                    ]
                    wait = min(
                        retry_after if retry_after is not None else schedule_wait,
                        self._RATE_LIMIT_BACKOFF_SCHEDULE[-1],
                    )
                    logger.warning("Brave Search rate limited (attempt %d), waiting %.1fs", attempt + 1, wait)
                    if attempt < self._MAX_RETRIES:
                        time.sleep(wait)
                        continue
                elif response.status_code < 500:
                    # Client error — don't retry
                    return json.dumps({
                        "error": f"Brave Search API error: {response.status_code}",
                        "results": [],
                    })
                logger.warning("Brave Search HTTP %d (attempt %d)", response.status_code, attempt + 1)
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning("Brave Search request error (attempt %d): %s", attempt + 1, e)

            if attempt < self._MAX_RETRIES:
                time.sleep(self._BACKOFF_BASE * (2 ** attempt))

        if isinstance(last_exc, requests.exceptions.HTTPError) and getattr(
            getattr(last_exc, "response", None), "status_code", None
        ) == 429:
            return json.dumps({
                "error": "Brave Search rate limited — try again shortly",
                "results": [],
            })
        return json.dumps({
            "error": f"Brave Search failed after retries: {last_exc}",
            "results": [],
        })


__all__ = ["BraveSearchTool"]
