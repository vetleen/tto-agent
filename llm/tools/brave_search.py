"""Brave Search tool — web search via Brave Search API."""

from __future__ import annotations

import json
import logging
import time

import requests
from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool

logger = logging.getLogger(__name__)


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

    _MAX_RETRIES: int = 2
    _BACKOFF_BASE: float = 0.5

    def _get_api_key(self) -> str:
        from django.conf import settings
        key = getattr(settings, "BRAVE_SEARCH_API_KEY", "")
        if not key:
            raise ValueError("BRAVE_SEARCH_API_KEY is not configured")
        return key

    def _run(self, query: str, count: int = 5) -> str:
        query = query.strip()
        if not query:
            return json.dumps({"error": "Empty search query", "results": []})

        count = max(1, min(count, 10))
        api_key = self._get_api_key()

        last_exc = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
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

                results = []
                for item in web_results:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                    })

                return json.dumps({"results": results, "count": len(results)})

            except requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning("Brave Search timeout (attempt %d)", attempt + 1)
            except requests.exceptions.HTTPError as e:
                last_exc = e
                if response.status_code < 500:
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

        return json.dumps({
            "error": f"Brave Search failed after retries: {last_exc}",
            "results": [],
        })


__all__ = ["BraveSearchTool"]
