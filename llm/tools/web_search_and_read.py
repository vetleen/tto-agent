"""Combined web search + read tool — search and read results in one call.

Two backends, chosen by ``settings.WEB_SEARCH_READ_BACKEND``:

* ``"llm_context"`` (default) — Brave's LLM Context endpoint: one call returns
  query-relevant extracts from the top pages, sized by ``depth``
  (``DEPTH_TOKEN_BUDGETS``). No local page fetching.
* ``"fetch"`` — the original path, kept as a rollback switch: Brave web search,
  then fetch every result page ourselves (``web_fetch._fetch_core``) and keep
  the first ``_PER_PAGE_CHARS`` of each.
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Context budgets — edit freely. Brave's approximate tokens for the whole
# result (``maximum_number_of_tokens``, Brave range 1024–32768).
# ---------------------------------------------------------------------------
DEPTH_TOKEN_BUDGETS: dict[str, int] = {
    "quick": 2048,      # a fact check
    "standard": 8192,   # normal research question
    "deep": 16384,      # complex research
}
DEFAULT_DEPTH = "standard"
# Default maximum number of sources per backend when the model omits ``count``.
# For llm_context the token budget is the real limiter (Brave's relevance
# threshold drops weak sources); for fetch every source is a page download.
DEFAULT_COUNT_LLM_CONTEXT = 10
DEFAULT_COUNT_FETCH = 5
MAX_COUNT = 20
# Cache TTL for llm_context results (same as web_search).
_LLM_CONTEXT_CACHE_TTL = 900

BACKEND_LLM_CONTEXT = "llm_context"
BACKEND_FETCH = "fetch"

# Per-page content budget for the fetch backend. 10k chars covers the vast
# majority of pages; for the rare longer page the truncation note points the
# model at web_fetch's start_index pagination to keep reading when relevant.
_PER_PAGE_CHARS = 10_000

_FULL_PAGE_NOTE = (
    "These are extracts relevant to the query, not full pages. "
    "Use web_fetch(url) to read a page in full."
)


def _backend() -> str:
    from django.conf import settings

    value = str(getattr(settings, "WEB_SEARCH_READ_BACKEND", BACKEND_LLM_CONTEXT) or "").strip().lower()
    return BACKEND_FETCH if value == BACKEND_FETCH else BACKEND_LLM_CONTEXT


def _resolve_depth(depth: str | None) -> str:
    value = (depth or "").strip().lower()
    return value if value in DEPTH_TOKEN_BUDGETS else DEFAULT_DEPTH


def _resolve_count(count: int | None, default: int) -> int:
    if count is None:
        return default
    try:
        count = int(count)
    except (TypeError, ValueError):
        return default
    return max(1, min(count, MAX_COUNT))


class WebSearchAndReadInput(ReasonBaseModel):
    query: str = Field(description="The search query.")
    count: int | None = Field(
        default=None,
        description=f"Maximum number of sources to read (1-{MAX_COUNT}). Leave empty for the default.",
    )
    freshness: str = Field(
        default="",
        description=(
            "Filter results by page age: pd (past day), pw (past week), "
            "pm (past month), py (past year), or a date range "
            "YYYY-MM-DDtoYYYY-MM-DD. Empty means no filter. "
            "Invalid values are ignored."
        ),
    )
    depth: Literal["quick", "standard", "deep"] = Field(
        default=DEFAULT_DEPTH,
        description=(
            "How much page content to return in total: quick (a fact check, "
            "~2k tokens), standard (default, ~8k), deep (complex research, ~16k)."
        ),
    )

    @field_validator("depth", mode="before")
    @classmethod
    def _coerce_depth(cls, value):
        # An unknown or empty depth falls back to the default rather than
        # failing the call (the schema still advertises the three values).
        return _resolve_depth(value if isinstance(value, str) else None)


def _format_llm_context(query: str, results: list[dict]) -> str:
    """Format _llm_context_core results as markdown for the LLM."""
    from llm.tools._text_cleaning import (
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_END,
        EXTERNAL_CONTENT_NOTE,
    )

    lines = [
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_NOTE,
        "",
        f"Search: {query}",
        f"Results: {len(results)}",
        "",
    ]
    for i, r in enumerate(results, 1):
        age_str = f" ({r['age']})" if r.get("age") else ""
        lines.append(f"--- Source {i}: {r.get('title') or '(no title)'}{age_str} ---")
        lines.append(f"URL: {r.get('url', '')}")
        lines.append("")
        lines.append("\n\n".join(r.get("snippets") or []))
        lines.append("")
    lines.append(_FULL_PAGE_NOTE)
    lines.append(EXTERNAL_CONTENT_END)
    return "\n".join(lines)


def _format_search_and_read(query: str, enriched: list[dict]) -> str:
    """Format enriched search results (with page content) as markdown."""
    from llm.tools._text_cleaning import (
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_END,
        EXTERNAL_CONTENT_NOTE,
    )

    lines = [
        EXTERNAL_CONTENT_BEGIN,
        EXTERNAL_CONTENT_NOTE,
        "",
        f"Search: {query}",
        f"Results: {len(enriched)}",
        "",
    ]
    for i, r in enumerate(enriched, 1):
        age_str = f" ({r['age']})" if r.get("age") else ""
        lines.append(f"--- Source {i}: {r.get('title') or '(no title)'}{age_str} ---")
        lines.append(f"URL: {r.get('url', '')}")
        if r.get("description"):
            lines.append(r["description"])
        content = r.get("content", "")
        if r.get("fetch_error"):
            lines.append(f"(Could not fetch content: {r['fetch_error']})")
        elif content:
            total_len = len(content)
            window = content[:_PER_PAGE_CHARS]
            if len(window) < total_len:
                # Truncate at a whitespace boundary so we don't cut mid-word.
                boundary = max(window.rfind(" "), window.rfind("\n"))
                if boundary > _PER_PAGE_CHARS // 2:
                    window = window[:boundary]
            lines.append("")
            lines.append(window)
            if len(window) < total_len:
                lines.append(
                    f"\n(Content truncated at {len(window)} of {total_len} chars. "
                    f"Use web_fetch with start_index={len(window)} to read more.)"
                )
        lines.append("")
    lines.append(EXTERNAL_CONTENT_END)
    return "\n".join(lines)


_LLM_CONTEXT_DESCRIPTION = (
    "Search the web and read the parts of the top result pages that are relevant "
    "to your query, in one step. Use this when you need to research a topic — it "
    "returns query-relevant extracts from each source, not full pages; call "
    "web_fetch(url) to read a page in full. Use depth to size the result "
    "(quick for a fact check, standard by default, deep for complex research) "
    "and freshness to restrict results to a recent period. For a plain list of "
    "links and short snippets, use web_search instead."
)

_FETCH_DESCRIPTION = (
    "Search the web and read the full content of each result page. "
    "Use this when you need to research a topic in depth — it searches "
    "and fetches all result pages in one step. For quick fact checks "
    "where snippets suffice, use web_search instead. "
    "Use freshness to restrict results to a recent period."
)


class WebSearchAndReadTool(ContextAwareTool):
    """Search the web and read the relevant content of each result page."""

    name: str = "web_search_read"
    # Sub-agent per-call timeout. llm_context is one Brave call (~1 s typical,
    # 20 s request timeout + retries); the fetch backend's 14-day p99 was 52 s,
    # so it keeps its old 80 s (see model_post_init). Re-tune from the p99.
    timeout_seconds: float | None = 45
    section: str = "skills"
    subagent_section: str = "chat"
    start_label: str = "Searching and reading web pages..."
    end_label: str = "Searched and read web pages"
    description: str = _LLM_CONTEXT_DESCRIPTION
    args_schema: type[BaseModel] = WebSearchAndReadInput

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        # The backend is a deploy-time setting: the model-facing description and
        # timeout follow it when the tool is instantiated (registry import).
        if _backend() == BACKEND_FETCH:
            self.description = _FETCH_DESCRIPTION
            self.timeout_seconds = 80

    def _run(
        self,
        query: str,
        count: int | None = None,
        freshness: str = "",
        depth: str = DEFAULT_DEPTH,
        **kwargs,
    ) -> str:
        from llm.tools.brave_search import _validate_freshness

        query = query.strip()
        if not query:
            return "Search error: empty search query."
        freshness = _validate_freshness(freshness)

        if _backend() == BACKEND_FETCH:
            return self._run_fetch_backend(
                query, _resolve_count(count, DEFAULT_COUNT_FETCH), freshness
            )
        return self._run_llm_context_backend(
            query,
            _resolve_count(count, DEFAULT_COUNT_LLM_CONTEXT),
            freshness,
            _resolve_depth(depth),
        )

    def _run_llm_context_backend(
        self, query: str, count: int, freshness: str, depth: str
    ) -> str:
        from django.core.cache import cache

        from guardrails.web_content import claim_thread_scan
        from llm.tools.brave_search import _llm_context_core, _scan_llm_context_results

        max_tokens = DEPTH_TOKEN_BUDGETS[depth]
        cache_key = "web_search_read_ctx_v1:" + hashlib.sha256(
            f"{query}:{count}:{freshness}:{max_tokens}".encode()
        ).hexdigest()
        try:
            cached = cache.get(cache_key)
        except Exception:
            logger.debug("web_search_read: cache read failed, proceeding without cache")
            cached = None
        if cached is not None:
            data = json.loads(cached)
            # The cache is shared across threads: a thread served from it still
            # gets its own scan (once per entry).
            if claim_thread_scan(cache, cache_key, self.context, _LLM_CONTEXT_CACHE_TTL):
                _scan_llm_context_results(data.get("results") or [], self.context)
        else:
            data = _llm_context_core(
                query,
                max_urls=count,
                max_tokens=max_tokens,
                freshness=freshness,
                context=self.context,
            )
            if "error" in data:
                return f"Search error: {data['error']}"
            claim_thread_scan(cache, cache_key, self.context, _LLM_CONTEXT_CACHE_TTL)  # scanned in the core
            try:
                cache.set(cache_key, json.dumps(data), timeout=_LLM_CONTEXT_CACHE_TTL)
            except Exception:
                logger.debug("web_search_read: cache write failed, continuing")

        results = data.get("results") or []
        if not results:
            return "No relevant content found."
        output = _format_llm_context(query, results)
        logger.info(
            "web_search_and_read: backend=llm_context query=%r depth=%s sources=%d chars=%d cached=%s",
            query, depth, len(results), len(output), cached is not None,
        )
        return output

    def _run_fetch_backend(self, query: str, count: int, freshness: str) -> str:
        from django.core.cache import cache

        from llm.tools.brave_search import _search_core
        from llm.tools.web_fetch import _fetch_core

        search_data = _search_core(
            query,
            count=count,
            freshness=freshness,
            categories=["web"],
            context=self.context,
        )
        if "error" in search_data:
            return f"Search error: {search_data['error']}"

        results = search_data.get("results", [])
        if not results:
            return "No results found."

        def _fetch_one(item: dict) -> dict:
            # Runs on ThreadPoolExecutor worker threads for multi-result fetches.
            # _fetch_core does ORM writes (the guardrail web-content scan), whose
            # thread-local DB connections must be released or they leak against
            # the connection cap — mirrors _execute_tool_calls in simple_chat.
            from django.db import close_old_connections

            try:
                url = item.get("url", "")
                if not url:
                    return {**item, "content": "", "fetch_error": "No URL"}
                try:
                    fetched = _fetch_core(url, cache, context=self.context)
                    if "error" in fetched:
                        return {**item, "content": "", "fetch_error": fetched["error"]}
                    return {**item, "content": fetched.get("content", "")}
                except Exception as e:
                    return {**item, "content": "", "fetch_error": str(e)}
            finally:
                close_old_connections()

        if len(results) == 1:
            enriched = [_fetch_one(results[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(len(results), 4)) as pool:
                enriched = list(pool.map(_fetch_one, results))

        logger.info(
            "web_search_and_read: backend=fetch query=%r fetched=%d/%d",
            query, sum(1 for r in enriched if r.get("content")), len(enriched),
        )

        return _format_search_and_read(query, enriched)


__all__ = ["WebSearchAndReadTool"]
