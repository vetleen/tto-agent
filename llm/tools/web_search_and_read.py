"""Combined web search + fetch tool — search and read results in one call."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)

# Per-page content budget for the combined tool. Deliberately generous: deep
# research runs are typically delegated to cheap subagents, so occasionally
# pulling a full 20k-char page is an accepted trade-off (see web_fetch
# pagination for reading beyond it).
_PER_PAGE_CHARS = 20_000


class WebSearchAndReadInput(ReasonBaseModel):
    query: str = Field(description="The search query.")
    count: int = Field(default=5, description="Number of results to search for and read (1-20, default 5).")
    freshness: str = Field(
        default="",
        description=(
            "Filter results by page age: pd (past day), pw (past week), "
            "pm (past month), py (past year), or a date range "
            "YYYY-MM-DDtoYYYY-MM-DD. Empty means no filter. "
            "Invalid values are ignored."
        ),
    )


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


class WebSearchAndReadTool(ContextAwareTool):
    """Search the web and read the content of each result page."""

    name: str = "web_search_and_read"
    description: str = (
        "Search the web and read the full content of each result page. "
        "Use this when you need to research a topic in depth — it searches "
        "and fetches all result pages in one step. For quick fact checks "
        "where snippets suffice, use brave_search instead. "
        "Use freshness to restrict results to a recent period."
    )
    args_schema: type[BaseModel] = WebSearchAndReadInput

    def _run(self, query: str, count: int = 5, freshness: str = "", **kwargs) -> str:
        from django.core.cache import cache

        from llm.tools.brave_search import _search_core, _validate_freshness
        from llm.tools.web_fetch import _fetch_core

        query = query.strip()
        if not query:
            return "Search error: empty search query."

        count = max(1, min(count, 20))
        freshness = _validate_freshness(freshness)

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

        if len(results) == 1:
            enriched = [_fetch_one(results[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(len(results), 4)) as pool:
                enriched = list(pool.map(_fetch_one, results))

        logger.info(
            "web_search_and_read: query=%r fetched=%d/%d",
            query, sum(1 for r in enriched if r.get("content")), len(enriched),
        )

        return _format_search_and_read(query, enriched)


__all__ = ["WebSearchAndReadTool"]
