"""Combined web search + fetch tool — search and read results in one call."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from llm.tools.interfaces import ContextAwareTool, ReasonBaseModel

logger = logging.getLogger(__name__)


class WebSearchAndReadInput(ReasonBaseModel):
    query: str = Field(description="The search query.")
    count: int = Field(default=5, description="Number of results to search for and read (1-10, default 5).")


class WebSearchAndReadTool(ContextAwareTool):
    """Search the web and read the content of each result page."""

    name: str = "web_search_and_read"
    description: str = (
        "Search the web and read the full content of each result page. "
        "Use this when you need to research a topic in depth — it searches "
        "and fetches all result pages in one step. For quick fact checks "
        "where snippets suffice, use brave_search instead."
    )
    args_schema: type[BaseModel] = WebSearchAndReadInput

    def _run(self, query: str, count: int = 5, **kwargs) -> str:
        from llm.tools.brave_search import BraveSearchTool
        from llm.tools.web_fetch import WebFetchTool

        count = max(1, min(count, 10))

        search_tool = BraveSearchTool()
        search_tool.set_context(self.context)
        search_result = json.loads(search_tool.invoke({"query": query, "count": count}))

        if "error" in search_result:
            return json.dumps(search_result)

        results = search_result.get("results", [])
        if not results:
            return json.dumps({"query": query, "results": [], "count": 0})

        fetch_tool = WebFetchTool()
        fetch_tool.set_context(self.context)

        def _fetch_one(item: dict) -> dict:
            url = item.get("url", "")
            if not url:
                return {**item, "content": "", "fetch_error": "No URL"}
            try:
                fetched = json.loads(fetch_tool.invoke({"url": url, "max_chars": 20_000}))
                if "error" in fetched:
                    return {**item, "content": "", "fetch_error": fetched["error"]}
                return {
                    **item,
                    "content": fetched.get("content", ""),
                    "truncated": fetched.get("truncated", False),
                    "char_count": fetched.get("char_count", 0),
                    "source": fetched.get("source", "direct"),
                }
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

        return json.dumps({"query": query, "results": enriched, "count": len(enriched)})


__all__ = ["WebSearchAndReadTool"]
