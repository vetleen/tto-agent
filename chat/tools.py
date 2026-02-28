"""RAG tool: search project documents via similarity search."""

from __future__ import annotations

import logging
from typing import Any, Dict

from llm.tools import get_tool_registry
from llm.types.context import RunContext

logger = logging.getLogger(__name__)


class SearchDocumentsTool:
    """Search project documents using semantic similarity."""

    name = "search_documents"
    description = (
        "Search the project's uploaded documents for information relevant to a query. "
        "Use this when the user asks about document contents, wants summaries, "
        "or needs specific information from their files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant document passages.",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5).",
            },
        },
        "required": ["query"],
    }

    def run(self, args: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        from documents.services.retrieval import similarity_search_chunks

        query = args.get("query", "").strip()
        if not query:
            raise ValueError("search_documents requires a non-empty 'query'")

        k = args.get("k", 5)
        if not isinstance(k, int) or k < 1:
            k = 5
        k = min(k, 10)

        project_id = context.conversation_id
        if not project_id:
            raise ValueError("search_documents requires a project context (conversation_id)")

        try:
            project_pk = int(project_id)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid project ID in context: {project_id}") from e

        try:
            docs = similarity_search_chunks(project_id=project_pk, query=query, k=k)
        except Exception as exc:
            logger.exception("search_documents: similarity_search_chunks failed")
            return {
                "results": [],
                "count": 0,
                "error": "Search failed",
                "error_type": type(exc).__name__,
            }

        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
            })

        return {"results": results, "count": len(results)}


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SearchDocumentsTool())
