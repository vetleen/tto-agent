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


class ReadDocumentTool:
    """Read the full text content of one or more project documents by index number."""

    name = "read_document"
    description = (
        "Read the full text content of one or more project documents by their "
        "index number. Use this when you need the complete content of a specific "
        "document rather than search excerpts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "doc_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of document index numbers to read (e.g. [1, 3]).",
            },
        },
        "required": ["doc_indices"],
    }

    # Cap total output to ~8000 tokens worth of characters (~32k chars)
    _MAX_TOTAL_CHARS = 32_000

    def run(self, args: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        from documents.models import ProjectDocument

        doc_indices = args.get("doc_indices", [])
        if not doc_indices or not isinstance(doc_indices, list):
            raise ValueError("read_document requires a non-empty 'doc_indices' list")

        project_id = context.conversation_id
        if not project_id:
            raise ValueError("read_document requires a project context (conversation_id)")

        try:
            project_pk = int(project_id)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid project ID in context: {project_id}") from e

        documents = []
        total_chars = 0

        for idx in doc_indices:
            try:
                doc = ProjectDocument.objects.get(
                    project_id=project_pk,
                    doc_index=idx,
                )
            except ProjectDocument.DoesNotExist:
                documents.append({
                    "doc_index": idx,
                    "error": f"No document with index {idx} found.",
                })
                continue

            chunks = doc.chunks.order_by("chunk_index").values_list("text", flat=True)
            content = "\n\n".join(chunks)

            remaining = self._MAX_TOTAL_CHARS - total_chars
            if remaining <= 0:
                documents.append({
                    "doc_index": idx,
                    "filename": doc.original_filename,
                    "error": "Output size limit reached; document omitted.",
                })
                continue

            if len(content) > remaining:
                content = content[:remaining] + "\n\n[... truncated due to size limit ...]"

            total_chars += len(content)
            documents.append({
                "doc_index": idx,
                "filename": doc.original_filename,
                "content": content,
            })

        return {"documents": documents}


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SearchDocumentsTool())
_registry.register_tool(ReadDocumentTool())
