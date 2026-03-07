"""RAG tools: search data room documents via similarity search."""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, get_tool_registry

logger = logging.getLogger(__name__)


# --- Input schemas ---

class SearchDocumentsInput(BaseModel):
    query: str = Field(description="The search query to find relevant document passages.")
    k: int = Field(default=5, description="Number of results to return (1-10, default 5).")


class ReadDocumentInput(BaseModel):
    doc_indices: list[int] = Field(description="List of document index numbers to read (e.g. [1, 3]).")
    data_room_id: Optional[int] = Field(
        default=None,
        description="Optional data room ID to disambiguate when multiple data rooms are attached.",
    )


# --- Tools ---

class SearchDocumentsTool(ContextAwareTool):
    """Search data room documents using semantic similarity."""

    name: str = "search_documents"
    description: str = (
        "Search the attached data rooms' documents for information relevant to a query. "
        "Use this when the user asks about document contents, wants summaries, "
        "or needs specific information from their files."
    )
    args_schema: type[BaseModel] = SearchDocumentsInput

    def _run(self, query: str, k: int = 5) -> str:
        from documents.models import DataRoom
        from documents.services.retrieval import similarity_search_chunks

        query = query.strip()
        if not query:
            raise ValueError("search_documents requires a non-empty 'query'")

        if not isinstance(k, int) or k < 1:
            k = 5
        k = min(k, 10)

        context = self.context
        data_room_ids = context.data_room_ids if context else []
        if not data_room_ids:
            return json.dumps({"error": "No data rooms attached", "results": [], "count": 0})

        # Verify the user has access to all data rooms
        user_id = context.user_id if context else None
        if user_id:
            accessible = set(
                DataRoom.objects.filter(
                    pk__in=data_room_ids, created_by_id=user_id,
                ).values_list("pk", flat=True)
            )
            data_room_ids = [rid for rid in data_room_ids if rid in accessible]
            if not data_room_ids:
                raise ValueError("Data rooms not found or access denied")

        try:
            docs = similarity_search_chunks(data_room_ids=data_room_ids, query=query, k=k)
        except Exception as exc:
            logger.exception("search_documents: similarity_search_chunks failed")
            return json.dumps({
                "results": [],
                "count": 0,
                "error": "Search failed",
                "error_type": type(exc).__name__,
            })

        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
            })

        return json.dumps({"results": results, "count": len(results)})


class ReadDocumentTool(ContextAwareTool):
    """Read the full text content of one or more documents by index number."""

    name: str = "read_document"
    description: str = (
        "Read the full text content of one or more documents by their "
        "index number. Use this when you need the complete content of a specific "
        "document rather than search excerpts."
    )
    args_schema: type[BaseModel] = ReadDocumentInput

    # Cap total output to ~8000 tokens worth of characters (~32k chars)
    _MAX_TOTAL_CHARS: int = 32_000

    def _run(self, doc_indices: list[int], data_room_id: int | None = None) -> str:
        from documents.models import DataRoom, DataRoomDocument

        if not doc_indices or not isinstance(doc_indices, list):
            raise ValueError("read_document requires a non-empty 'doc_indices' list")

        context = self.context
        data_room_ids = context.data_room_ids if context else []
        if not data_room_ids:
            return json.dumps({"error": "No data rooms attached", "documents": []})

        # Optionally scope to a single data room
        if data_room_id and data_room_id in data_room_ids:
            data_room_ids = [data_room_id]

        # Verify the user has access
        user_id = context.user_id if context else None
        if user_id:
            accessible = set(
                DataRoom.objects.filter(
                    pk__in=data_room_ids, created_by_id=user_id,
                ).values_list("pk", flat=True)
            )
            data_room_ids = [rid for rid in data_room_ids if rid in accessible]
            if not data_room_ids:
                raise ValueError("Data rooms not found or access denied")

        documents = []
        total_chars = 0

        for idx in doc_indices:
            try:
                doc = DataRoomDocument.objects.get(
                    data_room_id__in=data_room_ids,
                    doc_index=idx,
                    is_archived=False,
                )
            except DataRoomDocument.DoesNotExist:
                documents.append({
                    "doc_index": idx,
                    "error": f"No document with index {idx} found.",
                })
                continue
            except DataRoomDocument.MultipleObjectsReturned:
                # Same doc_index in multiple rooms — take the first match
                doc = DataRoomDocument.objects.filter(
                    data_room_id__in=data_room_ids,
                    doc_index=idx,
                    is_archived=False,
                ).first()

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
                "data_room_id": doc.data_room_id,
                "content": content,
            })

        return json.dumps({"documents": documents})


# Register on import
_registry = get_tool_registry()
_registry.register_tool(SearchDocumentsTool())
_registry.register_tool(ReadDocumentTool())
