"""
Backend retrieval: get chunks by project/document (ordered) or by similarity search.
"""
from __future__ import annotations

from typing import Any

from documents.models import ProjectDocumentChunk
from documents.services import vector_store as vs


def get_chunks_by_document(document_id: int) -> list[dict[str, Any]]:
    """Return chunks for a document in order (from DB)."""
    chunks = ProjectDocumentChunk.objects.filter(document_id=document_id).order_by("chunk_index")
    return [
        {
            "id": c.id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
            "source_page_start": c.source_page_start,
            "source_page_end": c.source_page_end,
        }
        for c in chunks
    ]


def get_chunks_by_project(project_id: int) -> list[dict[str, Any]]:
    """Return all chunks for a project, grouped by document (order preserved)."""
    chunks = (
        ProjectDocumentChunk.objects.filter(document__project_id=project_id)
        .select_related("document")
        .order_by("document_id", "chunk_index")
    )
    return [
        {
            "id": c.id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "heading": c.heading,
            "text": c.text,
            "token_count": c.token_count,
        }
        for c in chunks
    ]


def similarity_search_chunks(
    project_id: int,
    query: str,
    k: int = 10,
    document_id: int | None = None,
) -> list[Any]:
    """
    Run similarity search in the vector store; returns LangChain Document objects
    with metadata chunk_id, document_id, project_id. Backend-ready for RAG.
    """
    return vs.similarity_search(project_id=project_id, query=query, k=k, document_id=document_id)
