"""Generate a short description of a document using a cheap LLM."""

from __future__ import annotations

import logging

import tiktoken
from django.conf import settings

from documents.models import ProjectDocument

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_INPUT_TOKENS = 10_000
_HEAD_TOKENS = 5_000
_TAIL_TOKENS = 2_000

_SYSTEM_PROMPT = (
    "Read this document and write a single paragraph (target ~200 tokens) "
    "describing what the document contains. Focus on the type of document, "
    "its subject matter, and the key topics it covers. Output ONLY the description."
)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most *max_tokens* tokens."""
    token_ids = _ENCODING.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return _ENCODING.decode(token_ids[:max_tokens])


def _prepare_document_text(chunks_text: str) -> str:
    """Prepare document text for the LLM, truncating if needed."""
    token_ids = _ENCODING.encode(chunks_text)
    total_tokens = len(token_ids)

    if total_tokens <= _MAX_INPUT_TOKENS:
        return chunks_text

    head = _ENCODING.decode(token_ids[:_HEAD_TOKENS])
    tail = _ENCODING.decode(token_ids[-_TAIL_TOKENS:])
    omitted = total_tokens - _HEAD_TOKENS - _TAIL_TOKENS
    return f"{head}\n\n[... middle {omitted} tokens omitted ...]\n\n{tail}"


def generate_document_description(document_id: int) -> str:
    """Generate a one-paragraph description of a document using the cheap LLM.

    Returns the description text. Raises on LLM failure.
    """
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    doc = ProjectDocument.objects.get(pk=document_id)
    chunks = doc.chunks.order_by("chunk_index").values_list("text", flat=True)
    full_text = "\n\n".join(chunks)

    if not full_text.strip():
        return ""

    document_text = _prepare_document_text(full_text)

    context = RunContext.create(
        user_id=doc.uploaded_by_id,
        conversation_id=doc.project_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=document_text),
        ],
        model=settings.LLM_DEFAULT_CHEAP_MODEL,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    response = service.run("simple_chat", request)
    description = response.message.content.strip()

    doc.description = description
    doc.save(update_fields=["description", "updated_at"])

    logger.info(
        "generate_document_description: document_id=%s len=%s",
        document_id,
        len(description),
    )
    return description
