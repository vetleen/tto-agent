"""Generate a short description of a document using a cheap LLM."""

from __future__ import annotations

import logging

import tiktoken
from django.conf import settings

from documents.models import DataRoomDocument

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_INPUT_TOKENS = 10_000
_HEAD_TOKENS = 5_000
_TAIL_TOKENS = 2_000

_SYSTEM_PROMPT = (
    "Read this document and write a single paragraph (target ~100 tokens) "
    "that will help an AI agent decide whether to read the full document. "
    "This is not a summary — it is a relevance signal. Focus on what kind "
    "of document this is, what subject matter and entities it concerns, "
    "and what questions it could answer. Output ONLY the description."
)

_STRUCTURED_SYSTEM_PROMPT = (
    "Read this document and provide:\n"
    '1. "description": A single paragraph (~100 tokens) that will help an AI '
    "agent decide whether to read the full document. This is not a summary — "
    "it is a relevance signal. Focus on what kind of document this is, what "
    "subject matter and entities it concerns, and what questions it could answer.\n"
    '2. "document_type": A concise document type classification (e.g. Agreement, '
    "Patent, License, Report, Correspondence, Policy, Technical Specification, "
    "Disclosure, Application, Financial, Research Paper, Presentation, or Other)."
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


def generate_description_and_tags_from_text(
    text: str,
    user_id: int | None = None,
    data_room_id: int | None = None,
) -> dict:
    """Generate a description and document_type tag from raw text using the cheap LLM.

    Uses structured output (with_structured_output) for reliable JSON parsing.
    Returns ``{"description": "...", "tags": {"document_type": "Agreement"}}``.
    """
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext
    from llm.types.structured import DocumentDescriptionOutput

    if not text.strip():
        return {"description": "", "tags": {}}

    document_text = _prepare_document_text(text)

    context = RunContext.create(
        user_id=user_id,
        conversation_id=data_room_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=_STRUCTURED_SYSTEM_PROMPT),
            Message(role="user", content=document_text),
        ],
        model=settings.LLM_DEFAULT_CHEAP_MODEL,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    parsed, usage = service.run_structured(request, DocumentDescriptionOutput)

    description = parsed.description.strip()
    doc_type = parsed.document_type.strip()

    tags = {}
    if doc_type:
        tags["document_type"] = doc_type[:255]

    logger.info(
        "generate_description_and_tags_from_text: user_id=%s desc_len=%s tags=%s",
        user_id, len(description), list(tags.keys()),
    )
    return {"description": description, "tags": tags}


def generate_description_from_text(
    text: str,
    user_id: int | None = None,
    data_room_id: int | None = None,
) -> str:
    """Generate a one-paragraph description from raw text using the cheap LLM.

    Unlike generate_document_description(), this takes text directly instead
    of loading chunks from DB. Used early in the pipeline before chunks exist.
    """
    result = generate_description_and_tags_from_text(text, user_id, data_room_id)
    return result["description"]


def generate_document_description(document_id: int) -> str:
    """Generate a one-paragraph description of a document using the cheap LLM.

    Returns the description text. Raises on LLM failure.
    """
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    doc = DataRoomDocument.objects.get(pk=document_id)
    chunks = doc.chunks.order_by("chunk_index").values_list("text", flat=True)
    full_text = "\n\n".join(chunks)

    if not full_text.strip():
        return ""

    document_text = _prepare_document_text(full_text)

    context = RunContext.create(
        user_id=doc.uploaded_by_id,
        conversation_id=doc.data_room_id,
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
