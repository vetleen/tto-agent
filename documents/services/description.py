"""Generate a short description of a document using a cheap LLM."""

from __future__ import annotations

import logging

import tiktoken
from django.conf import settings


logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
_MAX_INPUT_TOKENS = 10_000
_HEAD_TOKENS = 5_000
_TAIL_TOKENS = 2_000

_STRUCTURED_SYSTEM_PROMPT = (
    "Read this document and provide:\n"
    '1. "description": A single paragraph (~100 tokens) that will help an AI '
    "agent decide whether to read the full document. This is not a summary — "
    "it is a relevance signal. Focus on what kind of document this is, what "
    "subject matter and entities it concerns, and what questions it could answer.\n"
    '2. "document_type": A concise document type classification (e.g. Agreement, '
    "Patent, License, Report, Correspondence, Policy, Technical Specification, "
    "Disclosure, Application, Financial, Research Paper, Presentation, or Other).\n"
    '3. "document_date": If the document has a clearly identifiable primary date '
    "(e.g. signing date, publication date, effective date, date of correspondence), "
    "return it as YYYY-MM-DD. Return null if no clear date is present."
)

# Appended when the document is an image: the text we pass is the vision model's
# detailed description, so the cheap model must summarise the IMAGE — not write a
# description *of the description*.
_IMAGE_INPUT_NOTE = (
    "\n\nIMPORTANT: This document is an image. The text below is a detailed, "
    "AI-generated description of that image — it is NOT text contained in a file. "
    'Write "description" as a concise account of what the IMAGE itself shows. Do '
    'NOT refer to it as "a description", "a caption", or "this document describes"; '
    "describe the image directly. Preserve the core facts, keep within the ~100-token "
    'limit, and do not add anything not supported by the text. Set "document_type" to "Image".'
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
    org_id: int | None = None,
    is_image: bool = False,
) -> dict:
    """Generate a description and document_type tag from raw text using the cheap LLM.

    Uses structured output (with_structured_output) for reliable JSON parsing.
    Returns ``{"description": "...", "tags": {"document_type": "Agreement"}}``.

    When *is_image* is set, *text* is the vision model's description of an image
    (not file text), so the prompt is adjusted to summarise the image directly.
    """
    from core.preferences import resolve_org_feature_model
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext
    from llm.types.structured import DocumentDescriptionOutput

    if not text.strip():
        return {"description": "", "tags": {}}

    document_text = _prepare_document_text(text)
    model = resolve_org_feature_model(org_id, "document_description")

    system_prompt = _STRUCTURED_SYSTEM_PROMPT + (_IMAGE_INPUT_NOTE if is_image else "")

    context = RunContext.create(
        user_id=user_id,
    )
    request = ChatRequest(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=document_text),
        ],
        model=model,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    parsed, usage = service.run_structured(request, DocumentDescriptionOutput)

    description = parsed.description.strip()
    doc_type = parsed.document_type.strip()

    document_date = None
    if parsed.document_date:
        try:
            from datetime import date

            document_date = date.fromisoformat(parsed.document_date.strip())
        except (ValueError, AttributeError):
            pass

    tags = {}
    if doc_type:
        tags["document_type"] = doc_type[:255]

    logger.info(
        "generate_description_and_tags_from_text: user_id=%s desc_len=%s tags=%s document_date=%s",
        user_id, len(description), list(tags.keys()), document_date,
    )
    return {"description": description, "tags": tags, "document_date": document_date}


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


