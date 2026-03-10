"""LLM-powered document normalization: convert extracted text to clean markdown."""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings

from core.tokens import count_tokens

logger = logging.getLogger(__name__)

MAX_BATCH_TOKENS = 1000
OVERLAP_TOKENS = 200
MAX_RETRIES = 2
MAX_WORKERS = 10

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_SYSTEM_PROMPT_TEMPLATE = """\
You are a document formatting assistant. Return the COMPLETE text below with \
structural markdown formatting applied. Output ONLY the reformatted text, \
nothing else.

You MUST:
- Add markdown headings (# ## ###) where section breaks occur
- Convert tables to proper markdown table syntax
- Convert lists to markdown list syntax (- or 1. 2. 3.)
- Format code snippets as code blocks
- Remove page numbers, repeated headers/footers, and other metadata artifacts
- Clean up OCR artifacts (broken words, extraneous whitespace)

You MUST NOT:
- Rephrase, paraphrase, or summarize any content
- Add new content that wasn't in the original
- Remove substantive text
- Wrap your response in markdown code fences

Document description: {description}

{overlap_section}\
"""


def _is_normalization_enabled(user_id: int | None) -> bool:
    """Check whether normalization is enabled for the given user."""
    if not getattr(settings, "LLM_DEFAULT_CHEAP_MODEL", ""):
        return False
    if user_id is None:
        return True
    try:
        from accounts.models import Membership
        membership = Membership.objects.filter(user_id=user_id).select_related("org").first()
        if not membership or not membership.org:
            return True
        org_tools = (membership.org.preferences or {}).get("tools", {})
        return org_tools.get("normalize_document", True) is not False
    except Exception:
        return True


def _compute_batches(text: str) -> list[str]:
    """Split text into batches at sentence boundaries respecting token limits.

    Falls back to word-level splitting when no sentence boundaries exist.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    total = len(tokens)

    if total <= MAX_BATCH_TOKENS:
        return [text]

    # Split into sentences first
    sentences = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        # No sentence boundaries — fall back to word-level splitting
        return _split_by_words(text, enc)

    batches: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = len(enc.encode(sentence))

        if sent_tokens > MAX_BATCH_TOKENS:
            # Flush accumulated sentences
            if current_sentences:
                batches.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            # Force-split this oversized sentence by words
            batches.extend(_split_by_words(sentence, enc))
            continue

        if current_tokens + sent_tokens > MAX_BATCH_TOKENS and current_sentences:
            batches.append(" ".join(current_sentences))
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    if current_sentences:
        batches.append(" ".join(current_sentences))

    return batches


def _split_by_words(text: str, enc) -> list[str]:
    """Split text by words respecting MAX_BATCH_TOKENS."""
    words = text.split()
    batches: list[str] = []
    current_words: list[str] = []
    current_tokens = 0

    for word in words:
        word_tokens = len(enc.encode(word))
        if current_words and current_tokens + word_tokens > MAX_BATCH_TOKENS:
            batches.append(" ".join(current_words))
            current_words = [word]
            current_tokens = word_tokens
        else:
            current_words.append(word)
            current_tokens += word_tokens

    if current_words:
        batches.append(" ".join(current_words))

    return batches


def _normalize_batch(
    batch_text: str,
    description: str,
    overlap_context: str,
    user_id: int | None,
    data_room_id: int | None,
) -> str:
    """Run a single normalization batch through the LLM."""
    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    batch_tokens = count_tokens(batch_text)
    logger.info("normalize_batch: starting, tokens=%d", batch_tokens)
    t0 = time.perf_counter()

    context = RunContext.create(
        user_id=user_id,
        conversation_id=data_room_id,
    )

    overlap_section = ""
    if overlap_context:
        overlap_section = (
            "For continuity, here is the raw text immediately before your section:\n"
            "---\n"
            f"{overlap_context}\n"
            "---\n"
            "Do not repeat this text. Continue formatting from where it ends."
        )

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        description=description or "(no description available)",
        overlap_section=overlap_section,
    )

    request = ChatRequest(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=batch_text),
        ],
        model=settings.LLM_DEFAULT_CHEAP_MODEL,
        stream=False,
        tools=[],
        context=context,
    )

    service = get_llm_service()
    response = service.run("simple_chat", request)

    result = response.message.content.strip()
    elapsed = time.perf_counter() - t0
    result_tokens = count_tokens(result)
    logger.info("normalize_batch: done, tokens=%d->%d, %.1fs", batch_tokens, result_tokens, elapsed)

    return result


def _get_tail_overlap(text: str) -> str:
    """Get the last ~OVERLAP_TOKENS tokens of text for continuity context."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= OVERLAP_TOKENS:
        return text
    return enc.decode(tokens[-OVERLAP_TOKENS:])


def _compute_overlaps(batches: list[str]) -> list[str]:
    """Compute overlap contexts upfront from raw text.

    Returns a list parallel to *batches* where overlaps[0] is always empty
    (no preceding batch) and overlaps[i] is the tail of batches[i-1].
    """
    overlaps = [""]
    for batch in batches[:-1]:
        overlaps.append(_get_tail_overlap(batch))
    return overlaps


def _process_single_batch(
    index: int,
    batch: str,
    overlap: str,
    description: str,
    user_id: int | None,
    data_room_id: int | None,
) -> tuple[int, str]:
    """Process one batch with retry logic. Never raises — returns raw text on failure."""
    last_error = None
    result = None
    batch_tokens = count_tokens(batch)
    total_batches = "?"  # logged by caller
    logger.info(
        "normalize_text: batch %d starting, tokens=%d",
        index + 1, batch_tokens,
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = _normalize_batch(
                batch, description, overlap,
                user_id=user_id, data_room_id=data_room_id,
            )
            break
        except Exception as e:
            last_error = e
            logger.warning(
                "normalize_text: batch %d attempt %d failed: %s",
                index + 1, attempt + 1, e,
            )

    if result is None:
        logger.warning(
            "normalize_text: batch %d failed after %d retries, using raw text. error=%s",
            index + 1, MAX_RETRIES, last_error,
        )
        result = batch

    return (index, result)


def normalize_text(
    text: str,
    description: str = "",
    user_id: int | None = None,
    data_room_id: int | None = None,
) -> str:
    """Normalize extracted text to clean markdown using an LLM.

    Returns normalized markdown text, or the original text on failure or
    when normalization is disabled.
    """
    if not _is_normalization_enabled(user_id):
        logger.info("normalize_text: skipped (disabled for user_id=%s)", user_id)
        return text

    batches = _compute_batches(text)
    overlaps = _compute_overlaps(batches)
    logger.info("normalize_text: %d batch(es), user_id=%s", len(batches), user_id)

    total_t0 = time.perf_counter()

    if len(batches) == 1:
        # Single batch — run directly, no thread pool overhead
        _, result = _process_single_batch(
            0, batches[0], overlaps[0], description, user_id, data_room_id,
        )
        total_elapsed = time.perf_counter() - total_t0
        logger.info("normalize_text: done, 1 batch, %.1fs total", total_elapsed)
        return result

    # Multiple batches — run in parallel
    results: list[str | None] = [None] * len(batches)
    workers = min(len(batches), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_single_batch,
                i, batch, overlaps[i], description, user_id, data_room_id,
            ): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    total_elapsed = time.perf_counter() - total_t0
    logger.info(
        "normalize_text: done, %d batch(es), %.1fs total",
        len(batches), total_elapsed,
    )
    return "\n\n".join(results)
