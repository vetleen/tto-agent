"""
Embedding service abstraction: embed_texts(texts) -> list[list[float]].
Backed by OpenAI (text-embedding-3-large by default) via LangChain.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from django.conf import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts. Returns one vector per text.
    Uses EMBEDDING_MODEL from settings (e.g. text-embedding-3-large).
    """
    if not texts:
        return []
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError("langchain-openai is required for embeddings. pip install langchain-openai")
    model = getattr(settings, "EMBEDDING_MODEL", "text-embedding-3-large")
    timeout = getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 120)
    embeddings = OpenAIEmbeddings(model=model, request_timeout=timeout)
    return embeddings.embed_documents(texts)
