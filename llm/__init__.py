"""
Internal LLM service app.

Public entrypoint:

    from llm import get_llm_service
    service = get_llm_service()
"""

from .service.llm_service import get_llm_service  # noqa: F401
from .service.transcription_service import get_transcription_service  # noqa: F401

__all__ = ["get_llm_service", "get_transcription_service"]

