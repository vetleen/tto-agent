"""Structured output schemas for LLM responses."""

from pydantic import BaseModel, Field


class DocumentDescriptionOutput(BaseModel):
    """Structured output for document description and type classification."""

    description: str = Field(description="A relevance-signal paragraph (~100 tokens)")
    document_type: str = Field(
        description="Document type classification (e.g. Agreement, Patent, License, Report)"
    )
