"""Pydantic schemas for guardrail classifier and reviewer outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClassifierResult(BaseModel):
    """Layer 1 cheap model classifier output (single chunk)."""

    is_suspicious: bool = Field(description="Whether the input appears suspicious")
    concern_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Standardized concern tags: prompt_injection, jailbreak, "
            "data_extraction, social_engineering, encoding_bypass, delimiter_injection"
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(description="Brief explanation of the classification")


class ChunkClassification(BaseModel):
    """Classification for a single chunk within a batch."""

    chunk_index: int = Field(description="The chunk_index as shown in the input")
    is_suspicious: bool = Field(description="Whether the chunk appears suspicious")
    concern_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Standardized concern tags: prompt_injection, jailbreak, "
            "data_extraction, social_engineering, encoding_bypass, delimiter_injection"
        ),
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(description="Brief explanation of the classification")


class BatchClassifierResult(BaseModel):
    """Layer 1 cheap model classifier output for a batch of chunks."""

    results: list[ChunkClassification] = Field(
        description="Classification result for each chunk in the batch",
    )


class ReviewerDecision(BaseModel):
    """Layer 2 top model judge output."""

    action: Literal["dismiss", "warn", "block", "suspend"] = Field(
        description="Decision: dismiss (false positive), warn, block, or suspend"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in your chosen action (0.0 = uncertain, 1.0 = certain)",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level of the concern"
    )
    reasoning: str = Field(description="Explanation of the decision")
    user_message: str = Field(
        description="Message to show the user (friendly, non-technical)"
    )


class ChunkReviewDecision(BaseModel):
    """Layer 2 top-model judge output for a flagged document chunk.

    Distinct from ``ReviewerDecision`` (the chat-message reviewer): a document
    chunk has no live user to message and the only meaningful outcomes are
    keep-it-retrievable vs. exclude-it-from-retrieval, so the action collapses to
    ``allow``/``quarantine`` and there is no ``user_message``/``warn``/``suspend``.
    """

    action: Literal["allow", "quarantine"] = Field(
        description="Decision: allow (false positive — keep the chunk retrievable) or quarantine (exclude it)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are in your chosen action (0.0 = uncertain, 1.0 = certain)",
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level of the concern"
    )
    reasoning: str = Field(description="Explanation of the decision")


class HeuristicResult(BaseModel):
    """Layer 0 heuristic scan output."""

    is_suspicious: bool = False
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    matched_patterns: list[str] = Field(default_factory=list)

    @property
    def should_block(self) -> bool:
        """High-confidence heuristic matches warrant immediate blocking."""
        return self.is_suspicious and self.confidence >= 0.9
