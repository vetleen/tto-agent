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
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Severity level of the concern"
    )
    reasoning: str = Field(description="Explanation of the decision")
    user_message: str = Field(
        description="Message to show the user (friendly, non-technical)"
    )


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
