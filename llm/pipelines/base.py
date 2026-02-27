"""Base pipeline interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator

from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class BasePipeline(ABC):
    """Abstract base for LLM pipelines."""

    id: str
    capabilities: Dict[str, bool]  # e.g. {"streaming": True, "tools": True}

    @abstractmethod
    def run(self, request: ChatRequest) -> ChatResponse:
        """Run a non-streaming completion."""
        ...

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream completion events. Default: raise; override if supported."""
        raise NotImplementedError(
            f"Pipeline {self.id} does not support streaming (capabilities={self.capabilities})"
        )


__all__ = ["BasePipeline"]
