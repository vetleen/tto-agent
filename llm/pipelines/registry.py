"""Registry for pipelines by id."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from llm.pipelines.base import BasePipeline
from llm.service.errors import LLMConfigurationError


@dataclass
class PipelineRegistry:
    """Maps pipeline_id to pipeline instance."""

    _pipelines: Dict[str, BasePipeline] = field(default_factory=dict)

    def register_pipeline(self, pipeline: BasePipeline) -> None:
        """Register a pipeline by its id."""
        if not pipeline.id:
            raise ValueError("Pipeline id must be non-empty")
        self._pipelines[pipeline.id] = pipeline

    def get_pipeline(self, pipeline_id: str) -> BasePipeline:
        """Return the pipeline for the given id. Raises LLMConfigurationError if missing."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            raise LLMConfigurationError(
                f"Unknown pipeline_id='{pipeline_id}'. "
                f"Available: {list(self._pipelines.keys()) or '[]'}"
            )
        return pipeline

    def clear(self) -> None:
        """Remove all registered pipelines."""
        self._pipelines.clear()


_global_registry: Optional[PipelineRegistry] = None
_global_registry_lock = threading.Lock()


def get_pipeline_registry() -> PipelineRegistry:
    """Return the process-wide PipelineRegistry singleton (thread-safe)."""
    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = PipelineRegistry()
    return _global_registry


__all__ = ["PipelineRegistry", "get_pipeline_registry"]
