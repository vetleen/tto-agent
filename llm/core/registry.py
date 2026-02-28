from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, TypeVar

from llm.core.interfaces import ChatModel
from llm.service.errors import LLMConfigurationError


TChatModelFactory = TypeVar("TChatModelFactory", bound=Callable[[str], ChatModel])


@dataclass
class ModelRegistry:
    """
    Registry mapping model name prefixes to ChatModel factories.

    A factory receives the full model name (e.g. "gpt-4o-mini") and returns
    an initialized ChatModel wrapper.
    """

    _prefix_factories: Dict[str, TChatModelFactory] = field(default_factory=dict)

    def register_model_prefix(self, prefix: str, factory: TChatModelFactory) -> None:
        """Register a factory for model names starting with the given prefix."""

        if not prefix:
            raise ValueError("prefix must be non-empty")
        self._prefix_factories[prefix] = factory

    def get_model(self, model_name: str) -> ChatModel:
        """
        Resolve a ChatModel for the given model name using registered prefixes.

        Raises LLMConfigurationError if no prefix matches.
        """

        # Sort longest-first so "gpt-4" beats "gpt-" regardless of registration order.
        for prefix in sorted(self._prefix_factories, key=len, reverse=True):
            if model_name.startswith(prefix):
                return self._prefix_factories[prefix](model_name)
        available_prefixes: List[str] = list(self._prefix_factories.keys())
        raise LLMConfigurationError(
            f"No ChatModel registered for model_name='{model_name}'. "
            f"Configured prefixes: {available_prefixes or '[]'}"
        )

    def clear(self) -> None:
        """Remove all registered prefix factories."""
        self._prefix_factories.clear()


_global_registry: ModelRegistry | None = None
_global_registry_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """Return the process-wide ModelRegistry singleton (thread-safe)."""

    global _global_registry
    if _global_registry is None:
        with _global_registry_lock:
            if _global_registry is None:
                _global_registry = ModelRegistry()
    return _global_registry


__all__ = ["ModelRegistry", "get_model_registry"]

