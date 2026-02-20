"""
Base interface for the LLM backend. Swap LiteLLM for another implementation without changing callers.
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator


class BaseLLMClient(ABC):
    """Abstract client for completion. Implementations: LiteLLMClient, etc."""

    @abstractmethod
    def completion(self, **kwargs: Any) -> Any:
        """
        Sync completion. If stream=True, returns an iterator of provider chunks.
        Otherwise returns the raw completion response.
        """
        ...

    @abstractmethod
    def acompletion(self, **kwargs: Any) -> Any:
        """
        Async completion. If stream=True, returns an async iterator of provider chunks.
        Otherwise returns the raw completion response (awaitable).
        """
        ...
