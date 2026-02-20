"""
Internal request/result types for policy, logging, and redaction.
Public API stays completion(**kwargs); we convert to these internally.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    """Structured request built from completion(**kwargs)."""
    model: str
    messages: list[dict[str, Any]]
    stream: bool
    metadata: dict[str, Any]
    raw_kwargs: dict[str, Any]
    user: Any = None  # Django User for attribution, optional

    def to_completion_kwargs(self) -> dict[str, Any]:
        """Reconstruct kwargs for the underlying client (model, messages, stream + raw_kwargs)."""
        out = {"model": self.model, "messages": self.messages, "stream": self.stream}
        for k, v in self.raw_kwargs.items():
            if k not in ("model", "messages", "stream", "metadata", "user"):
                out[k] = v
        return out


@dataclass
class LLMResult:
    """Normalized result from provider response or exception."""
    text: str | None = None
    usage: dict[str, int] | None = None  # input_tokens, output_tokens, total_tokens
    cost: float | None = None
    raw_response: Any = None
    raw_response_chunks: list[Any] | None = None  # for streaming: list of chunk objects
    error: Exception | None = None
    provider_response_id: str | None = None
    response_model: str | None = None

    @property
    def input_tokens(self) -> int:
        return (self.usage or {}).get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return (self.usage or {}).get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return (self.usage or {}).get("total_tokens", 0)

    @property
    def succeeded(self) -> bool:
        return self.error is None
