from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


@runtime_checkable
class ChatModel(Protocol):
    """
    Provider-agnostic chat model interface.

    Concrete implementations should wrap LangChain chat models
    (e.g. ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI) rather than
    calling provider SDKs directly.
    """

    name: str

    def generate(self, request: ChatRequest) -> ChatResponse:
        """Run a single non-streaming chat completion."""

        ...

    def stream(self, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream a sequence of events (tokens, metadata, etc.) for a chat completion."""

        ...


__all__ = ["ChatModel"]

