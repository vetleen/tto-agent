"""
LLMService: facade for running and streaming LLM pipelines.

Use from other apps (Channels consumers, Django views, Celery tasks):

    from llm import get_llm_service
    from llm.types import ChatRequest, Message, RunContext

    service = get_llm_service()
    request = ChatRequest(
        messages=[Message(role="user", content="Hello")],
        stream=False,
        model=None,  # use default from DEFAULT_LLM_MODEL / LLM_ALLOWED_MODELS
        context=RunContext.create(user_id=user.id, conversation_id=conv_id),
    )
    response = service.run("simple_chat", request)

For streaming:

    for event in service.stream("simple_chat", request):
        # send event.model_dump() or event.model_dump_json() over WebSocket
        ...
"""

from __future__ import annotations

from typing import Iterator

from llm.pipelines.registry import get_pipeline_registry
from llm.service.errors import LLMError, LLMPolicyDenied, LLMProviderError
from llm.service.policies import resolve_model
from llm.types.context import RunContext
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class LLMService:
    """Facade that routes pipeline calls, enforces policies, and normalizes errors."""

    def run(self, pipeline_id: str, request: ChatRequest) -> ChatResponse:
        """Run a non-streaming pipeline. Ensures context and model are set; delegates to pipeline."""
        self._ensure_context(request)
        request.model = resolve_model(request.model)
        pipeline = get_pipeline_registry().get_pipeline(pipeline_id)
        if request.stream and not pipeline.capabilities.get("streaming", False):
            raise LLMPolicyDenied(f"Pipeline {pipeline_id} does not support streaming")
        try:
            return pipeline.run(request)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Pipeline {pipeline_id} run failed") from exc

    def stream(self, pipeline_id: str, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream events from a pipeline. Ensures context and model; validates streaming capability."""
        self._ensure_context(request)
        request.model = resolve_model(request.model)
        pipeline = get_pipeline_registry().get_pipeline(pipeline_id)
        if not pipeline.capabilities.get("streaming", False):
            raise LLMPolicyDenied(f"Pipeline {pipeline_id} does not support streaming")
        try:
            yield from pipeline.stream(request)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Pipeline {pipeline_id} stream failed") from exc

    @staticmethod
    def _ensure_context(request: ChatRequest) -> None:
        if request.context is None:
            request.context = RunContext.create()


_global_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Return the process-wide LLMService singleton."""
    global _global_service
    if _global_service is None:
        _global_service = LLMService()
    return _global_service


__all__ = ["LLMService", "get_llm_service"]
