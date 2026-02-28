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

import asyncio
import os
import queue
import threading
import time
from typing import AsyncIterator, Callable, Iterator, Optional

from llm.pipelines.registry import PipelineRegistry, get_pipeline_registry
from llm.service.errors import LLMError, LLMPolicyDenied, LLMProviderError
from llm.service.logger import log_call, log_error, log_stream
from llm.service.policies import resolve_model
from llm.types.context import RunContext
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent

LLM_MAX_CONCURRENT_STREAMS = int(os.environ.get("LLM_MAX_CONCURRENT_STREAMS", "20"))


class LLMService:
    """Facade that routes pipeline calls, enforces policies, and normalizes errors.

    Accepts optional dependency overrides for testability. When omitted the
    process-wide singletons are used, so ``get_llm_service()`` keeps working
    unchanged.

    Future consideration: per-tenant registries could be passed here to
    support multi-tenant deployments without global state.
    """

    def __init__(
        self,
        pipeline_registry: PipelineRegistry | None = None,
        resolve_model_fn: Callable[[str | None], str] | None = None,
    ) -> None:
        self._pipeline_registry = pipeline_registry
        self._resolve_model_fn = resolve_model_fn
        self._stream_semaphore: Optional[asyncio.Semaphore] = None

    # -- private accessors --------------------------------------------------

    def _get_pipeline_registry(self) -> PipelineRegistry:
        return self._pipeline_registry or get_pipeline_registry()

    def _resolve_model(self, model: str | None) -> str:
        fn = self._resolve_model_fn or resolve_model
        return fn(model)

    # -- sync API -----------------------------------------------------------

    def run(self, pipeline_id: str, request: ChatRequest) -> ChatResponse:
        """Run a non-streaming pipeline. Ensures context and model are set; delegates to pipeline."""
        self._ensure_context(request)
        request.model = self._resolve_model(request.model)
        pipeline = self._get_pipeline_registry().get_pipeline(pipeline_id)
        if request.stream and not pipeline.capabilities.get("streaming", False):
            raise LLMPolicyDenied(f"Pipeline {pipeline_id} does not support streaming")
        t0 = time.monotonic()
        try:
            response = pipeline.run(request)
            log_call(request, response, int((time.monotonic() - t0) * 1000))
            return response
        except LLMError as exc:
            log_error(request, exc, int((time.monotonic() - t0) * 1000))
            raise
        except Exception as exc:
            log_error(request, exc, int((time.monotonic() - t0) * 1000))
            raise LLMProviderError(f"Pipeline {pipeline_id} run failed") from exc

    def stream(self, pipeline_id: str, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream events from a pipeline. Ensures context and model; validates streaming capability."""
        self._ensure_context(request)
        request.model = self._resolve_model(request.model)
        pipeline = self._get_pipeline_registry().get_pipeline(pipeline_id)
        if not pipeline.capabilities.get("streaming", False):
            raise LLMPolicyDenied(f"Pipeline {pipeline_id} does not support streaming")
        t0 = time.monotonic()
        events: list[StreamEvent] = []
        try:
            for event in pipeline.stream(request):
                events.append(event)
                yield event
            log_stream(request, events, int((time.monotonic() - t0) * 1000))
        except LLMError as exc:
            log_error(request, exc, int((time.monotonic() - t0) * 1000), is_stream=True)
            raise
        except Exception as exc:
            log_error(request, exc, int((time.monotonic() - t0) * 1000), is_stream=True)
            raise LLMProviderError(f"Pipeline {pipeline_id} stream failed") from exc

    # -- async bridge -------------------------------------------------------

    async def arun(self, pipeline_id: str, request: ChatRequest) -> ChatResponse:
        """Async wrapper around ``run()``. Executes the blocking call in a thread."""
        return await asyncio.to_thread(self.run, pipeline_id, request)

    _STREAM_SENTINEL = None  # sentinel to signal end of stream

    async def astream(
        self, pipeline_id: str, request: ChatRequest
    ) -> AsyncIterator[StreamEvent]:
        """Async wrapper around ``stream()`` with true token-level streaming.

        A background thread runs the sync ``stream()`` generator, pushing each
        event into an ``asyncio.Queue`` so the async caller receives tokens as
        they arrive rather than waiting for the full response.

        Concurrent streams are capped by ``LLM_MAX_CONCURRENT_STREAMS`` (default 20).
        """
        sem = self._get_stream_semaphore()
        async with sem:
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[StreamEvent | BaseException | None] = asyncio.Queue()

            def _produce() -> None:
                try:
                    for event in self.stream(pipeline_id, request):
                        loop.call_soon_threadsafe(q.put_nowait, event)
                except BaseException as exc:
                    loop.call_soon_threadsafe(q.put_nowait, exc)
                else:
                    loop.call_soon_threadsafe(q.put_nowait, self._STREAM_SENTINEL)

            thread = threading.Thread(target=_produce, daemon=True)
            thread.start()

            while True:
                item = await q.get()
                if item is self._STREAM_SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item

    def _get_stream_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init the semaphore inside a running event loop."""
        if self._stream_semaphore is None:
            self._stream_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENT_STREAMS)
        return self._stream_semaphore

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _ensure_context(request: ChatRequest) -> None:
        if request.context is None:
            request.context = RunContext.create()


_global_service: LLMService | None = None
_global_service_lock = threading.Lock()


def get_llm_service() -> LLMService:
    """Return the process-wide LLMService singleton (thread-safe)."""
    global _global_service
    if _global_service is None:
        with _global_service_lock:
            if _global_service is None:
                _global_service = LLMService()
    return _global_service


__all__ = ["LLMService", "get_llm_service"]
