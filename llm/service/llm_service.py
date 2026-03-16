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
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterator, Optional

from llm.pipelines.registry import PipelineRegistry, get_pipeline_registry
from llm.service.errors import LLMError, LLMPolicyDenied, LLMProviderError
from llm.service.logger import log_call, log_error, log_stream
from llm.service.policies import resolve_model
from llm.types.context import RunContext
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage
from llm.types.streaming import StreamEvent

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
        run_id = request.context.run_id if request.context else "n/a"
        logger.info(
            "LLMService.run start pipeline=%s model=%s run_id=%s",
            pipeline_id, request.model, run_id,
        )
        t0 = time.monotonic()
        try:
            response = pipeline.run(request)
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_call(request, response, duration_ms)
            logger.info(
                "LLMService.run complete pipeline=%s model=%s duration_ms=%d run_id=%s",
                pipeline_id, request.model, duration_ms, run_id,
            )
            return response
        except LLMError as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms)
            logger.error(
                "LLMService.run failed pipeline=%s model=%s duration_ms=%d error=%s run_id=%s",
                pipeline_id, request.model, duration_ms, type(exc).__name__, run_id,
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms)
            logger.error(
                "LLMService.run failed pipeline=%s model=%s duration_ms=%d error=%s run_id=%s",
                pipeline_id, request.model, duration_ms, type(exc).__name__, run_id,
            )
            raise LLMProviderError(f"Pipeline {pipeline_id} run failed") from exc

    def stream(self, pipeline_id: str, request: ChatRequest) -> Iterator[StreamEvent]:
        """Stream events from a pipeline. Ensures context and model; validates streaming capability."""
        self._ensure_context(request)
        request.model = self._resolve_model(request.model)
        pipeline = self._get_pipeline_registry().get_pipeline(pipeline_id)
        if not pipeline.capabilities.get("streaming", False):
            raise LLMPolicyDenied(f"Pipeline {pipeline_id} does not support streaming")
        run_id = request.context.run_id if request.context else "n/a"
        logger.info(
            "LLMService.stream start pipeline=%s model=%s run_id=%s",
            pipeline_id, request.model, run_id,
        )
        t0 = time.monotonic()
        events: list[StreamEvent] = []
        try:
            for event in pipeline.stream(request):
                events.append(event)
                yield event
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_stream(request, events, duration_ms)
            logger.info(
                "LLMService.stream complete pipeline=%s model=%s duration_ms=%d run_id=%s",
                pipeline_id, request.model, duration_ms, run_id,
            )
        except LLMError as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms, is_stream=True)
            logger.error(
                "LLMService.stream failed pipeline=%s model=%s duration_ms=%d error=%s run_id=%s",
                pipeline_id, request.model, duration_ms, type(exc).__name__, run_id,
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms, is_stream=True)
            logger.error(
                "LLMService.stream failed pipeline=%s model=%s duration_ms=%d error=%s run_id=%s",
                pipeline_id, request.model, duration_ms, type(exc).__name__, run_id,
            )
            raise LLMProviderError(f"Pipeline {pipeline_id} stream failed") from exc

    # -- structured output --------------------------------------------------

    def run_structured(
        self,
        request: ChatRequest,
        output_schema: type[BaseModel],
    ) -> tuple[BaseModel, Usage | None]:
        """Run a structured output request using .with_structured_output().

        Returns (parsed_result, usage). Uses ``include_raw=True`` so we can
        extract usage metadata from the raw AIMessage for cost tracking.
        """
        from llm.core.langchain_utils import to_langchain_messages
        from llm.core.model_factory import create_chat_model
        from llm.service.pricing import calculate_cost

        self._ensure_context(request)
        request.model = self._resolve_model(request.model)
        run_id = request.context.run_id if request.context else "n/a"
        logger.info(
            "LLMService.run_structured start model=%s schema=%s run_id=%s",
            request.model, output_schema.__name__, run_id,
        )
        t0 = time.monotonic()
        try:
            model = create_chat_model(request.model)
            lc_messages = to_langchain_messages(request.messages)
            structured_model = model._client.with_structured_output(
                output_schema, include_raw=True
            )
            result = structured_model.invoke(lc_messages)

            parsed = result["parsed"]
            raw_msg = result["raw"]

            # Extract usage from the raw AIMessage
            usage = None
            usage_meta = getattr(raw_msg, "usage_metadata", None)
            if isinstance(usage_meta, dict) and usage_meta.get("output_tokens"):
                input_tokens = usage_meta.get("input_tokens")
                output_tokens = usage_meta.get("output_tokens")
                details = usage_meta.get("input_token_details") or {}
                cached_tokens = details.get("cache_read") if isinstance(details, dict) else None
                cost = calculate_cost(request.model, input_tokens, output_tokens, cached_tokens)
                usage = Usage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=usage_meta.get("total_tokens"),
                    cached_tokens=cached_tokens,
                    cost_usd=float(cost) if cost is not None else None,
                )

            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "LLMService.run_structured complete model=%s schema=%s "
                "duration_ms=%d run_id=%s",
                request.model, output_schema.__name__, duration_ms, run_id,
            )
            return parsed, usage
        except LLMError:
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.error(
                "LLMService.run_structured failed model=%s schema=%s "
                "duration_ms=%d error=%s run_id=%s",
                request.model, output_schema.__name__, duration_ms,
                type(exc).__name__, run_id,
            )
            raise LLMProviderError(
                f"Structured output failed for model={request.model}"
            ) from exc

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
