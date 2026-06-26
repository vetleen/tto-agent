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
from llm.types.messages import Message
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
                "LLMService.run failed pipeline=%s model=%s error=%s",
                pipeline_id, request.model, type(exc).__name__,
                exc_info=True,
                extra={"run_id": run_id, "duration_ms": duration_ms},
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms)
            logger.error(
                "LLMService.run failed pipeline=%s model=%s error=%s",
                pipeline_id, request.model, type(exc).__name__,
                exc_info=True,
                extra={"run_id": run_id, "duration_ms": duration_ms},
            )
            raise LLMProviderError(f"Pipeline {pipeline_id} run failed") from exc

    def run_via_stream(self, pipeline_id: str, request: ChatRequest) -> ChatResponse:
        """Run a streaming pipeline but return a single collapsed ``ChatResponse``.

        Gives callers the blocking ergonomics of :meth:`run` with streaming's
        connection resilience: the provider streams tokens, so a long generation
        never trips a non-streaming read timeout (per Anthropic's "long requests"
        guidance), and the SSE events are reduced back into one ``ChatResponse``.
        The service-level analog of the Anthropic SDK's
        ``.stream().get_final_message()``.

        For headless callers (sub-agents) that don't surface tokens live.
        :meth:`stream` performs the single ``LLMCallLog`` write as it drains, so
        we deliberately don't log again here. Provider failures surface as
        ``error`` stream events; :meth:`_collapse_stream_events` re-raises them as
        the mapped ``LLMProviderError`` subclass so retry classification (e.g.
        ``timeout`` → ``LLMTimeoutError``) still works.
        """
        request.stream = True
        events = list(self.stream(pipeline_id, request))
        return self._collapse_stream_events(events, request)

    @staticmethod
    def _collapse_stream_events(
        events: list[StreamEvent], request: ChatRequest
    ) -> ChatResponse:
        """Reduce a drained stream into a ``ChatResponse`` (pure; no I/O).

        The final assistant text is rebuilt from ``token`` events, reset on each
        ``message_start`` so only the last turn's answer survives (the streaming
        tool-loop pops ``content`` from its final ``message_end``). Usage comes
        from the last ``message_end`` (which carries the aggregated totals). An
        ``error`` event — emitted by the provider instead of raising — is
        re-raised as the mapped exception after the drain (the pipeline stops
        without a synthetic ``message_end`` on error).
        """
        from llm.core.providers.base import exception_for_error_code

        text_parts: list[str] = []
        last_end: dict | None = None
        error_data: dict | None = None
        for event in events:
            et = event.event_type
            if et == "message_start":
                text_parts = []
            elif et == "token":
                text_parts.append(event.data.get("text", ""))
            elif et == "message_end":
                last_end = event.data
            elif et == "error":
                error_data = event.data
            # thinking / tool_start / tool_end: not part of the collapsed result

        if error_data is not None:
            error_code = error_data.get("error_code")
            raise exception_for_error_code(error_code)(
                error_data.get("message") or "LLM stream failed",
                error_code=error_code,
            )

        usage = None
        metadata: dict = {}
        if last_end:
            usage = Usage(
                prompt_tokens=last_end.get("input_tokens"),
                completion_tokens=last_end.get("output_tokens"),
                total_tokens=last_end.get("total_tokens"),
                cached_tokens=last_end.get("cached_tokens"),
                cache_write_tokens=last_end.get("cache_write_tokens"),
                reasoning_tokens=last_end.get("reasoning_tokens"),
                cost_usd=last_end.get("cost_usd"),
            )
            for key in ("response_metadata", "stop_reason", "provider_model_id"):
                if key in last_end:
                    metadata[key] = last_end[key]

        return ChatResponse(
            message=Message(role="assistant", content="".join(text_parts)),
            model=(last_end or {}).get("model") or request.model or "",
            usage=usage,
            metadata=metadata,
        )

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
        _logged = False
        try:
            for event in pipeline.stream(request):
                events.append(event)
                yield event
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_stream(request, events, duration_ms)
            _logged = True
            logger.info(
                "LLMService.stream complete pipeline=%s model=%s duration_ms=%d run_id=%s",
                pipeline_id, request.model, duration_ms, run_id,
            )
        except LLMError as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms, is_stream=True)
            _logged = True
            logger.error(
                "LLMService.stream failed pipeline=%s model=%s error=%s",
                pipeline_id, request.model, type(exc).__name__,
                exc_info=True,
                extra={"run_id": run_id, "duration_ms": duration_ms},
            )
            raise
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log_error(request, exc, duration_ms, is_stream=True)
            _logged = True
            logger.error(
                "LLMService.stream failed pipeline=%s model=%s error=%s",
                pipeline_id, request.model, type(exc).__name__,
                exc_info=True,
                extra={"run_id": run_id, "duration_ms": duration_ms},
            )
            raise LLMProviderError(f"Pipeline {pipeline_id} stream failed") from exc
        finally:
            if not _logged and events:
                try:
                    duration_ms = int((time.monotonic() - t0) * 1000)
                    log_stream(request, events, duration_ms)
                except Exception:
                    logger.debug("Failed to log interrupted stream (non-fatal)")

    # -- structured output --------------------------------------------------

    def run_structured(
        self,
        request: ChatRequest,
        output_schema: type[BaseModel],
    ) -> tuple[BaseModel, Usage | None]:
        """Run a structured output request via the structured_output pipeline.

        Returns (parsed_result, usage). Delegates to ``self.run()`` so the call
        is logged to ``LLMCallLog`` with timing and cost tracking.
        """
        params = dict(request.params or {})
        params["output_schema"] = output_schema
        req = request.model_copy(update={"params": params, "stream": False})
        response = self.run("structured_output", req)
        parsed = output_schema.model_validate(response.metadata["structured_response"])
        return parsed, response.usage

    # -- async bridge -------------------------------------------------------

    async def arun(self, pipeline_id: str, request: ChatRequest) -> ChatResponse:
        """Async wrapper around ``run()``. Executes the blocking call in a thread."""
        return await asyncio.to_thread(self.run, pipeline_id, request)

    _STREAM_SENTINEL = None  # sentinel to signal end of stream

    async def astream(
        self, pipeline_id: str, request: ChatRequest,
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Async wrapper around ``stream()`` with true token-level streaming.

        A background thread runs the sync ``stream()`` generator, pushing each
        event into an ``asyncio.Queue`` so the async caller receives tokens as
        they arrive rather than waiting for the full response.

        If *cancel_event* is provided and set, the producer thread stops
        enqueuing events and the async consumer exits gracefully.

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
                        if cancel_event and cancel_event.is_set():
                            break
                except BaseException as exc:
                    loop.call_soon_threadsafe(q.put_nowait, exc)
                else:
                    loop.call_soon_threadsafe(q.put_nowait, self._STREAM_SENTINEL)

            thread = threading.Thread(target=_produce, daemon=True)
            thread.start()

            while True:
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    item = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item is self._STREAM_SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item

    def _get_stream_semaphore(self) -> asyncio.Semaphore:
        """Lazy-init the semaphore inside a running event loop.

        NOTE: asyncio.Semaphore binds to the event loop that first awaits it.
        This is safe today because astream() is only called from the Channels
        consumer, which runs on a single Daphne event loop per process. If
        astream() is ever called from a second loop (Celery, a management
        command, multi-loop tests), waiters on the foreign loop will raise
        "attached to a different loop" — at that point make this per-loop,
        e.g. a dict keyed by id(asyncio.get_running_loop()).
        """
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
