"""Structured output pipeline: returns a validated Pydantic model from the LLM."""

from __future__ import annotations

import logging
import warnings

from llm.core.langchain_utils import to_langchain_messages
from llm.core.model_factory import create_chat_model
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import get_pipeline_registry
from llm.service.errors import LLMProviderError
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage

logger = logging.getLogger(__name__)


def _sum_usage(a: Usage | None, b: Usage | None) -> Usage | None:
    """Combine usage from multiple attempts so logs reflect actual spend."""
    if a is None:
        return b
    if b is None:
        return a

    def add(x: int | None, y: int | None) -> int | None:
        if x is None and y is None:
            return None
        return (x or 0) + (y or 0)

    cost = None
    if a.cost_usd is not None or b.cost_usd is not None:
        cost = (a.cost_usd or 0.0) + (b.cost_usd or 0.0)
    return Usage(
        prompt_tokens=add(a.prompt_tokens, b.prompt_tokens),
        completion_tokens=add(a.completion_tokens, b.completion_tokens),
        total_tokens=add(a.total_tokens, b.total_tokens),
        cached_tokens=add(a.cached_tokens, b.cached_tokens),
        cost_usd=cost,
    )


class StructuredOutputPipeline(BasePipeline):
    """Pipeline that uses .with_structured_output() to return a Pydantic model."""

    id = "structured_output"
    capabilities = {"streaming": False}

    # One retry when the model returns output that fails schema parsing.
    _MAX_ATTEMPTS = 2

    def run(self, request: ChatRequest) -> ChatResponse:
        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")

        output_schema = (request.params or {}).get("output_schema")
        if output_schema is None:
            raise ValueError("request.params must contain 'output_schema' (a Pydantic BaseModel class)")

        model = create_chat_model(request.model)
        lc_messages = to_langchain_messages(request.messages)
        structured_client = model._client.with_structured_output(
            output_schema, include_raw=True
        )

        schema_name = getattr(output_schema, "__name__", str(output_schema))
        parsed = None
        usage: Usage | None = None
        parsing_error = None
        for attempt in range(self._MAX_ATTEMPTS):
            # LangChain's include_raw=True wrapper types the `parsed` field as
            # Optional[schema]=None, causing a spurious Pydantic serializer
            # warning when the field is populated. Suppress it.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                # Route through the provider so transient errors are retried and
                # classified into a typed LLMProviderError, instead of surfacing
                # as a raw SDK exception the caller can't tell apart from fatal.
                # Also reuses the base usage extraction (cache-write/reasoning).
                result, attempt_usage = model.generate_structured(
                    structured_client, lc_messages, request
                )

            usage = _sum_usage(usage, attempt_usage)
            parsed = result["parsed"]
            if parsed is not None:
                break
            # include_raw=True swallows parse failures into parsing_error
            # instead of raising; parsed is None in that case.
            parsing_error = result.get("parsing_error")
            if attempt < self._MAX_ATTEMPTS - 1:
                logger.warning(
                    "structured_output: model=%s returned unparsable output for "
                    "schema=%s (attempt %d/%d); retrying. parsing_error=%s",
                    request.model, schema_name, attempt + 1, self._MAX_ATTEMPTS,
                    str(parsing_error)[:500],
                )

        if parsed is None:
            logger.error(
                "structured_output: model=%s failed to produce valid %s after "
                "%d attempts. parsing_error=%s",
                request.model, schema_name, self._MAX_ATTEMPTS,
                str(parsing_error)[:500],
            )
            raise LLMProviderError(
                f"The model failed to return valid structured output for schema "
                f"'{schema_name}' after {self._MAX_ATTEMPTS} attempts: "
                f"{str(parsing_error)[:300] if parsing_error else 'no parsed output'}",
                error_code="structured_output_parse_error",
            )

        return ChatResponse(
            message=Message(role="assistant", content=parsed.model_dump_json()),
            metadata={"structured_response": parsed.model_dump()},
            usage=usage,
            model=request.model,
        )


# Register so LLMService can resolve "structured_output"
_registry = get_pipeline_registry()
_registry.register_pipeline(StructuredOutputPipeline())

__all__ = ["StructuredOutputPipeline"]
