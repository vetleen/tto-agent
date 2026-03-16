"""Structured output pipeline: returns a validated Pydantic model from the LLM."""

from __future__ import annotations

import logging

from llm.core.langchain_utils import to_langchain_messages
from llm.core.model_factory import create_chat_model
from llm.pipelines.base import BasePipeline
from llm.pipelines.registry import get_pipeline_registry
from llm.service.pricing import calculate_cost
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse, Usage

logger = logging.getLogger(__name__)


class StructuredOutputPipeline(BasePipeline):
    """Pipeline that uses .with_structured_output() to return a Pydantic model."""

    id = "structured_output"
    capabilities = {"streaming": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        if not request.model:
            raise ValueError("request.model must be set by the service before calling pipeline")

        output_schema = (request.params or {}).get("output_schema")
        if output_schema is None:
            raise ValueError("request.params must contain 'output_schema' (a Pydantic BaseModel class)")

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

        return ChatResponse(
            message=Message(role="assistant", content=parsed.model_dump_json()),
            metadata={"structured_response": parsed},
            usage=usage,
            model=request.model,
        )


# Register so LLMService can resolve "structured_output"
_registry = get_pipeline_registry()
_registry.register_pipeline(StructuredOutputPipeline())

__all__ = ["StructuredOutputPipeline"]
