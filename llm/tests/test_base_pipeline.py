"""Tests for BasePipeline default stream behavior."""

from unittest.mock import MagicMock

from django.test import TestCase

from llm.pipelines.base import BasePipeline
from llm.types.messages import Message
from llm.types.requests import ChatRequest
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent


class ConcretePipeline(BasePipeline):
    """Minimal concrete pipeline that implements run but not stream."""

    id = "concrete"
    capabilities = {"streaming": False, "tools": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            message=Message(role="assistant", content="ok"),
            model=request.model or "",
            usage=None,
            metadata={},
        )


class BasePipelineStreamTests(TestCase):
    """Test that default stream() raises NotImplementedError."""

    def test_default_stream_raises_not_implemented_error(self):
        pipeline = ConcretePipeline()
        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            stream=True,
            model="gpt-4o",
            context=None,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            list(pipeline.stream(request))
        self.assertIn("concrete", str(ctx.exception))
        self.assertIn("streaming", str(ctx.exception))
        self.assertIn("capabilities", str(ctx.exception))
