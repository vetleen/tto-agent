"""Tests for StructuredOutputPipeline."""

from unittest.mock import MagicMock, patch

from django.test import TestCase
from pydantic import BaseModel, Field

from llm.pipelines.structured_output import StructuredOutputPipeline
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest


class _TestSchema(BaseModel):
    description: str = Field(description="A description")
    document_type: str = Field(description="Document type")


class StructuredOutputPipelineTests(TestCase):

    def _make_request(self, **overrides):
        defaults = dict(
            messages=[Message(role="user", content="Describe this doc")],
            stream=False,
            model="gpt-4o-mini",
            params={"output_schema": _TestSchema},
            context=RunContext.create(),
        )
        defaults.update(overrides)
        return ChatRequest(**defaults)

    @patch("llm.pipelines.structured_output.create_chat_model")
    def test_run_returns_parsed_in_metadata(self, mock_create):
        fake_parsed = _TestSchema(description="A patent.", document_type="Patent")
        fake_raw_msg = MagicMock()
        fake_raw_msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "raw": fake_raw_msg,
            "parsed": fake_parsed,
            "parsing_error": None,
        }
        fake_client = MagicMock()
        fake_client.with_structured_output.return_value = fake_structured
        fake_model = MagicMock()
        fake_model._client = fake_client
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        response = pipeline.run(self._make_request())

        self.assertEqual(response.metadata["structured_response"], fake_parsed)
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage.prompt_tokens, 100)
        self.assertEqual(response.usage.completion_tokens, 50)
        self.assertIn('"description"', response.message.content)

    def test_run_missing_model_raises(self):
        pipeline = StructuredOutputPipeline()
        request = self._make_request(model=None)
        with self.assertRaises(ValueError) as ctx:
            pipeline.run(request)
        self.assertIn("request.model", str(ctx.exception))

    def test_run_missing_schema_raises(self):
        pipeline = StructuredOutputPipeline()
        request = self._make_request(params={})
        with self.assertRaises(ValueError) as ctx:
            pipeline.run(request)
        self.assertIn("output_schema", str(ctx.exception))

    @patch("llm.pipelines.structured_output.create_chat_model")
    def test_run_no_usage_metadata(self, mock_create):
        fake_parsed = _TestSchema(description="A doc.", document_type="Report")
        fake_raw_msg = MagicMock()
        fake_raw_msg.usage_metadata = None
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "raw": fake_raw_msg,
            "parsed": fake_parsed,
            "parsing_error": None,
        }
        fake_client = MagicMock()
        fake_client.with_structured_output.return_value = fake_structured
        fake_model = MagicMock()
        fake_model._client = fake_client
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        response = pipeline.run(self._make_request())

        self.assertIsNone(response.usage)
        self.assertEqual(response.metadata["structured_response"].description, "A doc.")

    def test_stream_raises_not_implemented(self):
        pipeline = StructuredOutputPipeline()
        request = self._make_request()
        with self.assertRaises(NotImplementedError):
            list(pipeline.stream(request))
