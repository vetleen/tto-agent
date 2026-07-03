"""Tests for StructuredOutputPipeline."""

from unittest.mock import MagicMock, patch

from django.test import TestCase
from pydantic import BaseModel, Field

from llm.core.providers.base import BaseLangChainChatModel
from llm.pipelines.structured_output import StructuredOutputPipeline
from llm.service.errors import LLMProviderError
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
        # Real provider wrapper (mocked LC client) so generate_structured and the
        # shared usage extraction actually run instead of being stubbed away.
        fake_model = BaseLangChainChatModel(model_name="gpt-4o-mini", client=fake_client)
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        response = pipeline.run(self._make_request())

        self.assertEqual(response.metadata["structured_response"], fake_parsed.model_dump())
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
        # Real provider wrapper (mocked LC client) so generate_structured and the
        # shared usage extraction actually run instead of being stubbed away.
        fake_model = BaseLangChainChatModel(model_name="gpt-4o-mini", client=fake_client)
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        response = pipeline.run(self._make_request())

        self.assertIsNone(response.usage)
        self.assertEqual(response.metadata["structured_response"]["description"], "A doc.")

    @patch("llm.pipelines.structured_output.create_chat_model")
    def test_run_retries_once_on_parse_failure_and_sums_usage(self, mock_create):
        """parsed=None on the first attempt → one retry; usage covers both calls."""
        fake_parsed = _TestSchema(description="A patent.", document_type="Patent")

        failed_raw = MagicMock()
        failed_raw.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }
        ok_raw = MagicMock()
        ok_raw.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }
        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = [
            {"raw": failed_raw, "parsed": None, "parsing_error": ValueError("bad json")},
            {"raw": ok_raw, "parsed": fake_parsed, "parsing_error": None},
        ]
        fake_client = MagicMock()
        fake_client.with_structured_output.return_value = fake_structured
        # Real provider wrapper (mocked LC client) so generate_structured and the
        # shared usage extraction actually run instead of being stubbed away.
        fake_model = BaseLangChainChatModel(model_name="gpt-4o-mini", client=fake_client)
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        response = pipeline.run(self._make_request())

        self.assertEqual(fake_structured.invoke.call_count, 2)
        self.assertEqual(response.metadata["structured_response"], fake_parsed.model_dump())
        self.assertIsNotNone(response.usage)
        self.assertEqual(response.usage.prompt_tokens, 200)
        self.assertEqual(response.usage.completion_tokens, 90)
        self.assertEqual(response.usage.total_tokens, 290)

    @patch("llm.pipelines.structured_output.create_chat_model")
    def test_run_raises_provider_error_after_two_parse_failures(self, mock_create):
        failed_raw = MagicMock()
        failed_raw.usage_metadata = None
        fake_structured = MagicMock()
        fake_structured.invoke.return_value = {
            "raw": failed_raw,
            "parsed": None,
            "parsing_error": ValueError("output did not match schema"),
        }
        fake_client = MagicMock()
        fake_client.with_structured_output.return_value = fake_structured
        # Real provider wrapper (mocked LC client) so generate_structured and the
        # shared usage extraction actually run instead of being stubbed away.
        fake_model = BaseLangChainChatModel(model_name="gpt-4o-mini", client=fake_client)
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        with self.assertRaises(LLMProviderError) as ctx:
            pipeline.run(self._make_request())

        self.assertEqual(fake_structured.invoke.call_count, 2)
        self.assertIn("_TestSchema", str(ctx.exception))
        self.assertIn("output did not match schema", str(ctx.exception))

    @patch("llm.core.providers.base._wait_before_retry", return_value=True)
    @patch("llm.pipelines.structured_output.create_chat_model")
    def test_run_transient_error_classified_not_generic(self, mock_create, _wait):
        """A provider overload during a structured call is retried and raised as
        the typed LLMOverloadedError, not a generic 'run failed' — so callers
        (guardrails / PII gate) can tell retryable from fatal."""
        from llm.service.errors import LLMOverloadedError

        overloaded = Exception("overloaded")
        overloaded.status_code = 529
        fake_structured = MagicMock()
        fake_structured.invoke.side_effect = overloaded
        fake_client = MagicMock()
        fake_client.with_structured_output.return_value = fake_structured
        fake_model = BaseLangChainChatModel(model_name="gpt-4o-mini", client=fake_client)
        fake_model._provider_label = "Anthropic"
        mock_create.return_value = fake_model

        pipeline = StructuredOutputPipeline()
        with self.assertRaises(LLMOverloadedError):
            pipeline.run(self._make_request())

    def test_stream_raises_not_implemented(self):
        pipeline = StructuredOutputPipeline()
        request = self._make_request()
        with self.assertRaises(NotImplementedError):
            list(pipeline.stream(request))
