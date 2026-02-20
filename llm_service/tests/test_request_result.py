"""Tests for LLMRequest and LLMResult."""
from django.test import TestCase

from llm_service.request_result import LLMRequest, LLMResult


class LLMRequestTestCase(TestCase):
    """Test LLMRequest construction and to_completion_kwargs."""

    def test_to_completion_kwargs_includes_model_messages_stream(self):
        req = LLMRequest(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
            metadata={},
            raw_kwargs={"temperature": 0.7},
        )
        kwargs = req.to_completion_kwargs()
        self.assertEqual(kwargs["model"], "openai/gpt-4o")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Hi"}])
        self.assertFalse(kwargs["stream"])
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_to_completion_kwargs_excludes_metadata_user(self):
        req = LLMRequest(
            model="m",
            messages=[],
            stream=False,
            metadata={"request_id": "r1"},
            raw_kwargs={"metadata": "x", "user": "y"},
        )
        kwargs = req.to_completion_kwargs()
        self.assertNotIn("metadata", kwargs)
        self.assertNotIn("user", kwargs)


class LLMResultTestCase(TestCase):
    """Test LLMResult and token/cost properties."""

    def test_usage_properties(self):
        result = LLMResult(usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30})
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 20)
        self.assertEqual(result.total_tokens, 30)

    def test_usage_empty_defaults_zero(self):
        result = LLMResult()
        self.assertEqual(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 0)
        self.assertEqual(result.total_tokens, 0)

    def test_succeeded_true_when_no_error(self):
        self.assertTrue(LLMResult(text="ok").succeeded)
        self.assertFalse(LLMResult(error=ValueError("x")).succeeded)
