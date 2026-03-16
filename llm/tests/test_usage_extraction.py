"""Tests for BaseLangChainChatModel._extract_usage_dict fallback logic."""

from django.test import SimpleTestCase

from llm.core.providers.base import BaseLangChainChatModel


class _FakeResult:
    """Minimal stand-in for a LangChain AIMessage / AIMessageChunk."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


class ExtractUsageDictTests(SimpleTestCase):
    """Verify _extract_usage_dict picks the right source for token counts."""

    def test_usage_metadata_preferred(self):
        result = _FakeResult(
            usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        )
        usage = BaseLangChainChatModel._extract_usage_dict(result)
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 20)
        self.assertEqual(usage["total_tokens"], 30)

    def test_response_metadata_usage_fallback(self):
        """OpenAI Responses API: usage lives in response_metadata['usage']."""
        result = _FakeResult(
            usage_metadata=None,
            response_metadata={
                "usage": {"input_tokens": 453, "output_tokens": 163, "total_tokens": 616},
            },
        )
        usage = BaseLangChainChatModel._extract_usage_dict(result)
        self.assertEqual(usage["input_tokens"], 453)
        self.assertEqual(usage["output_tokens"], 163)
        self.assertEqual(usage["total_tokens"], 616)

    def test_empty_usage_metadata_falls_through(self):
        """usage_metadata present but with no output_tokens should fall through."""
        result = _FakeResult(
            usage_metadata={"input_tokens": 10, "output_tokens": 0},
            response_metadata={
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        )
        usage = BaseLangChainChatModel._extract_usage_dict(result)
        # Should have fallen through to response_metadata
        self.assertEqual(usage["output_tokens"], 20)

    def test_none_result_returns_none(self):
        self.assertIsNone(BaseLangChainChatModel._extract_usage_dict(None))

    def test_no_metadata_returns_none(self):
        result = _FakeResult()
        self.assertIsNone(BaseLangChainChatModel._extract_usage_dict(result))
