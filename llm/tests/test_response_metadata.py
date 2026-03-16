"""Tests for _extract_response_metadata() and _extract_reasoning_tokens() on BaseLangChainChatModel."""

from django.test import TestCase

from llm.core.providers.base import BaseLangChainChatModel


class _FakeResult:
    """Simulates a LangChain AIMessage with response_metadata."""

    def __init__(self, response_metadata=None):
        self.response_metadata = response_metadata


class ExtractResponseMetadataTests(TestCase):
    """Test _extract_response_metadata with different provider formats."""

    def test_anthropic_format(self):
        result = _FakeResult(response_metadata={
            "stop_reason": "end_turn",
            "model_id": "claude-sonnet-4-6-20250514",
        })
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["stop_reason"], "end_turn")
        self.assertEqual(meta["provider_model_id"], "claude-sonnet-4-6-20250514")
        self.assertIn("stop_reason", meta["response_metadata"])

    def test_openai_format(self):
        result = _FakeResult(response_metadata={
            "finish_reason": "stop",
            "model_name": "gpt-5-mini-2025-01-31",
        })
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["stop_reason"], "stop")
        self.assertEqual(meta["provider_model_id"], "gpt-5-mini-2025-01-31")

    def test_generic_model_key(self):
        result = _FakeResult(response_metadata={
            "model": "gemini-2.5-flash",
        })
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["provider_model_id"], "gemini-2.5-flash")
        self.assertEqual(meta["stop_reason"], "")

    def test_empty_response_metadata(self):
        result = _FakeResult(response_metadata={})
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["stop_reason"], "")
        self.assertEqual(meta["provider_model_id"], "")
        self.assertEqual(meta["response_metadata"], {})

    def test_none_response_metadata(self):
        result = _FakeResult(response_metadata=None)
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["stop_reason"], "")
        self.assertEqual(meta["provider_model_id"], "")

    def test_no_response_metadata_attr(self):
        result = object()
        meta = BaseLangChainChatModel._extract_response_metadata(result)
        self.assertEqual(meta["stop_reason"], "")
        self.assertEqual(meta["provider_model_id"], "")


class ExtractReasoningTokensTests(TestCase):
    """Test _extract_reasoning_tokens with different usage metadata formats."""

    def test_extracts_reasoning_from_output_token_details(self):
        usage_meta = {
            "output_tokens": 500,
            "output_token_details": {"reasoning": 200},
        }
        self.assertEqual(BaseLangChainChatModel._extract_reasoning_tokens(usage_meta), 200)

    def test_extracts_reasoning_tokens_key(self):
        usage_meta = {
            "output_tokens": 500,
            "output_token_details": {"reasoning_tokens": 150},
        }
        self.assertEqual(BaseLangChainChatModel._extract_reasoning_tokens(usage_meta), 150)

    def test_returns_none_when_no_details(self):
        usage_meta = {"output_tokens": 500}
        self.assertIsNone(BaseLangChainChatModel._extract_reasoning_tokens(usage_meta))

    def test_returns_none_for_none_input(self):
        self.assertIsNone(BaseLangChainChatModel._extract_reasoning_tokens(None))

    def test_returns_none_when_details_empty(self):
        usage_meta = {"output_token_details": {}}
        self.assertIsNone(BaseLangChainChatModel._extract_reasoning_tokens(usage_meta))
