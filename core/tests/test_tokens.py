"""Tests for core.tokens.count_tokens."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from django.test import TestCase
from pydantic import BaseModel, Field

from core.tokens import count_tokens, estimate_chat_request_tokens
from llm.types import Message


class CountTokensTests(TestCase):
    def test_empty_string(self):
        self.assertEqual(count_tokens(""), 0)

    def test_none_input(self):
        self.assertEqual(count_tokens(None), 0)

    def test_whitespace_only(self):
        self.assertEqual(count_tokens("   "), 0)

    def test_known_string_returns_positive(self):
        result = count_tokens("Hello, world!")
        self.assertGreater(result, 0)

    def test_longer_text_returns_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("Hello, this is a much longer sentence with many words.")
        self.assertGreater(long, short)

    def test_fallback_when_tiktoken_fails(self):
        """Simulate tiktoken failure by importing a broken module."""
        import tiktoken
        original = tiktoken.get_encoding
        tiktoken.get_encoding = MagicMock(side_effect=RuntimeError("fail"))
        try:
            result = count_tokens("Hello world test")
            self.assertGreater(result, 0)
        finally:
            tiktoken.get_encoding = original

    def test_fallback_returns_positive_for_words(self):
        import tiktoken
        original = tiktoken.get_encoding
        tiktoken.get_encoding = MagicMock(side_effect=RuntimeError("fail"))
        try:
            text = "one two three four five"
            result = count_tokens(text)
            # Fallback uses max of word-count and char-estimate
            self.assertGreaterEqual(result, 5)
        finally:
            tiktoken.get_encoding = original

    def test_fallback_uses_char_estimate_for_dense_text(self):
        import tiktoken
        original = tiktoken.get_encoding
        tiktoken.get_encoding = MagicMock(side_effect=RuntimeError("fail"))
        try:
            text = "abcdefghijklmnop"
            result = count_tokens(text)
            self.assertGreater(result, 1)
        finally:
            tiktoken.get_encoding = original


class TiktokenFallbackWarnOnceTests(TestCase):
    """The tiktoken-failure WARNING fires once, then drops to DEBUG.

    A persistent failure (e.g. no network to fetch the BPE file) would
    otherwise emit one WARNING per count — flooding logs and Sentry.
    """

    def setUp(self):
        import core.tokens
        self._saved_flag = core.tokens._tiktoken_fallback_warned
        core.tokens._tiktoken_fallback_warned = False

    def tearDown(self):
        import core.tokens
        core.tokens._tiktoken_fallback_warned = self._saved_flag

    def test_first_failure_warns_subsequent_do_not(self):
        import tiktoken
        original = tiktoken.get_encoding
        tiktoken.get_encoding = MagicMock(side_effect=RuntimeError("fail"))
        try:
            with self.assertLogs("core.tokens", level="WARNING"):
                count_tokens("first call")
            with self.assertNoLogs("core.tokens", level="WARNING"):
                count_tokens("second call")
                count_tokens("third call")
        finally:
            tiktoken.get_encoding = original

    def test_fallback_still_counts_after_warning_suppressed(self):
        import tiktoken
        original = tiktoken.get_encoding
        tiktoken.get_encoding = MagicMock(side_effect=RuntimeError("fail"))
        try:
            count_tokens("first call")
            self.assertGreater(count_tokens("one two three"), 0)
        finally:
            tiktoken.get_encoding = original


class CountTokensListContentTests(TestCase):
    """Test count_tokens with multimodal list content."""

    def test_text_block(self):
        content = [{"type": "text", "text": "Hello world"}]
        result = count_tokens(content)
        self.assertGreater(result, 0)

    def test_image_block_returns_estimate(self):
        content = [{"type": "image", "base64": "abc123"}]
        result = count_tokens(content)
        self.assertEqual(result, 170)

    def test_image_url_block_returns_estimate(self):
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]
        result = count_tokens(content)
        self.assertEqual(result, 170)

    def test_mixed_blocks(self):
        content = [
            {"type": "text", "text": "Describe this image:"},
            {"type": "image", "base64": "abc123"},
        ]
        result = count_tokens(content)
        text_tokens = count_tokens("Describe this image:")
        self.assertEqual(result, text_tokens + 170)

    def test_empty_list(self):
        self.assertEqual(count_tokens([]), 0)

    def test_non_dict_block(self):
        content = ["plain string"]
        result = count_tokens(content)
        self.assertGreater(result, 0)

    def test_unknown_block_type(self):
        content = [{"type": "audio", "data": "binary"}]
        result = count_tokens(content)
        self.assertGreater(result, 0)


class EstimateChatRequestTokensTests(TestCase):
    class SearchArgs(BaseModel):
        query: str = Field(description="Search query")

    def test_more_message_content_increases_estimate(self):
        short = estimate_chat_request_tokens([
            Message(role="user", content="Hello"),
        ])
        long = estimate_chat_request_tokens([
            Message(role="user", content="Hello " * 100),
        ])
        self.assertGreater(long, short)

    def test_multimodal_image_increases_estimate(self):
        text_only = estimate_chat_request_tokens([
            Message(role="user", content=[{"type": "text", "text": "Look"}]),
        ])
        with_image = estimate_chat_request_tokens([
            Message(role="user", content=[
                {"type": "text", "text": "Look"},
                {"type": "image", "base64": "abc"},
            ]),
        ])
        self.assertGreaterEqual(with_image - text_only, 170)

    def test_tool_schema_increases_estimate(self):
        messages = [Message(role="user", content="Find it")]
        without_tool = estimate_chat_request_tokens(messages)
        tool = SimpleNamespace(
            name="search",
            description="Search for records",
            args_schema=self.SearchArgs,
        )
        with_tool = estimate_chat_request_tokens(messages, [tool])
        self.assertGreater(with_tool, without_tool)
