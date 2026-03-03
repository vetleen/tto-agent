"""Tests for core.tokens.count_tokens."""

from unittest.mock import patch, MagicMock

from django.test import TestCase

from core.tokens import count_tokens


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
