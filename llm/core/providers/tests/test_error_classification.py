"""Tests for classify_api_error() in BaseLangChainChatModel base module."""

from __future__ import annotations

from unittest import TestCase

from llm.core.providers.base import classify_api_error


def _exc_with_status(status_code: int, message: str = "error") -> Exception:
    exc = Exception(message)
    exc.status_code = status_code
    return exc


class TestClassifyApiError(TestCase):

    def test_429_maps_to_rate_limited(self):
        result = classify_api_error(_exc_with_status(429), "Anthropic")
        self.assertEqual(result.error_code, "rate_limited")
        self.assertIn("rate limiting", result.user_message)
        self.assertIn("Anthropic", result.user_message)
        self.assertEqual(result.log_level, "warning")

    def test_503_maps_to_overloaded(self):
        result = classify_api_error(_exc_with_status(503), "Anthropic")
        self.assertEqual(result.error_code, "overloaded")
        self.assertIn("overloaded", result.user_message)
        self.assertIn("Anthropic", result.user_message)

    def test_529_maps_to_overloaded(self):
        result = classify_api_error(_exc_with_status(529), "OpenAI")
        self.assertEqual(result.error_code, "overloaded")
        self.assertIn("overloaded", result.user_message)
        self.assertIn("OpenAI", result.user_message)
        self.assertEqual(result.log_level, "warning")

    def test_401_maps_to_auth_error(self):
        result = classify_api_error(_exc_with_status(401), "Anthropic")
        self.assertEqual(result.error_code, "auth_error")
        self.assertIn("Authentication failed", result.user_message)
        self.assertEqual(result.log_level, "error")

    def test_403_maps_to_auth_error(self):
        result = classify_api_error(_exc_with_status(403), "Google")
        self.assertEqual(result.error_code, "auth_error")
        self.assertIn("Google", result.user_message)

    def test_400_with_token_keyword_maps_to_request_too_large(self):
        result = classify_api_error(
            _exc_with_status(400, "maximum context length exceeded, too many tokens"),
            "OpenAI",
        )
        self.assertEqual(result.error_code, "request_too_large")
        self.assertIn("too large", result.user_message)

    def test_400_generic_maps_to_unknown(self):
        result = classify_api_error(
            _exc_with_status(400, "invalid parameter"),
            "Anthropic",
        )
        self.assertEqual(result.error_code, "unknown")

    def test_500_maps_to_server_error(self):
        result = classify_api_error(_exc_with_status(500), "Anthropic")
        self.assertEqual(result.error_code, "server_error")
        self.assertIn("internal error", result.user_message)
        self.assertEqual(result.log_level, "error")

    def test_408_maps_to_timeout(self):
        result = classify_api_error(_exc_with_status(408), "OpenAI")
        self.assertEqual(result.error_code, "timeout")
        self.assertIn("timed out", result.user_message)

    def test_timeout_error_maps_to_timeout(self):
        result = classify_api_error(TimeoutError("timed out"), "Anthropic")
        self.assertEqual(result.error_code, "timeout")
        self.assertIn("timed out", result.user_message)
        self.assertIn("Anthropic", result.user_message)

    def test_connection_error_maps_to_connection_error(self):
        result = classify_api_error(ConnectionError("refused"), "Anthropic")
        self.assertEqual(result.error_code, "connection_error")
        self.assertIn("Unable to reach", result.user_message)

    def test_os_error_maps_to_connection_error(self):
        result = classify_api_error(OSError("network unreachable"), "Google")
        self.assertEqual(result.error_code, "connection_error")
        self.assertIn("Google", result.user_message)

    def test_unknown_exception_maps_to_unknown(self):
        result = classify_api_error(RuntimeError("something broke"), "Anthropic")
        self.assertEqual(result.error_code, "unknown")
        self.assertIn("unexpected error", result.user_message)
        self.assertEqual(result.log_level, "error")

    def test_provider_label_appears_in_messages(self):
        """Provider label is included in user messages for most error codes."""
        for status, expected_code in [(429, "rate_limited"), (503, "overloaded"),
                                       (401, "auth_error"), (500, "server_error")]:
            result = classify_api_error(_exc_with_status(status), "TestProvider")
            self.assertIn("TestProvider", result.user_message,
                          f"Provider label missing for status {status}")
