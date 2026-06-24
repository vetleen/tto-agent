"""Tests for classify_api_error() in BaseLangChainChatModel base module."""

from __future__ import annotations

from unittest import TestCase

from llm.core.providers.base import (
    classify_api_error,
    _is_overloaded_error,
    _is_rate_limit_error,
    _is_retryable_transient_error,
)


def _exc_with_status(status_code: int, message: str = "error") -> Exception:
    exc = Exception(message)
    exc.status_code = status_code
    return exc


def _gemini_exc(*, code=None, status=None, message="error") -> Exception:
    """Mimic ``langchain_google_genai.ChatGoogleGenerativeAIError``.

    The wrapper carries no status attributes of its own; the real status lives
    on the chained ``google.genai.errors`` cause (numeric ``code`` + gRPC
    ``status`` enum). The wrapper message also embeds the enum and number.
    """
    cause = Exception(message)
    if code is not None:
        cause.code = code
    if status is not None:
        cause.status = status
    wrapper = Exception(message)
    wrapper.__cause__ = cause
    return wrapper


def _exc_mid_stream(error_type: str, message: str = "error") -> Exception:
    """Simulate an SDK exception raised mid-stream over a 200 response.

    The Anthropic SDK maps SSE ``error`` events to ``APIStatusError`` whose
    ``status_code`` reflects the initial 200 response, with the real error
    type in ``exc.body["error"]["type"]``.
    """
    exc = Exception(message)
    exc.status_code = 200
    exc.body = {"type": "error", "error": {"type": error_type, "message": message}}
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

    def test_mid_stream_overloaded_body_maps_to_overloaded(self):
        """SSE-delivered overloaded_error (status_code=200) still classifies correctly."""
        exc = _exc_mid_stream("overloaded_error", "Overloaded")
        result = classify_api_error(exc, "Anthropic")
        self.assertEqual(result.error_code, "overloaded")
        self.assertIn("overloaded", result.user_message)
        self.assertIn("Anthropic", result.user_message)
        self.assertEqual(result.log_level, "warning")

    def test_mid_stream_rate_limit_body_maps_to_rate_limited(self):
        """SSE-delivered rate_limit_error (status_code=200) still classifies correctly."""
        exc = _exc_mid_stream("rate_limit_error", "Rate limited")
        result = classify_api_error(exc, "Anthropic")
        self.assertEqual(result.error_code, "rate_limited")
        self.assertIn("rate limiting", result.user_message)

    # -- Gemini (google-genai) errors: status on the chained cause -----------

    def test_gemini_resource_exhausted_maps_to_rate_limited(self):
        exc = _gemini_exc(
            code=429, status="RESOURCE_EXHAUSTED",
            message="Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): "
                    "429 RESOURCE_EXHAUSTED. {...}",
        )
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "rate_limited")
        self.assertEqual(result.log_level, "warning")
        self.assertTrue(_is_rate_limit_error(exc))
        self.assertTrue(_is_retryable_transient_error(exc))

    def test_gemini_unavailable_maps_to_overloaded_and_retryable(self):
        exc = _gemini_exc(code=503, status="UNAVAILABLE")
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "overloaded")
        self.assertTrue(_is_overloaded_error(exc))
        self.assertTrue(_is_retryable_transient_error(exc))

    def test_gemini_deadline_exceeded_message_only_maps_to_timeout(self):
        # No structured code/status — only the message carries the enum.
        exc = Exception(
            "Error calling model 'gemini-3.5-flash' (DEADLINE_EXCEEDED): "
            "504 DEADLINE_EXCEEDED."
        )
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "timeout")
        # Timeouts are not retried inside the provider (the whole sub-agent run
        # is retried at the Celery layer instead).
        self.assertFalse(_is_retryable_transient_error(exc))

    def test_gemini_internal_maps_to_server_error(self):
        exc = _gemini_exc(code=500, status="INTERNAL")
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "server_error")
        self.assertEqual(result.log_level, "error")

    def test_gemini_message_only_enum_maps_to_rate_limited(self):
        # Cause without a numeric code; only the enum string is present.
        exc = _gemini_exc(status="RESOURCE_EXHAUSTED", message="429 RESOURCE_EXHAUSTED. quota")
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "rate_limited")

    def test_gemini_permission_denied_maps_to_auth_error(self):
        exc = _gemini_exc(code=403, status="PERMISSION_DENIED")
        result = classify_api_error(exc, "Gemini")
        self.assertEqual(result.error_code, "auth_error")
        self.assertFalse(_is_retryable_transient_error(exc))

    def test_provider_label_appears_in_messages(self):
        """Provider label is included in user messages for most error codes."""
        for status, expected_code in [(429, "rate_limited"), (503, "overloaded"),
                                       (401, "auth_error"), (500, "server_error")]:
            result = classify_api_error(_exc_with_status(status), "TestProvider")
            self.assertIn("TestProvider", result.user_message,
                          f"Provider label missing for status {status}")
