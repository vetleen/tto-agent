"""Provider errors must reach Sentry despite the LangChain integration + Dedupe.

Sentry's LangChain integration captures the provider exception before our
classified log lines run; DedupeIntegration then drops every later event
carrying the same exception object, so ``logger.error(..., exc_info=True)``
never reached Sentry (2026-10-19 Gemini 400s). ``_report_to_sentry`` emits an
``exc_info``-less, fingerprinted event that Dedupe lets through.
"""

from __future__ import annotations

import logging
from unittest import TestCase
from unittest.mock import MagicMock, patch

import sentry_sdk
from sentry_sdk.integrations.dedupe import DedupeIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.transport import Transport

from llm.core.providers.base import BaseLangChainChatModel
from llm.types.messages import Message
from llm.types.requests import ChatRequest

_events: list[dict] = []


class _CaptureTransport(Transport):
    def capture_envelope(self, envelope):
        for item in envelope.items:
            if item.type == "event":
                _events.append(item.payload.json)


def _make_model():
    model = BaseLangChainChatModel(model_name="test-model", client=MagicMock())
    model._provider_label = "Test"
    model._client.bind_tools = MagicMock(return_value=model._client)
    return model


def _make_request():
    return ChatRequest(messages=[Message(role="user", content="hi")], model="test-model")


def _captured_first(exc):
    """Mimic the LangChain integration: capture the exception, then raise it."""
    def _raise(*args, **kwargs):
        sentry_sdk.capture_exception(exc)
        raise exc
    return _raise


def _message(event) -> str:
    entry = event.get("logentry") or {}
    return entry.get("formatted") or entry.get("message") or ""


@patch("llm.core.providers.base.time.sleep", return_value=None)
class ReportToSentryTests(TestCase):

    def setUp(self):
        self._previous_client = sentry_sdk.get_client()
        _events.clear()
        sentry_sdk.init(
            dsn="https://key@o0.ingest.sentry.io/0",
            transport=_CaptureTransport,
            default_integrations=False,
            integrations=[
                DedupeIntegration(),
                LoggingIntegration(level=logging.INFO, event_level=logging.WARNING),
            ],
        )

    def tearDown(self):
        sentry_sdk.flush()
        sentry_sdk.get_global_scope().set_client(self._previous_client)

    def _ours(self):
        sentry_sdk.flush()
        return [e for e in _events if (e.get("fingerprint") or [None])[0] == "llm-provider-error"]

    def test_unmapped_stream_error_reaches_sentry_after_dedupe(self, _sleep):
        exc = RuntimeError("something new")
        model = _make_model()
        model._client.stream = MagicMock(side_effect=_captured_first(exc))

        list(model.stream(_make_request()))

        ours = self._ours()
        self.assertEqual(len(ours), 1)
        self.assertIn("Unmapped LLM provider error", _message(ours[0]))
        self.assertEqual(ours[0]["fingerprint"], ["llm-provider-error", "Test", "unknown", "RuntimeError"])
        self.assertEqual(ours[0]["tags"]["llm.error_code"], "unknown")
        # The traceback-carrying "LLM stream error" line was deduped, as in prod.
        self.assertFalse(any("LLM stream error" in _message(e) for e in _events))

    def test_gemini_rejected_reaches_sentry_after_retry(self, _sleep):
        exc = Exception("400 INVALID_ARGUMENT. Request contains an invalid argument.")
        exc.code = 400
        model = _make_model()
        model._client.stream = MagicMock(side_effect=_captured_first(exc))

        list(model.stream(_make_request()))

        ours = self._ours()
        self.assertEqual(len(ours), 1)
        self.assertEqual(ours[0]["tags"]["llm.error_code"], "provider_rejected_request")
        self.assertIn("detail=400 INVALID_ARGUMENT", _message(ours[0]))

    def test_generate_error_reaches_sentry(self, _sleep):
        exc = Exception("server exploded")
        exc.status_code = 500
        model = _make_model()
        model._client.invoke = MagicMock(side_effect=_captured_first(exc))

        with self.assertRaises(Exception):
            model.generate(_make_request())

        ours = self._ours()
        self.assertEqual(len(ours), 1)
        self.assertEqual(ours[0]["tags"]["llm.error_code"], "server_error")

    def test_warning_level_error_not_reported_until_exhausted(self, _sleep):
        """A rate limit that later succeeds must not page anyone."""
        exc = Exception("rate limited")
        exc.status_code = 429
        chunk = MagicMock()
        chunk.content = "hello"
        chunk.usage_metadata = {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        chunk.response_metadata = {}
        model = _make_model()
        model._client.stream = MagicMock(side_effect=[exc, [chunk]])

        list(model.stream(_make_request()))

        self.assertEqual(self._ours(), [])

    def test_rate_limit_exhausted_is_reported(self, _sleep):
        exc = Exception("rate limited")
        exc.status_code = 429
        model = _make_model()
        model._client.stream = MagicMock(side_effect=_captured_first(exc))

        list(model.stream(_make_request()))

        ours = self._ours()
        self.assertEqual(len(ours), 1)
        self.assertEqual(ours[0]["tags"]["llm.error_code"], "rate_limited")
