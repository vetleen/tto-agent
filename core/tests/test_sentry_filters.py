"""Tests for core.sentry_filters.drop_duplicate_provider_error (WILFRED-A0 / WILFRED-AP)."""
from django.test import SimpleTestCase
from google.genai import errors as genai_errors

from core.sentry_filters import drop_duplicate_provider_error


def _event(mechanism: str | None) -> dict:
    value = {"type": "ClientError", "value": "boom"}
    if mechanism:
        value["mechanism"] = {"type": mechanism, "handled": False}
    return {"exception": {"values": [value]}}


def _hint(exc: BaseException) -> dict:
    return {"exc_info": (type(exc), exc, None)}


def _genai_429() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {"error": {"code": 429, "message": "Resource exhausted.", "status": "RESOURCE_EXHAUSTED"}},
    )


class DropDuplicateProviderErrorTests(SimpleTestCase):
    def test_drops_langchain_capture_of_genai_client_error(self):
        self.assertIsNone(drop_duplicate_provider_error(_event("langchain"), _hint(_genai_429())))

    def test_drops_when_provider_error_is_the_cause(self):
        try:
            try:
                raise _genai_429()
            except genai_errors.ClientError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            exc = outer
        self.assertIsNone(drop_duplicate_provider_error(_event("langchain"), _hint(exc)))

    def test_keeps_langchain_capture_of_non_provider_bug(self):
        event = _event("langchain")
        self.assertIs(drop_duplicate_provider_error(event, _hint(KeyError("x"))), event)

    def test_keeps_provider_error_not_from_langchain(self):
        # Our own code paths (or other integrations) still report provider errors.
        event = _event("logging")
        self.assertIs(drop_duplicate_provider_error(event, _hint(_genai_429())), event)

    def test_keeps_message_events_and_missing_hint(self):
        message_event = {"message": "LLM provider error error_code=rate_limited"}
        self.assertIs(drop_duplicate_provider_error(message_event, {}), message_event)
        event = _event("langchain")
        self.assertIs(drop_duplicate_provider_error(event, None), event)

    def test_none_event_passes_through(self):
        self.assertIsNone(drop_duplicate_provider_error(None, None))
