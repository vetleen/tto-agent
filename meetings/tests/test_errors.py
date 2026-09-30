"""Unit tests for the transcription error classifier."""
from __future__ import annotations

import logging

from django.test import SimpleTestCase

from meetings.services.errors import classify_transcription_error, log_unmapped


class _FakeStatusError(Exception):
    """Exception carrying a status_code attribute (mimics an HTTP client error)."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


# These deliberately reuse the exact class names the classifier sniffs for
# (the real OpenAI / orchestrator exceptions), since it matches on
# ``type(exc).__name__`` rather than importing the classes.
class RateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class AudioSplitTimeoutError(Exception):
    pass


class ClassifyTranscriptionErrorTests(SimpleTestCase):
    def test_rate_limited_by_class_name(self):
        c = classify_transcription_error(RateLimitError("slow down"))
        self.assertEqual(c.error_code, "rate_limited")
        self.assertEqual(c.log_level, "warning")
        self.assertIn("rate-limited", c.user_message.lower())

    def test_rate_limited_by_status(self):
        c = classify_transcription_error(_FakeStatusError("nope", 429))
        self.assertEqual(c.error_code, "rate_limited")

    def test_connection_error_by_builtin(self):
        c = classify_transcription_error(ConnectionError("reset"))
        self.assertEqual(c.error_code, "connection_error")
        self.assertEqual(c.log_level, "warning")

    def test_connection_error_by_class_name(self):
        c = classify_transcription_error(APIConnectionError("dns"))
        self.assertEqual(c.error_code, "connection_error")

    def test_undecodable_audio(self):
        c = classify_transcription_error(
            ValueError("Audio file is corrupted or unreadable: x.webm")
        )
        self.assertEqual(c.error_code, "undecodable_audio")
        self.assertEqual(c.log_level, "warning")

    def test_split_timeout_by_class_name(self):
        c = classify_transcription_error(AudioSplitTimeoutError("hung"))
        self.assertEqual(c.error_code, "split_timeout")

    def test_too_large(self):
        c = classify_transcription_error(
            RuntimeError("Audio file is 30 MB which exceeds the 25 MB API limit.")
        )
        self.assertEqual(c.error_code, "too_large")

    def test_catch_all_unknown(self):
        c = classify_transcription_error(RuntimeError("something weird"))
        self.assertEqual(c.error_code, "unknown")
        self.assertEqual(c.log_level, "error")
        self.assertIn("unexpectedly", c.user_message.lower())


class LogUnmappedTests(SimpleTestCase):
    def test_logs_error_with_exc_info_for_unknown(self):
        exc = RuntimeError("mystery")
        classified = classify_transcription_error(exc)
        with self.assertLogs("meetings.services.errors", level="ERROR") as cm:
            log_unmapped(classified, exc, context="unit_test")
        self.assertTrue(any(r.levelname == "ERROR" for r in cm.records))
        self.assertTrue(any(r.exc_info for r in cm.records))
        self.assertTrue(any("unit_test" in r.getMessage() for r in cm.records))

    def test_noop_for_mapped_error(self):
        exc = ConnectionError("reset")
        classified = classify_transcription_error(exc)
        logger = logging.getLogger("meetings.services.errors")
        # No ERROR record should be emitted for an already-mapped error.
        with self.assertNoLogs(logger, level="ERROR"):
            log_unmapped(classified, exc, context="unit_test")
