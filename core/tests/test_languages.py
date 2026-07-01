"""Unit tests for the shared transcription-language helpers."""
from __future__ import annotations

from django.test import SimpleTestCase

from core.languages import (
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    VALID_TRANSCRIPTION_LANGUAGE_VALUES,
    effective_meeting_language,
    is_valid_transcription_language,
    language_label,
    resolve_api_language,
)


class ResolveApiLanguageTests(SimpleTestCase):
    def test_blank_and_auto_map_to_none(self):
        self.assertIsNone(resolve_api_language(""))
        self.assertIsNone(resolve_api_language(None))
        self.assertIsNone(resolve_api_language("auto"))

    def test_concrete_code_passes_through(self):
        self.assertEqual(resolve_api_language("no"), "no")
        self.assertEqual(resolve_api_language("en"), "en")


class EffectiveMeetingLanguageTests(SimpleTestCase):
    def test_meeting_value_wins(self):
        self.assertEqual(effective_meeting_language("en", "no"), "en")

    def test_falls_back_to_default_when_meeting_blank(self):
        self.assertEqual(effective_meeting_language("", "no"), "no")
        self.assertEqual(effective_meeting_language("  ", "no"), "no")

    def test_meeting_auto_forces_detection_over_default(self):
        self.assertIsNone(effective_meeting_language("auto", "no"))

    def test_auto_default_is_none(self):
        self.assertIsNone(effective_meeting_language("", "auto"))

    def test_nothing_set_is_none(self):
        self.assertIsNone(effective_meeting_language("", ""))
        self.assertIsNone(effective_meeting_language(None, None))


class MiscHelperTests(SimpleTestCase):
    def test_default_is_auto(self):
        self.assertEqual(DEFAULT_TRANSCRIPTION_LANGUAGE, "auto")

    def test_validity(self):
        self.assertTrue(is_valid_transcription_language(""))
        self.assertTrue(is_valid_transcription_language("auto"))
        self.assertTrue(is_valid_transcription_language("no"))
        self.assertFalse(is_valid_transcription_language("klingon"))
        self.assertFalse(is_valid_transcription_language(None))

    def test_valid_values_include_backcompat_codes(self):
        self.assertIn("nb", VALID_TRANSCRIPTION_LANGUAGE_VALUES)
        self.assertIn("nn", VALID_TRANSCRIPTION_LANGUAGE_VALUES)

    def test_label(self):
        self.assertEqual(language_label("no"), "Norwegian")
        self.assertEqual(language_label("auto"), "Auto-detect")
        self.assertEqual(language_label(""), "Auto-detect")
        # Unknown code echoes back unchanged.
        self.assertEqual(language_label("xx"), "xx")
