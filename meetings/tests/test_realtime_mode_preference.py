"""Preference cascade tests for the live_transcription_mode flag.

The system default is hardcoded to ``realtime_with_fallback`` — there is
no env var to override it and no org-level preference. Users can change
their own path via the settings page. Invalid values fall back to the
shipping default.
"""
from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from core.preferences import get_preferences


class _MockUser:
    id = 1
    is_anonymous = False


def _prefs(*, org=None, user=None):
    org_dict = org if org is not None else {}
    user_dict = user if user is not None else {}
    with patch("core.preferences._get_org_preferences", return_value=org_dict), \
         patch("core.preferences._get_user_preferences", return_value=user_dict), \
         patch("llm.service.policies.get_allowed_models", return_value=[]), \
         patch("agent_skills.services.get_available_skills", return_value=[]):
        return get_preferences(_MockUser())


class LiveTranscriptionModeCascadeTests(TestCase):
    def test_shipping_default_is_realtime_with_fallback(self):
        prefs = _prefs()
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")

    def test_user_override_wins(self):
        prefs = _prefs(user={"live_transcription_mode": "chunked"})
        self.assertEqual(prefs.live_transcription_mode, "chunked")

    def test_user_override_realtime_only(self):
        prefs = _prefs(user={"live_transcription_mode": "realtime"})
        self.assertEqual(prefs.live_transcription_mode, "realtime")

    def test_invalid_user_value_falls_back_to_shipping_default(self):
        prefs = _prefs(user={"live_transcription_mode": "teleportation"})
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")

    def test_org_preference_ignored(self):
        # There is no org-level override. If an org has an old value in
        # preferences it's silently ignored (shipping default applies).
        prefs = _prefs(org={"live_transcription_mode": "chunked"})
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")
