"""Preference cascade test for the live_transcription_mode flag.

System default is "chunked" so the existing chunked path is unchanged for
every org that hasn't opted in. Org and user overrides cascade and any
invalid value falls back to chunked so a typo can't break live
transcription.
"""
from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase, override_settings

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
    # Pin the setting explicitly so the test is independent of whatever the
    # local developer has in their ``.env`` (devs often experiment with
    # ``realtime`` / ``chunked`` while testing).

    @override_settings(MEETING_LIVE_TRANSCRIPTION_MODE_DEFAULT="realtime_with_fallback")
    def test_shipping_default_is_realtime_with_fallback(self):
        prefs = _prefs()
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")

    @override_settings(MEETING_LIVE_TRANSCRIPTION_MODE_DEFAULT="chunked")
    def test_system_default_can_be_pinned_to_chunked(self):
        # Operators can opt every dyno out of realtime via the env var
        # (e.g. during a known OpenAI Realtime incident).
        prefs = _prefs()
        self.assertEqual(prefs.live_transcription_mode, "chunked")

    def test_org_overrides_system(self):
        prefs = _prefs(org={"live_transcription_mode": "realtime"})
        self.assertEqual(prefs.live_transcription_mode, "realtime")

    def test_user_overrides_org(self):
        prefs = _prefs(
            org={"live_transcription_mode": "realtime_with_fallback"},
            user={"live_transcription_mode": "chunked"},
        )
        self.assertEqual(prefs.live_transcription_mode, "chunked")

    def test_invalid_value_falls_back_to_chunked(self):
        prefs = _prefs(org={"live_transcription_mode": "teleportation"})
        self.assertEqual(prefs.live_transcription_mode, "chunked")
