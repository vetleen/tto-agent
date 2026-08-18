"""Resolution tests for the live_transcription_mode flag.

The mode is fixed at ``realtime_with_fallback`` for everyone — there is no
user- or org-facing choice (the settings picker was removed). Any stale
stored value at either layer is ignored.
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
         patch("agent_skills.services.get_available_skills", return_value=[]), \
         patch("agent_skills.services.get_subagent_skills", return_value=[]):
        return get_preferences(_MockUser())


class LiveTranscriptionModeCascadeTests(TestCase):
    def test_shipping_default_is_realtime_with_fallback(self):
        prefs = _prefs()
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")

    def test_stale_user_value_is_ignored(self):
        # The picker is gone; a leftover user preference no longer takes effect.
        prefs = _prefs(user={"live_transcription_mode": "chunked"})
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")

    def test_stale_org_value_is_ignored(self):
        prefs = _prefs(org={"live_transcription_mode": "chunked"})
        self.assertEqual(prefs.live_transcription_mode, "realtime_with_fallback")
