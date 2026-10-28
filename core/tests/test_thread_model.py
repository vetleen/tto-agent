"""Tests for core.preferences.resolve_thread_model (per-thread model + fallback)."""

from django.test import SimpleTestCase

from core.preferences import ResolvedPreferences, resolve_thread_model
from llm.model_registry import get_models_by_tier


def _prefs(allowed, top, medium, cheap, chat=None):
    return ResolvedPreferences(
        top_model=top, medium_model=medium, cheap_model=cheap,
        allowed_models=list(allowed),
        feature_models=({"chat": chat} if chat else {}),
    )


class ResolveThreadModelTests(SimpleTestCase):
    def test_stored_allowed_is_honored(self):
        prefs = _prefs(["a", "b"], top="a", medium="b", cheap="c")
        self.assertEqual(resolve_thread_model("b", prefs), "b")

    def test_stored_thread_model_wins_over_org_chat_default(self):
        prefs = _prefs(
            ["org-default", "thread-choice"],
            top="org-default",
            medium="thread-choice",
            cheap="cheap",
            chat="org-default",
        )
        self.assertEqual(
            resolve_thread_model("thread-choice", prefs), "thread-choice"
        )

    def test_empty_uses_chat_feature_model(self):
        prefs = _prefs(["a", "b"], top="a", medium="b", cheap="c", chat="b")
        self.assertEqual(resolve_thread_model("", prefs), "b")

    def test_empty_without_chat_feature_falls_to_top(self):
        prefs = _prefs(["a"], top="a", medium="b", cheap="c")
        self.assertEqual(resolve_thread_model("", prefs), "a")

    def test_disallowed_medium_model_falls_back_to_medium(self):
        medium_model = get_models_by_tier("medium")[0]
        # medium_model is NOT in allowed → fall back by tier to prefs.medium_model.
        prefs = _prefs(
            ["keep-top", "keep-medium", "keep-cheap"],
            top="keep-top", medium="keep-medium", cheap="keep-cheap",
        )
        self.assertEqual(resolve_thread_model(medium_model, prefs), "keep-medium")

    def test_disallowed_standard_model_falls_back_to_top(self):
        std_model = get_models_by_tier("standard")[0]
        prefs = _prefs(
            ["keep-top"], top="keep-top", medium="keep-medium", cheap="keep-cheap",
        )
        self.assertEqual(resolve_thread_model(std_model, prefs), "keep-top")

    def test_unknown_disallowed_model_falls_back_to_chat_default(self):
        # Not in the registry → no tier → preferred chat model.
        prefs = _prefs(
            ["keep-top"], top="keep-top", medium="keep-medium", cheap="keep-cheap",
            chat="keep-top",
        )
        self.assertEqual(
            resolve_thread_model("vendor/totally-unknown", prefs), "keep-top",
        )
