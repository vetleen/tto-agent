"""Tests for core.preferences.resolve_thread_model (per-thread model + fallback)."""

from django.test import SimpleTestCase

from core.preferences import ResolvedPreferences, resolve_thread_model
from llm.model_registry import get_models_by_tier


def _prefs(allowed, top, mid, cheap, chat=None):
    return ResolvedPreferences(
        top_model=top, mid_model=mid, cheap_model=cheap,
        allowed_models=list(allowed),
        feature_models=({"chat": chat} if chat else {}),
    )


class ResolveThreadModelTests(SimpleTestCase):
    def test_stored_allowed_is_honored(self):
        prefs = _prefs(["a", "b"], top="a", mid="b", cheap="c")
        self.assertEqual(resolve_thread_model("b", prefs), "b")

    def test_empty_uses_chat_feature_model(self):
        prefs = _prefs(["a", "b"], top="a", mid="b", cheap="c", chat="b")
        self.assertEqual(resolve_thread_model("", prefs), "b")

    def test_empty_without_chat_feature_falls_to_top(self):
        prefs = _prefs(["a"], top="a", mid="b", cheap="c")
        self.assertEqual(resolve_thread_model("", prefs), "a")

    def test_disallowed_mid_model_falls_back_to_mid(self):
        mid_model = get_models_by_tier("mid")[0]
        # mid_model is NOT in allowed → fall back by tier to prefs.mid_model.
        prefs = _prefs(
            ["keep-top", "keep-mid", "keep-cheap"],
            top="keep-top", mid="keep-mid", cheap="keep-cheap",
        )
        self.assertEqual(resolve_thread_model(mid_model, prefs), "keep-mid")

    def test_disallowed_standard_model_falls_back_to_top(self):
        std_model = get_models_by_tier("standard")[0]
        prefs = _prefs(
            ["keep-top"], top="keep-top", mid="keep-mid", cheap="keep-cheap",
        )
        self.assertEqual(resolve_thread_model(std_model, prefs), "keep-top")

    def test_unknown_disallowed_model_falls_back_to_chat_default(self):
        # Not in the registry → no tier → preferred chat model.
        prefs = _prefs(
            ["keep-top"], top="keep-top", mid="keep-mid", cheap="keep-cheap",
            chat="keep-top",
        )
        self.assertEqual(
            resolve_thread_model("vendor/totally-unknown", prefs), "keep-top",
        )
