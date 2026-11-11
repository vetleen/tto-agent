"""Protocol validation for model-specific reasoning controls."""

from django.test import SimpleTestCase

from chat.consumers import _is_retired_model_id, _resolve_reasoning_level


class ResolveReasoningLevelTests(SimpleTestCase):
    def test_missing_value_uses_model_default(self):
        self.assertEqual(_resolve_reasoning_level("openai/gpt-6.1-terra", None), "medium")
        self.assertEqual(_resolve_reasoning_level("anthropic/claude-fable-5-2", None), "high")
        self.assertEqual(_resolve_reasoning_level("gemini/gemini-3.5-flash-lite", None), "minimal")

    def test_supported_value_is_preserved(self):
        self.assertEqual(_resolve_reasoning_level("openai/gpt-6.1-sol", "max"), "max")
        self.assertEqual(_resolve_reasoning_level("anthropic/claude-sonnet-5-5", "xhigh"), "xhigh")
        self.assertEqual(_resolve_reasoning_level("gemini/gemini-3.7-flash", "low"), "low")

    def test_invalid_value_is_rejected_instead_of_clamped(self):
        with self.assertRaisesMessage(ValueError, "not available"):
            _resolve_reasoning_level("gemini/gemini-3.7-flash", "max")
        with self.assertRaisesMessage(ValueError, "not available"):
            _resolve_reasoning_level("anthropic/claude-fable-5-2", "off")

    def test_a_retired_models_level_the_successor_lacks_is_rejected_here(self):
        # The chat turn catches this for a retired requested ID and falls
        # back to the successor's default (see _is_retired_model_id).
        with self.assertRaisesMessage(ValueError, "Claude Sonnet 5.5"):
            _resolve_reasoning_level("anthropic/claude-sonnet-5", "off")
        with self.assertRaisesMessage(ValueError, "GPT-6.1 Sol"):
            _resolve_reasoning_level("openai/gpt-6-sol", "none")
        self.assertEqual(_resolve_reasoning_level("anthropic/claude-sonnet-5", None), "high")


class RetiredModelIdTests(SimpleTestCase):
    def test_retired_ids_are_recognised_with_or_without_prefix(self):
        for model_id in (
            "anthropic/claude-sonnet-5", "claude-sonnet-5", "openai/gpt-6-sol",
            "openai/gpt-5.6-terra", "anthropic/claude-fable-5", "anthropic/claude-fable-5-1",
        ):
            with self.subTest(model=model_id):
                self.assertTrue(_is_retired_model_id(model_id))

    def test_live_unknown_and_empty_ids_are_not_retired(self):
        for model_id in (
            "anthropic/claude-sonnet-5-5", "claude-sonnet-5-5", "gemini/gemini-4-carbon",
            "bogus/model", "", None, 7,
        ):
            with self.subTest(model=model_id):
                self.assertFalse(_is_retired_model_id(model_id))
