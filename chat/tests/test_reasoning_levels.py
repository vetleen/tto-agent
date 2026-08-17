"""Protocol validation for model-specific reasoning controls."""

from django.test import SimpleTestCase

from chat.consumers import _resolve_reasoning_level


class ResolveReasoningLevelTests(SimpleTestCase):
    def test_missing_value_uses_model_default(self):
        self.assertEqual(_resolve_reasoning_level("openai/gpt-5.6-terra", None), "medium")
        self.assertEqual(_resolve_reasoning_level("anthropic/claude-fable-5", None), "high")
        self.assertEqual(_resolve_reasoning_level("gemini/gemini-3.5-flash-lite", None), "minimal")

    def test_supported_value_is_preserved(self):
        self.assertEqual(_resolve_reasoning_level("openai/gpt-5.6-sol", "max"), "max")
        self.assertEqual(_resolve_reasoning_level("anthropic/claude-sonnet-5", "xhigh"), "xhigh")
        self.assertEqual(_resolve_reasoning_level("gemini/gemini-3.7-flash", "low"), "low")

    def test_invalid_value_is_rejected_instead_of_clamped(self):
        with self.assertRaisesMessage(ValueError, "not available"):
            _resolve_reasoning_level("gemini/gemini-3.7-flash", "max")
        with self.assertRaisesMessage(ValueError, "not available"):
            _resolve_reasoning_level("anthropic/claude-fable-5", "off")
