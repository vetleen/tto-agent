"""Capability consistency checks across registries and provider routing."""

from django.test import SimpleTestCase

from llm.core.model_factory import _RESPONSES_API_PREFIXES, _parse_provider
from llm.model_registry import _MODELS
from llm.transcription_registry import _TRANSCRIPTION_MODELS

_ADAPTIVE_ANTHROPIC = {
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-sonnet-5",
}


class ModelRegistryCapabilityTests(SimpleTestCase):
    def test_anthropic_reasoning_transport_is_correct(self):
        for model_id, info in _MODELS.items():
            if info.provider != "anthropic":
                continue
            with self.subTest(model=model_id):
                expected = "adaptive" if info.api_model in _ADAPTIVE_ANTHROPIC else "extended"
                self.assertEqual(info.thinking_mode, expected)

    def test_reasoning_defaults_are_valid(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                if info.supports_thinking:
                    self.assertIn(info.default_reasoning_level, info.reasoning_levels)

    def test_responses_api_flags_match_openai_family(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                expected = info.provider == "openai" and any(
                    info.api_model.startswith(prefix) for prefix in _RESPONSES_API_PREFIXES
                )
                self.assertEqual(info.uses_responses_api, expected)

    def test_parse_provider_matches_registry(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                self.assertEqual(_parse_provider(model_id), (info.provider, info.api_model))


class TranscriptionCapabilityTests(SimpleTestCase):
    def test_chunking_strategy_matches_gpt4o_rule(self):
        for model_id, info in _TRANSCRIPTION_MODELS.items():
            with self.subTest(model=model_id):
                self.assertEqual(
                    info.supports_chunking_strategy,
                    info.api_model.startswith("gpt-4o"),
                )
