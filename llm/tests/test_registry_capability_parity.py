"""Golden parity guard for the model-capability registry refactor.

Per-model capability knowledge (adaptive vs extended thinking, max-effort,
Responses-API, provider resolution, transcription chunking) was moved out of
scattered hardcoded sets/prefix lists into the registries. These tests assert
the registry-derived values still exactly match the *old* logic for every
registered model, so the refactor is provably behavior-preserving — and a new
model that omits a load-bearing flag (e.g. an adaptive-only Anthropic model
without thinking_mode) is caught here instead of 400-ing in production.
"""

from django.test import TestCase

from llm.core.model_factory import _parse_provider, _RESPONSES_API_PREFIXES
from llm.model_registry import _MODELS, get_model_info
from llm.transcription_registry import _TRANSCRIPTION_MODELS

# The exact hardcoded sets/rules this refactor replaced.
_OLD_ADAPTIVE = {"claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-5"}
_OLD_MAX_EFFORT = {"claude-opus-4-7", "claude-opus-4-8"}


class ModelRegistryCapabilityParityTests(TestCase):
    def test_thinking_mode_matches_old_adaptive_set(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                is_adaptive = info.thinking_mode == "adaptive"
                self.assertEqual(is_adaptive, info.api_model in _OLD_ADAPTIVE)

    def test_every_thinking_anthropic_model_declares_a_mode(self):
        # A thinking-capable Anthropic model with thinking_mode=None would route
        # through the extended budget_tokens path — which 400s for adaptive-only
        # models. Force an explicit choice.
        for model_id, info in _MODELS.items():
            if info.provider == "anthropic" and info.supports_thinking:
                with self.subTest(model=model_id):
                    self.assertIn(info.thinking_mode, ("adaptive", "extended"))

    def test_supports_max_effort_matches_old_set(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                self.assertEqual(info.supports_max_effort, info.api_model in _OLD_MAX_EFFORT)

    def test_uses_responses_api_matches_old_prefix_rule(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                old = info.provider == "openai" and any(
                    info.api_model.startswith(p) for p in _RESPONSES_API_PREFIXES
                )
                self.assertEqual(info.uses_responses_api, old)

    def test_parse_provider_matches_registry_for_every_model(self):
        for model_id, info in _MODELS.items():
            with self.subTest(model=model_id):
                self.assertEqual(_parse_provider(model_id), (info.provider, info.api_model))


class TranscriptionCapabilityParityTests(TestCase):
    def test_supports_chunking_strategy_matches_gpt4o_rule(self):
        for model_id, info in _TRANSCRIPTION_MODELS.items():
            with self.subTest(model=model_id):
                self.assertEqual(
                    info.supports_chunking_strategy,
                    info.api_model.startswith("gpt-4o"),
                )
