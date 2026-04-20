"""Tests for LLM policies (resolve_model, get_allowed_models)."""

import os
from unittest.mock import patch

from django.test import TestCase

from llm.service.errors import LLMConfigurationError, LLMPolicyDenied
from llm.service.policies import (
    get_allowed_models,
    get_env_unregistered_models,
    resolve_model,
)


class PoliciesTests(TestCase):
    """Test model resolution and allowed-list behavior.

    Model IDs used here must exist in ``llm.model_registry`` — ``get_allowed_models``
    cross-checks the env against the registry and drops anything unknown.
    """

    def test_resolve_model_empty_allowed_raises_configuration_error(self):
        with patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "", "DEFAULT_LLM_MODEL": ""}, clear=False):
            with self.assertRaises(LLMConfigurationError) as ctx:
                resolve_model(None)
            self.assertIn("LLM_ALLOWED_MODELS", str(ctx.exception))

    def test_resolve_model_only_unregistered_models_raises_configuration_error(self):
        """Env models that aren't in the registry are dropped, leaving an empty list."""
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "not-a-model,also-not-a-model"},
            clear=False,
        ):
            with self.assertRaises(LLMConfigurationError):
                resolve_model(None)

    def test_resolve_model_requested_not_allowed_raises_policy_denied(self):
        with patch.dict(
            os.environ,
            {
                "LLM_ALLOWED_MODELS": "openai/gpt-5.4,anthropic/claude-sonnet-4-6",
                "DEFAULT_LLM_MODEL": "openai/gpt-5.4",
            },
            clear=False,
        ):
            with self.assertRaises(LLMPolicyDenied) as ctx:
                resolve_model("gemini/gemini-2.5-pro")
            self.assertIn("gemini/gemini-2.5-pro", str(ctx.exception))
            self.assertIn("not in LLM_ALLOWED_MODELS", str(ctx.exception))

    def test_resolve_model_requested_allowed_returns_requested(self):
        with patch.dict(
            os.environ,
            {
                "LLM_ALLOWED_MODELS": "openai/gpt-5.4,anthropic/claude-sonnet-4-6",
                "DEFAULT_LLM_MODEL": "anthropic/claude-sonnet-4-6",
            },
            clear=False,
        ):
            self.assertEqual(resolve_model("openai/gpt-5.4"), "openai/gpt-5.4")
            self.assertEqual(
                resolve_model("anthropic/claude-sonnet-4-6"),
                "anthropic/claude-sonnet-4-6",
            )

    def test_resolve_model_none_uses_default_when_allowed(self):
        with patch.dict(
            os.environ,
            {
                "LLM_ALLOWED_MODELS": "openai/gpt-5.4,anthropic/claude-sonnet-4-6",
                "DEFAULT_LLM_MODEL": "anthropic/claude-sonnet-4-6",
            },
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "anthropic/claude-sonnet-4-6")

    def test_resolve_model_none_default_not_in_allowed_uses_first_allowed(self):
        with patch.dict(
            os.environ,
            {
                "LLM_ALLOWED_MODELS": "openai/gpt-5.4,anthropic/claude-sonnet-4-6",
                "DEFAULT_LLM_MODEL": "gemini/gemini-2.5-pro",
            },
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "openai/gpt-5.4")

    def test_resolve_model_none_no_default_uses_first_allowed(self):
        with patch.dict(
            os.environ,
            {
                "LLM_ALLOWED_MODELS": "anthropic/claude-sonnet-4-6,openai/gpt-5.4",
                "DEFAULT_LLM_MODEL": "",
                "LLM_DEFAULT_MODEL": "",
            },
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "anthropic/claude-sonnet-4-6")

    def test_get_allowed_models_drops_unregistered_entries(self):
        """Env entries without a registry hit never reach callers."""
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "openai/gpt-5.4,bogus-model,anthropic/claude-sonnet-4-6"},
            clear=False,
        ):
            self.assertEqual(
                get_allowed_models(),
                ["openai/gpt-5.4", "anthropic/claude-sonnet-4-6"],
            )

    def test_get_env_unregistered_models_lists_dropped_entries(self):
        """Surfaces the dropped entries so the startup check can log them."""
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "openai/gpt-5.4,bogus-model,another-bogus"},
            clear=False,
        ):
            self.assertEqual(
                get_env_unregistered_models(),
                ["bogus-model", "another-bogus"],
            )
