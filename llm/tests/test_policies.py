"""Tests for LLM policies (resolve_model, get_allowed_models)."""

import os
from unittest.mock import patch

from django.test import TestCase

from llm.service.errors import LLMConfigurationError, LLMPolicyDenied
from llm.service.policies import get_allowed_models, resolve_model


class PoliciesTests(TestCase):
    """Test model resolution and allowed-list behavior."""

    def test_resolve_model_empty_allowed_raises_configuration_error(self):
        with patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "", "DEFAULT_LLM_MODEL": ""}, clear=False):
            with self.assertRaises(LLMConfigurationError) as ctx:
                resolve_model(None)
            self.assertIn("LLM_ALLOWED_MODELS", str(ctx.exception))

    def test_resolve_model_requested_not_allowed_raises_policy_denied(self):
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "gpt-4o,claude-3-5-sonnet", "DEFAULT_LLM_MODEL": "gpt-4o"},
            clear=False,
        ):
            with self.assertRaises(LLMPolicyDenied) as ctx:
                resolve_model("gemini-1")
            self.assertIn("gemini-1", str(ctx.exception))
            self.assertIn("not in LLM_ALLOWED_MODELS", str(ctx.exception))

    def test_resolve_model_requested_allowed_returns_requested(self):
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "gpt-4o,claude-3-5-sonnet", "DEFAULT_LLM_MODEL": "claude-3-5-sonnet"},
            clear=False,
        ):
            self.assertEqual(resolve_model("gpt-4o"), "gpt-4o")
            self.assertEqual(resolve_model("claude-3-5-sonnet"), "claude-3-5-sonnet")

    def test_resolve_model_none_uses_default_when_allowed(self):
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "gpt-4o,claude-3-5-sonnet", "DEFAULT_LLM_MODEL": "claude-3-5-sonnet"},
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "claude-3-5-sonnet")

    def test_resolve_model_none_default_not_in_allowed_uses_first_allowed(self):
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "gpt-4o,claude-3-5-sonnet", "DEFAULT_LLM_MODEL": "gemini-1"},
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "gpt-4o")

    def test_resolve_model_none_no_default_uses_first_allowed(self):
        with patch.dict(
            os.environ,
            {"LLM_ALLOWED_MODELS": "claude-3-5-sonnet,gpt-4o", "DEFAULT_LLM_MODEL": ""},
            clear=False,
        ):
            self.assertEqual(resolve_model(None), "claude-3-5-sonnet")

    def test_get_allowed_models_returns_list(self):
        with patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "a,b,c"}, clear=False):
            self.assertEqual(get_allowed_models(), ["a", "b", "c"])
