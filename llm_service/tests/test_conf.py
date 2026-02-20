"""Tests for llm_service.conf."""
from unittest import mock

from django.test import TestCase, override_settings


class ConfTestCase(TestCase):
    """Test configuration getters and model allow-list."""

    @override_settings(
        LLM_DEFAULT_MODEL="openai/gpt-4o-mini",
        LLM_ALLOWED_MODELS=["openai/gpt-4o", "openai/gpt-4o-mini"],
        LLM_REQUEST_TIMEOUT=30.0,
        LLM_MAX_RETRIES=3,
        LLM_LOG_WRITE_TIMEOUT=10.0,
    )
    def test_get_default_model(self):
        from llm_service.conf import get_default_model
        self.assertEqual(get_default_model(), "openai/gpt-4o-mini")

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o", "anthropic/claude-3"])
    def test_get_allowed_models(self):
        from llm_service.conf import get_allowed_models
        self.assertEqual(get_allowed_models(), ["openai/gpt-4o", "anthropic/claude-3"])

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    def test_is_model_allowed_when_list_set(self):
        from llm_service.conf import is_model_allowed
        self.assertTrue(is_model_allowed("openai/gpt-4o"))
        self.assertFalse(is_model_allowed("anthropic/claude-3"))
        self.assertFalse(is_model_allowed(None))

    @override_settings(LLM_ALLOWED_MODELS=[])
    def test_is_model_allowed_empty_means_allow_any(self):
        from llm_service.conf import is_model_allowed
        self.assertTrue(is_model_allowed("openai/gpt-4o"))
        self.assertTrue(is_model_allowed("any/model"))

    def test_get_request_timeout(self):
        from llm_service.conf import get_request_timeout
        with override_settings(LLM_REQUEST_TIMEOUT=45.0):
            self.assertEqual(get_request_timeout(), 45.0)

    def test_get_max_retries(self):
        from llm_service.conf import get_max_retries
        with override_settings(LLM_MAX_RETRIES=5):
            self.assertEqual(get_max_retries(), 5)

    def test_get_pre_call_hooks_default_empty(self):
        from llm_service.conf import get_pre_call_hooks
        self.assertEqual(get_pre_call_hooks(), [])

    @override_settings(LLM_PRE_CALL_HOOKS=[lambda req: None])
    def test_get_pre_call_hooks_from_settings(self):
        from llm_service.conf import get_pre_call_hooks
        hooks = get_pre_call_hooks()
        self.assertEqual(len(hooks), 1)
