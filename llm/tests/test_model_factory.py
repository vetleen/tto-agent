"""Tests for the model factory (provider detection, wrapper selection, rate limiter)."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from llm.core.model_factory import (
    _get_rate_limiter,
    _parse_provider,
    _rate_limiters,
    detect_provider,
)
from llm.service.errors import LLMConfigurationError


class ParseProviderTests(SimpleTestCase):
    """Test _parse_provider provider detection and prefix stripping."""

    def test_explicit_openai_prefix(self):
        provider, api_model = _parse_provider("openai/gpt-5-mini")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "gpt-5-mini")

    def test_explicit_anthropic_prefix(self):
        provider, api_model = _parse_provider("anthropic/claude-sonnet-4-6")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(api_model, "claude-sonnet-4-6")

    def test_explicit_gemini_prefix(self):
        provider, api_model = _parse_provider("gemini/gemini-2.5-flash")
        self.assertEqual(provider, "google_genai")
        self.assertEqual(api_model, "gemini-2.5-flash")

    def test_auto_detect_gpt(self):
        provider, api_model = _parse_provider("gpt-5-mini")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "gpt-5-mini")

    def test_auto_detect_o1(self):
        provider, api_model = _parse_provider("o1-preview")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "o1-preview")

    def test_auto_detect_o3(self):
        provider, api_model = _parse_provider("o3-mini")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "o3-mini")

    def test_auto_detect_o4(self):
        provider, api_model = _parse_provider("o4-mini")
        self.assertEqual(provider, "openai")
        self.assertEqual(api_model, "o4-mini")

    def test_auto_detect_claude(self):
        provider, api_model = _parse_provider("claude-opus-4-6")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(api_model, "claude-opus-4-6")

    def test_auto_detect_gemini(self):
        provider, api_model = _parse_provider("gemini-2.5-pro")
        self.assertEqual(provider, "google_genai")
        self.assertEqual(api_model, "gemini-2.5-pro")

    def test_unknown_model_raises(self):
        with self.assertRaises(LLMConfigurationError) as ctx:
            _parse_provider("llama-3-70b")
        self.assertIn("llama-3-70b", str(ctx.exception))
        self.assertIn("Cannot determine provider", str(ctx.exception))


class DetectProviderTests(SimpleTestCase):
    """detect_provider must handle prefix-less model names without raising —
    block-building for multimodal messages relies on this."""

    def test_explicit_prefix(self):
        self.assertEqual(detect_provider("anthropic/claude-sonnet-4-6"), "anthropic")
        self.assertEqual(detect_provider("openai/gpt-5-mini"), "openai")
        self.assertEqual(detect_provider("gemini/gemini-2.5-flash"), "google_genai")

    def test_prefix_less_claude(self):
        self.assertEqual(detect_provider("claude-sonnet-4-6"), "anthropic")

    def test_prefix_less_gpt(self):
        self.assertEqual(detect_provider("gpt-5-mini"), "openai")

    def test_prefix_less_gemini(self):
        self.assertEqual(detect_provider("gemini-2.5-pro"), "google_genai")

    def test_empty_and_none(self):
        self.assertEqual(detect_provider(""), "")
        self.assertEqual(detect_provider(None), "")

    def test_unknown_model_returns_empty(self):
        """Unknown model must not raise — callers fall through to a safe default."""
        self.assertEqual(detect_provider("llama-3-70b"), "")


class CreateChatModelTests(SimpleTestCase):
    """Test create_chat_model wrapper class selection."""

    @patch("llm.core.model_factory.init_chat_model")
    def test_openai_model_uses_openai_wrapper(self, mock_init):
        from llm.core.model_factory import create_chat_model
        from llm.core.providers.openai import OpenAIChatModel

        mock_client = MagicMock()
        mock_init.return_value = mock_client

        model = create_chat_model("gpt-5-mini")
        self.assertIsInstance(model, OpenAIChatModel)
        self.assertEqual(model.name, "gpt-5-mini")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args
        self.assertEqual(call_kwargs[0][0], "gpt-5-mini")
        self.assertEqual(call_kwargs[1]["model_provider"], "openai")
        self.assertEqual(call_kwargs[1]["max_retries"], 3)
        self.assertEqual(call_kwargs[1]["stream_usage"], True)

    @patch("llm.core.model_factory.init_chat_model")
    def test_anthropic_model_uses_anthropic_wrapper(self, mock_init):
        from llm.core.model_factory import create_chat_model
        from llm.core.providers.anthropic import AnthropicChatModel

        mock_client = MagicMock()
        mock_init.return_value = mock_client

        model = create_chat_model("claude-sonnet-4-6")
        self.assertIsInstance(model, AnthropicChatModel)
        self.assertEqual(model.name, "claude-sonnet-4-6")

    @patch("llm.core.model_factory.init_chat_model")
    def test_gemini_model_uses_gemini_wrapper(self, mock_init):
        from llm.core.model_factory import create_chat_model
        from llm.core.providers.gemini import GeminiChatModel

        mock_client = MagicMock()
        mock_init.return_value = mock_client

        model = create_chat_model("gemini-2.5-pro")
        self.assertIsInstance(model, GeminiChatModel)
        self.assertEqual(model.name, "gemini-2.5-pro")
        call_kwargs = mock_init.call_args
        self.assertNotIn("stream_usage", call_kwargs[1])

    @patch("llm.core.model_factory.init_chat_model")
    def test_explicit_prefix_stripped_for_api(self, mock_init):
        from llm.core.model_factory import create_chat_model

        mock_client = MagicMock()
        mock_init.return_value = mock_client

        create_chat_model("openai/gpt-5-mini")
        call_args = mock_init.call_args
        # The api_model passed to init_chat_model should be stripped
        self.assertEqual(call_args[0][0], "gpt-5-mini")

    @patch("llm.core.model_factory.init_chat_model")
    def test_responses_api_enabled_for_gpt54(self, mock_init):
        from llm.core.model_factory import create_chat_model

        mock_client = MagicMock()
        mock_init.return_value = mock_client

        create_chat_model("gpt-5.4")
        call_kwargs = mock_init.call_args[1]
        self.assertTrue(call_kwargs.get("use_responses_api"))

    def test_unknown_model_raises_configuration_error(self):
        from llm.core.model_factory import create_chat_model

        with self.assertRaises(LLMConfigurationError):
            create_chat_model("unknown-model-xyz")


class RateLimiterTests(SimpleTestCase):
    """Test rate limiter singleton behavior."""

    def setUp(self):
        _rate_limiters.clear()

    def tearDown(self):
        _rate_limiters.clear()

    @patch.dict("os.environ", {"LLM_RATE_LIMIT_OPENAI_RPS": "10"})
    def test_rate_limiter_created_when_env_set(self):
        limiter = _get_rate_limiter("openai")
        self.assertIsNotNone(limiter)

    @patch.dict("os.environ", {"LLM_RATE_LIMIT_OPENAI_RPS": "10"})
    def test_rate_limiter_reused(self):
        limiter1 = _get_rate_limiter("openai")
        limiter2 = _get_rate_limiter("openai")
        self.assertIs(limiter1, limiter2)

    def test_rate_limiter_none_when_no_env(self):
        limiter = _get_rate_limiter("openai")
        self.assertIsNone(limiter)
