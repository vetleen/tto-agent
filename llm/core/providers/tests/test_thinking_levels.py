"""Tests for provider thinking level support."""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from llm.core.providers.anthropic import AnthropicChatModel, _ANTHROPIC_THINKING
from llm.core.providers.gemini import GeminiChatModel, _GEMINI_THINKING_BUDGETS
from llm.core.providers.openai import OpenAIChatModel
from llm.types.requests import ChatRequest


def _make_request(thinking_level="off", tools=None):
    return ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="test",
        stream=True,
        params={"thinking_level": thinking_level},
        tools=tools or [],
    )


class AnthropicThinkingLevelTests(TestCase):

    def _make_model(self):
        client = MagicMock()
        return AnthropicChatModel("anthropic/claude-sonnet-4-6", client)

    def test_off_returns_standard_client(self):
        model = self._make_model()
        request = _make_request("off")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_low_creates_variant_with_4k_budget(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model()
        request = _make_request("low")
        model._get_streaming_client(request)
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["thinking"]["budget_tokens"], 4_096)
        self.assertEqual(call_kwargs["max_tokens"], 16_384)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_medium_creates_variant_with_10k_budget(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model()
        request = _make_request("medium")
        model._get_streaming_client(request)
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["thinking"]["budget_tokens"], 10_000)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_high_creates_variant_with_32k_budget(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model()
        request = _make_request("high")
        model._get_streaming_client(request)
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["thinking"]["budget_tokens"], 32_000)
        self.assertEqual(call_kwargs["max_tokens"], 40_000)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_fallback_on_create_failure(self, mock_create):
        mock_create.side_effect = Exception("API error")
        model = self._make_model()
        request = _make_request("high")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)


class OpenAIThinkingLevelTests(TestCase):

    def _make_model(self, name="openai/gpt-5.4"):
        client = MagicMock()
        return OpenAIChatModel(name, client)

    def test_off_returns_standard_client(self):
        model = self._make_model()
        request = _make_request("off")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)

    @patch("llm.core.providers.openai.create_variant_client")
    def test_gpt54_supports_reasoning(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model("openai/gpt-5.4")
        request = _make_request("medium")
        model._get_streaming_client(request)
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["model_kwargs"]["reasoning_effort"], "medium")

    @patch("llm.core.providers.openai.create_variant_client")
    def test_o3_supports_reasoning(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model("openai/o3")
        request = _make_request("high")
        model._get_streaming_client(request)
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["model_kwargs"]["reasoning_effort"], "high")

    def test_non_reasoning_model_ignores_level(self):
        model = self._make_model("openai/gpt-5-mini")
        request = _make_request("high")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)

    @patch("llm.core.providers.openai.create_variant_client")
    def test_level_maps_directly_to_reasoning_effort(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model("openai/gpt-5.4")
        for level in ("low", "medium", "high"):
            mock_create.reset_mock()
            request = _make_request(level)
            model._get_streaming_client(request)
            call_kwargs = mock_create.call_args[1]
            self.assertEqual(call_kwargs["model_kwargs"]["reasoning_effort"], level)


class GeminiThinkingLevelTests(TestCase):

    def _make_model(self, name="gemini/gemini-3-flash-preview"):
        client = MagicMock()
        return GeminiChatModel(name, client)

    def test_off_returns_standard_client(self):
        model = self._make_model()
        request = _make_request("off")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_thinking_enabled_creates_variant(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model("gemini/gemini-3-flash-preview")
        request = _make_request("medium")
        model._get_streaming_client(request)
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["thinking_budget"], 8_192)
        self.assertTrue(call_kwargs["include_thoughts"])

    def test_non_thinking_model_ignores_level(self):
        model = self._make_model("gemini/gemini-2.5-flash")
        request = _make_request("high")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_high_uses_correct_budget(self, mock_create):
        mock_create.return_value = MagicMock()
        model = self._make_model("gemini/gemini-3.1-pro-preview")
        request = _make_request("high")
        model._get_streaming_client(request)
        call_kwargs = mock_create.call_args[1]
        self.assertEqual(call_kwargs["thinking_budget"], 24_576)

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_fallback_on_create_failure(self, mock_create):
        mock_create.side_effect = Exception("API error")
        model = self._make_model("gemini/gemini-3-flash-preview")
        request = _make_request("medium")
        client = model._get_streaming_client(request)
        self.assertIs(client, model._client)
