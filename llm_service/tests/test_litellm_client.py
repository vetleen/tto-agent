"""Tests for LiteLLMClient (underlying client implementation)."""
from unittest import mock

from django.test import TestCase, override_settings

from llm_service.base import BaseLLMClient
from llm_service.litellm_client import LiteLLMClient


class LiteLLMClientTestCase(TestCase):
    """Test LiteLLMClient passes timeout and calls litellm."""

    @override_settings(LLM_REQUEST_TIMEOUT=45.0)
    @mock.patch("litellm.completion")
    def test_completion_passes_timeout_and_calls_litellm(self, mock_completion):
        mock_completion.return_value = "response"
        client = LiteLLMClient()
        result = client.completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "response")
        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs.get("timeout"), 45.0)
        self.assertEqual(call_kwargs["model"], "openai/gpt-4o")

    @mock.patch("litellm.completion")
    def test_completion_does_not_override_explicit_timeout(self, mock_completion):
        mock_completion.return_value = "ok"
        client = LiteLLMClient()
        client.completion(model="m", messages=[], timeout=90)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["timeout"], 90)

    @override_settings(LLM_REQUEST_TIMEOUT=30.0)
    @mock.patch("litellm.acompletion")
    def test_acompletion_calls_litellm_acompletion(self, mock_acompletion):
        async def fake_acompletion(**kwargs):
            return "async-response"
        mock_acompletion.side_effect = fake_acompletion
        client = LiteLLMClient()
        import asyncio
        result = asyncio.run(
            client.acompletion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        )
        self.assertEqual(result, "async-response")

    def test_litellm_client_is_base_client(self):
        self.assertTrue(issubclass(LiteLLMClient, BaseLLMClient))
