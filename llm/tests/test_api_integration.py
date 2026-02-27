"""
Live API integration tests: call each provider through the public LLM service.

Run only when TEST_APIS=True in the environment and the corresponding API key is set.
Use: get_llm_service().run("simple_chat", request) as another app would.
"""

from unittest.mock import patch

from django.test import TestCase

from llm import get_llm_service
from llm.tests.utils import require_test_apis
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest


@require_test_apis()
class ProviderLiveAPITests(TestCase):
    """Send one message per provider and assert a valid response. Requires TEST_APIS=True."""

    def _run_simple_chat(self, model: str) -> str:
        service = get_llm_service()
        request = ChatRequest(
            messages=[Message(role="user", content="Reply with exactly the word OK and nothing else.")],
            stream=False,
            model=model,
            context=RunContext.create(),
        )
        response = service.run("simple_chat", request)
        self.assertEqual(response.message.role, "assistant")
        self.assertIsInstance(response.message.content, str)
        self.assertGreater(len(response.message.content.strip()), 0)
        return response.message.content

    def test_openai_returns_valid_response(self):
        """Call OpenAI (gpt-5-mini) through LLMService and assert valid response."""
        with patch.dict(
            "os.environ",
            {"LLM_ALLOWED_MODELS": "gpt-5-mini", "DEFAULT_LLM_MODEL": "gpt-5-mini"},
            clear=False,
        ):
            content = self._run_simple_chat("gpt-5-mini")
        self.assertIn("OK", content.upper())

    def test_anthropic_returns_valid_response(self):
        """Call Anthropic (claude-haiku-4-5-20251001) through LLMService and assert valid response."""
        with patch.dict(
            "os.environ",
            {"LLM_ALLOWED_MODELS": "claude-haiku-4-5-20251001", "DEFAULT_LLM_MODEL": "claude-haiku-4-5-20251001"},
            clear=False,
        ):
            content = self._run_simple_chat("claude-haiku-4-5-20251001")
        self.assertIn("OK", content.upper())

    def test_gemini_returns_valid_response(self):
        """Call Gemini (gemini-3.1-flash-image-preview) through LLMService and assert valid response."""
        with patch.dict(
            "os.environ",
            {"LLM_ALLOWED_MODELS": "gemini-3.1-flash-image-preview", "DEFAULT_LLM_MODEL": "gemini-3.1-flash-image-preview"},
            clear=False,
        ):
            content = self._run_simple_chat("gemini-3.1-flash-image-preview")
        self.assertIn("OK", content.upper())
