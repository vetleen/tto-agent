"""
Live API tests: real calls to providers (OpenAI, Anthropic, Gemini, Moonshot).

Run only when TEST_APIS=True and the relevant API key is set:
    TEST_APIS=True python manage.py test llm_service.tests.test_live_api -v 2

Tests: basic completion (OpenAI + one per provider), structured JSON, streaming, tool use.
(Chat-setup compatibility per model lives in llm_chat.tests.test_live_chat_setup.)
"""
import json
import os
import unittest

from django.test import TestCase, override_settings

from llm_service.models import LLMCallLog

# Skip entire module unless TEST_APIS is set and truthy
RUN_LIVE = os.environ.get("TEST_APIS", "").strip().lower() in ("1", "true", "yes")
REQUIRES_LIVE = unittest.skipUnless(
    RUN_LIVE,
    "Live API tests disabled. Set TEST_APIS=True and OPENAI_API_KEY to run.",
)

LIVE_MODEL = "openai/gpt-5-nano"


@override_settings(LLM_ALLOWED_MODELS=[LIVE_MODEL])
@REQUIRES_LIVE
class LiveBasicCompletionTest(TestCase):
    """Basic completion against the real API."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_basic_completion_returns_content(self):
        from llm_service.client import completion

        initial_count = LLMCallLog.objects.count()
        resp = completion(
            model=LIVE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
        )
        self.assertIsNotNone(resp)
        self.assertTrue(getattr(resp, "choices", None) and len(resp.choices) > 0)
        content = getattr(resp.choices[0].message, "content", None) or ""
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0, "Expected non-empty response content")
        self.assertEqual(LLMCallLog.objects.count(), initial_count + 1)
        log = LLMCallLog.objects.order_by("-created_at").first()
        self.assertEqual(log.model, LIVE_MODEL)
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)
        self.assertGreater(log.total_tokens, 0)


@override_settings(LLM_ALLOWED_MODELS=[LIVE_MODEL])
@REQUIRES_LIVE
class LiveStructuredResponseTest(TestCase):
    """Structured JSON response via response_format."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_structured_json_response_is_valid_json(self):
        from llm_service.client import completion

        resp = completion(
            model=LIVE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Return valid JSON only, with one key 'answer' and value a single word: yes or no.",
                }
            ],
            response_format={"type": "json_object"},
        )
        self.assertIsNotNone(resp)
        content = (getattr(resp.choices[0].message, "content", None) or "").strip()
        self.assertGreater(len(content), 0)
        parsed = json.loads(content)
        self.assertIsInstance(parsed, dict)
        self.assertIn("answer", parsed)
        self.assertIn(parsed["answer"], ("yes", "no"))


@override_settings(LLM_ALLOWED_MODELS=[LIVE_MODEL])
@REQUIRES_LIVE
class LiveStreamingTest(TestCase):
    """Streaming completion against the real API."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_streaming_returns_chunks_and_logs_once(self):
        from llm_service.client import completion

        initial_count = LLMCallLog.objects.count()
        stream = completion(
            model=LIVE_MODEL,
            messages=[{"role": "user", "content": "Say hello in one short word."}],
            stream=True,
        )
        chunks = list(stream)
        self.assertGreater(len(chunks), 0, "Expected at least one stream chunk")
        text_parts = []
        for chunk in chunks:
            if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    text_parts.append(delta.content)
        full_text = "".join(text_parts)
        self.assertGreater(len(full_text.strip()), 0, "Expected non-empty streamed content")
        self.assertEqual(LLMCallLog.objects.count(), initial_count + 1)
        log = LLMCallLog.objects.order_by("-created_at").first()
        self.assertTrue(log.is_stream)
        self.assertEqual(log.model, LIVE_MODEL)
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)


@override_settings(LLM_ALLOWED_MODELS=[LIVE_MODEL])
@REQUIRES_LIVE
class LiveToolUseTest(TestCase):
    """Tool use: request with tools and assert we get a valid response (content or tool_calls)."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_tool_use_returns_content_or_tool_calls(self):
        from llm_service.client import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        resp = completion(
            model=LIVE_MODEL,
            messages=[
                {"role": "user", "content": "What is the weather in Oslo? Use the get_weather tool if you have it."}
            ],
            tools=tools,
        )
        self.assertIsNotNone(resp)
        self.assertTrue(getattr(resp, "choices", None) and len(resp.choices) > 0)
        choice = resp.choices[0]
        message = getattr(choice, "message", None)
        self.assertIsNotNone(message)
        # Response may be content and/or tool_calls
        content = getattr(message, "content", None) or ""
        tool_calls = getattr(message, "tool_calls", None) or []
        has_content = isinstance(content, str) and len(content.strip()) > 0
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        self.assertTrue(
            has_content or has_tool_calls,
            "Expected either non-empty content or tool_calls from the API",
        )
        if has_tool_calls:
            tc = tool_calls[0]
            self.assertIn("function", tc)
            self.assertEqual(tc["function"].get("name"), "get_weather")
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args) if args.strip() else {}
            self.assertIn("location", args)


def _live_simple_completion_test(model: str) -> None:
    """Helper: run one simple completion and assert non-empty content and one log."""
    from llm_service.client import completion

    initial_count = LLMCallLog.objects.count()
    resp = completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
    )
    assert resp is not None
    assert getattr(resp, "choices", None) and len(resp.choices) > 0
    content = getattr(resp.choices[0].message, "content", None) or ""
    assert isinstance(content, str)
    assert len(content.strip()) > 0, "Expected non-empty response content"
    assert LLMCallLog.objects.count() == initial_count + 1
    log = LLMCallLog.objects.order_by("-created_at").first()
    assert log.model == model
    assert log.status == LLMCallLog.Status.SUCCESS


@override_settings(LLM_ALLOWED_MODELS=["anthropic/claude-sonnet-4-5-20250929"])
@REQUIRES_LIVE
class LiveAnthropicCompletionTest(TestCase):
    """Simple completion against Anthropic (requires ANTHROPIC_API_KEY)."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_anthropic_simple_completion(self):
        _live_simple_completion_test("anthropic/claude-sonnet-4-5-20250929")


@override_settings(LLM_ALLOWED_MODELS=["gemini/gemini-3-flash-preview"])
@REQUIRES_LIVE
class LiveGeminiCompletionTest(TestCase):
    """Simple completion against Gemini (requires GEMINI_API_KEY)."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_gemini_simple_completion(self):
        _live_simple_completion_test("gemini/gemini-3-flash-preview")
