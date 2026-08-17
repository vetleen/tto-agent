"""Tests for model-specific provider reasoning parameters."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from llm.core.providers.anthropic import AnthropicChatModel
from llm.core.providers.gemini import GeminiChatModel
from llm.core.providers.openai import OpenAIChatModel
from llm.types.requests import ChatRequest


def _request(level):
    return ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="test",
        stream=True,
        params={"thinking_level": level},
        tools=[],
    )


class AnthropicReasoningTests(SimpleTestCase):
    def test_off_uses_base_client(self):
        client = MagicMock()
        model = AnthropicChatModel("anthropic/claude-opus-5", client)
        result = model._get_streaming_client(_request("off"))
        self.assertIs(result, client.bind.return_value)
        client.bind.assert_called_once_with(cache_control={"type": "ephemeral"})

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_adaptive_effort_is_sent_in_output_config(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-fable-5", MagicMock())
        model._get_streaming_client(_request("xhigh"))
        kwargs = create_variant.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(kwargs["output_config"], {"effort": "xhigh"})
        self.assertEqual(kwargs["max_tokens"], 128_000)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_opus_46_uses_adaptive_thinking(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-opus-4-6", MagicMock())
        model._get_streaming_client(_request("max"))
        self.assertEqual(create_variant.call_args.kwargs["thinking"]["type"], "adaptive")
        self.assertEqual(create_variant.call_args.kwargs["output_config"], {"effort": "max"})

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_haiku_keeps_budgeted_extended_thinking(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-haiku-4-5", MagicMock())
        model._get_streaming_client(_request("medium"))
        kwargs = create_variant.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "enabled", "budget_tokens": 10_000})
        self.assertEqual(kwargs["max_tokens"], 16_384)

    def test_refusal_content_is_visible(self):
        model = AnthropicChatModel("anthropic/claude-fable-5", MagicMock())
        chunk = SimpleNamespace(content=[{"type": "refusal", "refusal": "I can't help."}])
        self.assertEqual(model._parse_chunk(chunk), [("token", {"text": "I can't help."})])

    @patch("llm.core.providers.anthropic.create_variant_client", side_effect=ValueError("bad effort"))
    def test_invalid_provider_config_is_not_silently_disabled(self, _create_variant):
        model = AnthropicChatModel("anthropic/claude-opus-5", MagicMock())
        with self.assertRaisesRegex(ValueError, "bad effort"):
            model._get_streaming_client(_request("high"))


class OpenAIReasoningTests(SimpleTestCase):
    @patch("llm.core.providers.openai.create_variant_client")
    def test_all_gpt56_levels_are_forwarded_without_clamping(self, create_variant):
        create_variant.return_value = MagicMock()
        model = OpenAIChatModel("openai/gpt-5.6-sol", MagicMock())
        for level in ("none", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(level=level):
                create_variant.reset_mock()
                model._get_streaming_client(_request(level))
                reasoning = create_variant.call_args.kwargs["reasoning"]
                self.assertEqual(reasoning["effort"], level)
                if level == "none":
                    self.assertNotIn("summary", reasoning)
                else:
                    self.assertEqual(reasoning["summary"], "auto")

    def test_reasoning_summary_is_parsed_from_responses_content(self):
        model = OpenAIChatModel("openai/gpt-5.6-terra", MagicMock())
        chunk = SimpleNamespace(
            additional_kwargs={},
            content=[
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Plan"}]},
                {"type": "text", "text": "Answer"},
            ],
        )
        self.assertEqual(
            model._parse_chunk(chunk),
            [("thinking", {"text": "Plan"}), ("token", {"text": "Answer"})],
        )

    def test_encrypted_reasoning_is_preserved_for_replay(self):
        model = OpenAIChatModel("openai/gpt-5.6-terra", MagicMock())
        content = [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher", "summary": []},
            {"type": "text", "text": "Calling a tool"},
            {"type": "function_call", "name": "lookup"},
        ]
        self.assertEqual(
            model._extract_replay_metadata(SimpleNamespace(content=content)),
            {"content_blocks": content[:2]},
        )

    @patch("llm.core.providers.openai.create_variant_client", side_effect=ValueError("bad effort"))
    def test_invalid_provider_config_is_not_silently_disabled(self, _create_variant):
        model = OpenAIChatModel("openai/gpt-5.6-terra", MagicMock())
        with self.assertRaisesRegex(ValueError, "bad effort"):
            model._get_streaming_client(_request("max"))


class GeminiReasoningTests(SimpleTestCase):
    @patch("llm.core.providers.gemini.create_variant_client")
    def test_thinking_level_is_sent_as_string(self, create_variant):
        create_variant.return_value = MagicMock()
        model = GeminiChatModel("gemini/gemini-3.7-flash", MagicMock())
        model._get_streaming_client(_request("medium"))
        self.assertEqual(create_variant.call_args.kwargs["thinking_level"], "medium")
        self.assertTrue(create_variant.call_args.kwargs["include_thoughts"])
        self.assertNotIn("thinking_budget", create_variant.call_args.kwargs)

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_flash_lite_minimal_is_forwarded(self, create_variant):
        create_variant.return_value = MagicMock()
        model = GeminiChatModel("gemini/gemini-3.5-flash-lite", MagicMock())
        model._get_streaming_client(_request("minimal"))
        self.assertEqual(create_variant.call_args.kwargs["thinking_level"], "minimal")

    def test_function_call_thought_signature_is_preserved(self):
        key = "__gemini_function_call_thought_signatures__"
        model = GeminiChatModel("gemini/gemini-3.7-flash", MagicMock())
        self.assertEqual(
            model._extract_replay_metadata(
                SimpleNamespace(additional_kwargs={key: {"call_1": "signature"}})
            ),
            {"additional_kwargs": {key: {"call_1": "signature"}}},
        )

    @patch("llm.core.providers.gemini.create_variant_client", side_effect=ValueError("bad effort"))
    def test_invalid_provider_config_is_not_silently_disabled(self, _create_variant):
        model = GeminiChatModel("gemini/gemini-3.7-flash", MagicMock())
        with self.assertRaisesRegex(ValueError, "bad effort"):
            model._get_streaming_client(_request("medium"))
