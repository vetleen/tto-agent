"""Tests for model-specific provider reasoning parameters."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from llm.context_budget import output_token_cap
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
    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_off_explicitly_disables_thinking(self, create_variant):
        # Opus 5 / Sonnet 5 think by default, so "off" must send disabled
        # rather than omitting `thinking`.
        variant = MagicMock()
        create_variant.return_value = variant
        model = AnthropicChatModel("anthropic/claude-opus-5", MagicMock())
        result = model._get_streaming_client(_request("off"))
        self.assertEqual(create_variant.call_args.kwargs["thinking"], {"type": "disabled"})
        self.assertIs(result, variant.bind.return_value)
        variant.bind.assert_called_once_with(cache_control={"type": "ephemeral"})

    def test_missing_level_uses_base_client(self):
        client = MagicMock()
        model = AnthropicChatModel("anthropic/claude-opus-5", client)
        result = model._get_streaming_client(_request(None))
        self.assertIs(result, client.bind.return_value)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_opus_55_default_is_adaptive_medium(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-opus-5-5", MagicMock())
        model._get_streaming_client(_request("medium"))
        kwargs = create_variant.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(kwargs["output_config"], {"effort": "medium"})

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_sonnet_55_default_is_adaptive_high(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-sonnet-5-5", MagicMock())
        model._get_streaming_client(_request("high"))
        kwargs = create_variant.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "adaptive", "display": "summarized"})
        self.assertEqual(kwargs["output_config"], {"effort": "high"})
        self.assertEqual(kwargs["max_tokens"], 128_000)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_haiku_55_is_adaptive_with_drop_block(self, create_variant):
        # Released after 2026-10-01: binds thinking to the prefix, so the
        # request opts into drop_block instead of 400ing at an edit point.
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-haiku-5-5", MagicMock())
        model._get_streaming_client(_request("medium"))
        kwargs = create_variant.call_args.kwargs
        self.assertEqual(
            kwargs["thinking"],
            {
                "type": "adaptive",
                "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            },
        )
        self.assertEqual(kwargs["output_config"], {"effort": "medium"})
        self.assertEqual(kwargs["max_tokens"], 128_000)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_haiku_55_off_disables_thinking(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-haiku-5-5", MagicMock())
        model._get_streaming_client(_request("off"))
        self.assertEqual(create_variant.call_args.kwargs["thinking"], {"type": "disabled"})

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_opus_48_xhigh_is_adaptive(self, create_variant):
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-opus-4-8", MagicMock())
        model._get_streaming_client(_request("xhigh"))
        self.assertEqual(create_variant.call_args.kwargs["output_config"], {"effort": "xhigh"})

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_structured_client_uses_native_json_schema_with_thinking(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = AnthropicChatModel("anthropic/claude-opus-5-5", MagicMock())
        schema = object()
        result = model.structured_client(_request("medium"), schema)
        self.assertEqual(create_variant.call_args.kwargs["output_config"], {"effort": "medium"})
        variant.with_structured_output.assert_called_once_with(
            schema, include_raw=True, method="json_schema"
        )
        self.assertIs(result, variant.with_structured_output.return_value)

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
    def test_unregistered_model_uses_budgeted_extended_thinking(self, create_variant):
        # No registered model is extended-only any more (Haiku 4.5 retired);
        # unregistered Anthropic names still fall back to budget_tokens.
        create_variant.return_value = MagicMock()
        model = AnthropicChatModel("anthropic/claude-custom-legacy", MagicMock())
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

    @patch("llm.core.providers.openai.create_variant_client")
    def test_structured_client_honours_reasoning_with_default_method(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = OpenAIChatModel("openai/gpt-5.6-luna", MagicMock())
        schema = object()
        model.structured_client(_request("max"), schema)
        self.assertEqual(create_variant.call_args.kwargs["reasoning"]["effort"], "max")
        variant.with_structured_output.assert_called_once_with(schema, include_raw=True)

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

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_structured_client_honours_thinking_level(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        schema = object()
        model.structured_client(_request("medium"), schema)
        self.assertEqual(create_variant.call_args.kwargs["thinking_level"], "medium")
        variant.with_structured_output.assert_called_once_with(schema, include_raw=True)

    def test_function_call_thought_signature_is_preserved(self):
        key = "__gemini_function_call_thought_signatures__"
        model = GeminiChatModel("gemini/gemini-3.7-flash", MagicMock())
        self.assertEqual(
            model._extract_replay_metadata(
                SimpleNamespace(additional_kwargs={key: {"call_1": "signature"}})
            ),
            {"additional_kwargs": {key: {"call_1": "signature"}}},
        )

    def test_thinking_parts_are_parsed(self):
        # langchain-google-genai 4.x shape for thought summaries.
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        chunk = SimpleNamespace(content=[
            {"type": "thinking", "thinking": "Plan", "index": "0"},
            {"type": "text", "text": "Answer", "index": "1"},
        ])
        self.assertEqual(
            model._parse_chunk(chunk),
            [("thinking", {"text": "Plan"}), ("token", {"text": "Answer"})],
        )

    def test_legacy_thought_parts_still_parsed(self):
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        chunk = SimpleNamespace(content=[{"thought": True, "text": "Plan"}])
        self.assertEqual(model._parse_chunk(chunk), [("thinking", {"text": "Plan"})])

    def test_signature_only_text_part_yields_nothing(self):
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        chunk = SimpleNamespace(content=[
            {"type": "text", "text": "", "extras": {"signature": "abc"}, "index": "1"},
        ])
        self.assertEqual(model._parse_chunk(chunk), [])

    def test_signed_text_part_is_kept_for_replay_without_thinking(self):
        # The base keeps content blocks only when a thinking block exists; a
        # signed text part alone must still travel back.
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        content = [
            {"type": "text", "text": "Answer"},
            {"type": "text", "text": "", "extras": {"signature": "c2ln"}},
        ]
        meta = model._extract_replay_metadata(SimpleNamespace(additional_kwargs={}, content=content))
        self.assertEqual(meta, {"content_blocks": content})

    def test_unsigned_reply_needs_no_replay(self):
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        meta = model._extract_replay_metadata(
            SimpleNamespace(additional_kwargs={}, content=[{"type": "text", "text": "Answer"}])
        )
        self.assertEqual(meta, {})

    def test_guard_sees_gemini_thinking(self):
        # A runaway inside Gemini's reasoning must trip the repetition guard.
        from langchain_core.messages import AIMessageChunk

        chunks = [
            AIMessageChunk(content=[{"type": "thinking", "thinking": " |" * 200, "index": "0"}])
            for _ in range(120)
        ]
        client = MagicMock()
        client.stream = MagicMock(return_value=iter(chunks))
        client.bind = MagicMock(return_value=client)
        model = GeminiChatModel("gemini/gemini-3.8-flash", client)
        request = ChatRequest(
            messages=[{"role": "user", "content": "hi"}], model="gemini/gemini-3.8-flash",
            stream=True,
        )
        events = list(model.stream(request))
        self.assertTrue(any(e.event_type == "thinking" for e in events))
        end = next(e for e in events if e.event_type == "message_end")
        self.assertTrue(end.data["output_truncated"])
        self.assertEqual(end.data["stop_reason"], "repetition")

    @patch("llm.core.providers.gemini.create_variant_client", side_effect=ValueError("bad effort"))
    def test_invalid_provider_config_is_not_silently_disabled(self, _create_variant):
        model = GeminiChatModel("gemini/gemini-3.7-flash", MagicMock())
        with self.assertRaisesRegex(ValueError, "bad effort"):
            model._get_streaming_client(_request("medium"))


class OutputCapBindingTests(SimpleTestCase):
    """OpenAI and Gemini requests carry a per-request output-token cap; Anthropic
    manages its own ``max_tokens`` alongside its thinking config."""

    @patch("llm.core.providers.openai.create_variant_client")
    def test_openai_binds_max_tokens(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = OpenAIChatModel("openai/gpt-5.6-sol", MagicMock())
        result = model._get_streaming_client(_request("high"))
        variant.bind.assert_called_once_with(
            max_tokens=output_token_cap("openai/gpt-5.6-sol", "high")
        )
        self.assertIs(result, variant.bind.return_value)

    @patch("llm.core.providers.openai.create_variant_client")
    def test_openai_cap_is_bound_after_tools(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = OpenAIChatModel("openai/gpt-5.6-sol", MagicMock())
        req = _request("medium").model_copy(update={"tool_schemas": [MagicMock()]})
        result = model._get_streaming_client(req)
        with_tools = variant.bind_tools.return_value
        with_tools.bind.assert_called_once_with(
            max_tokens=output_token_cap("openai/gpt-5.6-sol", "medium")
        )
        self.assertIs(result, with_tools.bind.return_value)

    def test_openai_without_reasoning_level_still_capped(self):
        client = MagicMock()
        model = OpenAIChatModel("openai/gpt-5.6-sol", client)
        model._get_streaming_client(_request(None))
        client.bind.assert_called_once_with(max_tokens=output_token_cap("openai/gpt-5.6-sol", None))

    @patch("llm.core.providers.openai.create_variant_client")
    def test_openai_generate_uses_the_capped_client(self, create_variant):
        from langchain_core.messages import AIMessage

        variant = MagicMock()
        create_variant.return_value = variant
        bound = variant.bind.return_value
        bound.invoke.return_value = AIMessage(content="hi")
        model = OpenAIChatModel("openai/gpt-5.6-sol", MagicMock())
        response = model.generate(_request("high"))
        bound.invoke.assert_called_once()
        self.assertEqual(response.message.content, "hi")
        self.assertFalse(response.metadata["output_truncated"])

    @patch("llm.core.providers.gemini.create_variant_client")
    def test_gemini_binds_max_output_tokens(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = GeminiChatModel("gemini/gemini-3.8-flash", MagicMock())
        result = model._get_streaming_client(_request("medium"))
        variant.bind.assert_called_once_with(
            max_output_tokens=output_token_cap("gemini/gemini-3.8-flash", "medium")
        )
        self.assertIs(result, variant.bind.return_value)

    @patch("llm.core.providers.anthropic.create_variant_client")
    def test_anthropic_binds_only_cache_control(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        model = AnthropicChatModel("anthropic/claude-opus-5-5", MagicMock())
        model._get_streaming_client(_request("high"))
        variant.bind.assert_called_once_with(cache_control={"type": "ephemeral"})

    @patch("llm.core.providers.openai.create_variant_client")
    def test_structured_client_is_not_capped(self, create_variant):
        variant = MagicMock()
        create_variant.return_value = variant
        OpenAIChatModel("openai/gpt-5.6-sol", MagicMock()).structured_client(_request("max"), object())
        variant.bind.assert_not_called()
