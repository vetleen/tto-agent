"""Pins the LangChain behaviour the output cap relies on: a kwarg bound on the
client reaches the provider payload under the parameter each API expects.
Real client objects with dummy keys; nothing is sent."""

from __future__ import annotations

from unittest import TestCase

from langchain_core.messages import HumanMessage


class OpenAIOutputCapWiringTests(TestCase):
    def test_responses_api_renames_max_tokens_to_max_output_tokens(self):
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(model="gpt-5.4", api_key="x", use_responses_api=True)
        payload = client._get_request_payload([HumanMessage("hi")], max_tokens=123)
        self.assertEqual(payload["max_output_tokens"], 123)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

    def test_chat_completions_renames_max_tokens_to_max_completion_tokens(self):
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(model="gpt-4o-mini", api_key="x")
        payload = client._get_request_payload([HumanMessage("hi")], max_tokens=123)
        self.assertEqual(payload["max_completion_tokens"], 123)
        self.assertNotIn("max_tokens", payload)

    def test_bound_kwarg_reaches_the_payload(self):
        # .bind(max_tokens=...) is what the adapter does; the bound kwargs are
        # merged into the request payload on every call.
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(model="gpt-5.4", api_key="x", use_responses_api=True)
        bound = client.bind(max_tokens=321)
        payload = client._get_request_payload([HumanMessage("hi")], **bound.kwargs)
        self.assertEqual(payload["max_output_tokens"], 321)


class GeminiOutputCapWiringTests(TestCase):
    def test_call_time_max_output_tokens_reaches_generation_config(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key="x")
        config = client._prepare_params(None, max_output_tokens=123)
        self.assertEqual(config.max_output_tokens, 123)
        bound = client.bind(max_output_tokens=321)
        config = client._prepare_params(None, **bound.kwargs)
        self.assertEqual(config.max_output_tokens, 321)
