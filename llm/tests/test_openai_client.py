"""Tests for the shared OpenAI SDK client (llm.service.openai_client)."""

from unittest.mock import patch

from django.test import SimpleTestCase


class SharedOpenAIClientTests(SimpleTestCase):
    def test_transcription_alias_is_the_shared_client(self):
        # The lift must not fork the client: transcription and decisions share
        # one connection pool through the same function object.
        from llm.service import transcription_service as ts
        from llm.service.openai_client import get_openai_client

        self.assertIs(ts._get_openai_client, get_openai_client)

    def test_client_cached_per_class_with_pinned_timeouts(self):
        from llm.service.openai_client import get_openai_client

        with patch("openai.OpenAI") as first_cls:
            c1 = get_openai_client()
            c2 = get_openai_client()
            self.assertIs(c1, c2)
            self.assertIs(c1, first_cls.return_value)
            first_cls.assert_called_once()
            kwargs = first_cls.call_args.kwargs
            self.assertEqual(kwargs["max_retries"], 2)
            self.assertEqual(kwargs["timeout"].connect, 10.0)
            self.assertEqual(kwargs["timeout"].read, 300.0)
            self.assertEqual(kwargs["timeout"].write, 120.0)

        with patch("openai.OpenAI") as second_cls:
            # A new class object (a fresh patch) gets a freshly built client.
            c3 = get_openai_client()
            self.assertIs(c3, second_cls.return_value)
            self.assertIsNot(c3, c1)
