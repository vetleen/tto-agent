"""
Integration tests: full completion() flow with mocked LiteLLM.
No real API calls unless TEST_APIS=True (see test_with_real_api below).
"""
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from llm_service.models import LLMCallLog

User = get_user_model()


def _fake_response(usage=None, content="Hello", model="openai/gpt-4o", id="resp-1"):
    u = usage or {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    class Usage:
        prompt_tokens = u.get("prompt_tokens", 0)
        completion_tokens = u.get("completion_tokens", 0)
        total_tokens = u.get("total_tokens", 0)
        input_tokens = u.get("input_tokens", u.get("prompt_tokens", 0))
        output_tokens = u.get("output_tokens", u.get("completion_tokens", 0))
    class Message:
        pass
    class Choice:
        message = None
        delta = None
    class Response:
        pass
    msg = Message()
    msg.content = content
    choice = Choice()
    choice.message = msg
    r = Response()
    r.usage = Usage()
    r.choices = [choice]
    r.model = model
    r.id = id
    r._hidden_params = {"response_cost": 0.0002}
    return r


class CompletionFlowIntegrationTestCase(TestCase):
    """Full flow: completion() -> client -> litellm (mocked) -> log."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o", "openai/gpt-4o-mini"])
    @mock.patch("litellm.completion")
    def test_full_completion_flow_creates_log_with_usage_and_cost(self, mock_completion):
        from llm_service.client import completion
        mock_completion.return_value = _fake_response(content="Hi back")
        self.assertEqual(LLMCallLog.objects.count(), 0)
        response = completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertEqual(log.model, "openai/gpt-4o")
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)
        self.assertEqual(log.response_preview, "Hi back")
        self.assertGreater(log.total_tokens, 0)
        self.assertIsNotNone(log.cost_usd)
        self.assertIsNotNone(log.duration_ms)
        self.assertEqual(response.choices[0].message.content, "Hi back")

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("litellm.completion")
    def test_full_flow_with_user_and_metadata(self, mock_completion):
        from llm_service.client import completion
        user = User.objects.create_user(email="u@example.com", password="pw")
        mock_completion.return_value = _fake_response()
        completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            user=user,
            metadata={"feature": "chat", "request_id": "req-456"},
        )
        log = LLMCallLog.objects.get()
        self.assertEqual(log.user_id, user.id)
        self.assertEqual(log.metadata.get("feature"), "chat")
        self.assertEqual(log.request_id, "req-456")

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("litellm.completion")
    def test_stream_flow_creates_single_log_after_consumption(self, mock_completion):
        from llm_service.client import completion
        class C:
            usage = None
            choices = [type("X", (), {"delta": type("D", (), {"content": "a"})()})()]
        def gen():
            yield C()
            yield C()
        mock_completion.return_value = gen()
        self.assertEqual(LLMCallLog.objects.count(), 0)
        list(completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True))
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertTrue(log.is_stream)


class MinimalLogFallbackTestCase(TestCase):
    """When full log write fails, minimal log is written."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    @mock.patch("llm_service.client._write_log")
    def test_minimal_log_created_when_full_log_fails(self, mock_write_log, mock_get_client):
        from llm_service.client import completion
        mock_get_client.return_value.completion.return_value = _fake_response()
        mock_write_log.return_value = None  # simulate failure
        completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertEqual(log.status, LLMCallLog.Status.LOGGING_FAILED)
        self.assertIn("log write failed", log.error_message)
