"""Tests for completion(), get_client(), model validation, hooks, and logging."""
from unittest import mock

from django.test import TestCase, override_settings

from llm_service.models import LLMCallLog


def _fake_response(usage=None, content="ok", model="openai/gpt-4o", id="resp-1"):
    """Build a minimal object that looks like a LiteLLM completion response."""
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
    r._hidden_params = {"response_cost": 0.0001}
    return r


class GetClientTestCase(TestCase):
    """Test get_client returns a client and is stable."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    def test_get_client_returns_same_instance(self):
        from llm_service.client import get_client
        c1 = get_client()
        c2 = get_client()
        self.assertIs(c1, c2)

    def test_get_client_has_completion_and_acompletion(self):
        from llm_service.client import get_client
        client = get_client()
        self.assertTrue(callable(getattr(client, "completion", None)))
        self.assertTrue(callable(getattr(client, "acompletion", None)))


class CompletionModelValidationTestCase(TestCase):
    """Test that completion() validates model against allowed list."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_completion_rejects_disallowed_model(self, mock_get_client):
        from llm_service.client import completion
        with self.assertRaises(ValueError) as ctx:
            completion(model="anthropic/claude-3", messages=[{"role": "user", "content": "Hi"}])
        self.assertIn("Model not allowed", str(ctx.exception))
        mock_get_client.assert_not_called()

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_completion_accepts_allowed_model(self, mock_get_client):
        from llm_service.client import completion
        mock_client = mock.Mock()
        mock_client.completion.return_value = _fake_response()
        mock_get_client.return_value = mock_client
        completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        mock_client.completion.assert_called_once()
        call_kwargs = mock_client.completion.call_args[1]
        self.assertEqual(call_kwargs["model"], "openai/gpt-4o")


class CompletionPrePostHooksTestCase(TestCase):
    """Test pre-call and post-call hooks."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_pre_call_hook_that_raises_blocks_call(self, mock_get_client):
        from llm_service.client import completion
        def block(request):
            raise ValueError("blocked")
        with override_settings(LLM_PRE_CALL_HOOKS=[block]):
            with self.assertRaises(ValueError) as ctx:
                completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
            self.assertIn("blocked", str(ctx.exception))
        mock_get_client.return_value.completion.assert_not_called()

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_post_call_hook_receives_result(self, mock_get_client):
        from llm_service.client import completion
        mock_client = mock.Mock()
        mock_client.completion.return_value = _fake_response(content="hello")
        mock_get_client.return_value = mock_client
        seen = []
        def record(result):
            seen.append(result)
        with override_settings(LLM_POST_CALL_HOOKS=[record]):
            completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].text, "hello")


class CompletionLoggingTestCase(TestCase):
    """Test that completion() writes LLMCallLog on success and on error."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_successful_completion_creates_log(self, mock_get_client):
        from llm_service.client import completion
        mock_client = mock.Mock()
        mock_client.completion.return_value = _fake_response()
        mock_get_client.return_value = mock_client
        self.assertEqual(LLMCallLog.objects.count(), 0)
        completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertEqual(log.model, "openai/gpt-4o")
        self.assertEqual(log.status, LLMCallLog.Status.SUCCESS)
        self.assertFalse(log.is_stream)
        self.assertIsNotNone(log.duration_ms)

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_failed_completion_creates_error_log(self, mock_get_client):
        from llm_service.client import completion
        mock_client = mock.Mock()
        mock_client.completion.side_effect = RuntimeError("API down")
        mock_get_client.return_value = mock_client
        self.assertEqual(LLMCallLog.objects.count(), 0)
        with self.assertRaises(RuntimeError):
            completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}])
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertEqual(log.status, LLMCallLog.Status.ERROR)
        self.assertIn("API down", log.error_message)
        self.assertEqual(log.error_type, "RuntimeError")

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_metadata_persisted_in_log(self, mock_get_client):
        from llm_service.client import completion
        mock_client = mock.Mock()
        mock_client.completion.return_value = _fake_response()
        mock_get_client.return_value = mock_client
        completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"feature": "test", "request_id": "req-123"},
        )
        log = LLMCallLog.objects.get()
        self.assertEqual(log.metadata.get("feature"), "test")
        self.assertEqual(log.request_id, "req-123")


class CompletionStreamTestCase(TestCase):
    """Test streaming: iterator yields chunks and writes one log when done."""

    def tearDown(self):
        import llm_service.client as client_mod
        client_mod._client = None

    @override_settings(LLM_ALLOWED_MODELS=["openai/gpt-4o"])
    @mock.patch("llm_service.client.get_client")
    def test_stream_yields_chunks_and_creates_one_log(self, mock_get_client):
        from llm_service.client import completion
        class FakeChunk:
            usage = None
            choices = [type("Delta", (), {"delta": type("C", (), {"content": "x"})()})()]
        def fake_stream(*args, **kwargs):
            yield FakeChunk()
            yield FakeChunk()
        mock_client = mock.Mock()
        mock_client.completion.side_effect = fake_stream
        mock_get_client.return_value = mock_client
        self.assertEqual(LLMCallLog.objects.count(), 0)
        chunks = list(completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi"}], stream=True))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(LLMCallLog.objects.count(), 1)
        log = LLMCallLog.objects.get()
        self.assertTrue(log.is_stream)
