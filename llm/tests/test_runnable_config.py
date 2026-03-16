"""Tests for _build_config() on BaseLangChainChatModel."""

from django.test import TestCase

from llm.core.providers.base import BaseLangChainChatModel
from llm.types.context import RunContext
from llm.types.messages import Message
from llm.types.requests import ChatRequest


class _FakeModel(BaseLangChainChatModel):
    """Minimal concrete subclass for testing."""

    def __init__(self):
        super().__init__(model_name="test-model", client=None)
        self._provider_label = "TestProvider"


class BuildConfigTests(TestCase):

    def setUp(self):
        self.model = _FakeModel()

    def test_basic_structure_with_callbacks(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=RunContext.create(user_id="u1", conversation_id="c1"),
        )
        callbacks = ["cb1"]
        config = self.model._build_config(request, callbacks)

        self.assertEqual(config["callbacks"], ["cb1"])
        self.assertIn("metadata", config)
        self.assertIn("tags", config)
        self.assertIn("run_name", config)

    def test_metadata_contains_context_fields(self):
        ctx = RunContext.create(user_id="u42", conversation_id="conv-99")
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=ctx,
        )
        config = self.model._build_config(request, [])

        meta = config["metadata"]
        self.assertEqual(meta["run_id"], ctx.run_id)
        self.assertEqual(meta["trace_id"], ctx.trace_id)
        self.assertEqual(meta["user_id"], "u42")
        self.assertEqual(meta["conversation_id"], "conv-99")

    def test_tags_include_user_and_model(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=RunContext.create(user_id="u1"),
        )
        config = self.model._build_config(request, [])
        self.assertIn("user:u1", config["tags"])
        self.assertIn("model:test-model", config["tags"])

    def test_anonymous_user_tag(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=RunContext.create(),
        )
        config = self.model._build_config(request, [])
        self.assertIn("user:anon", config["tags"])

    def test_run_name_format(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=RunContext.create(),
        )
        config = self.model._build_config(request, [])
        self.assertEqual(config["run_name"], "TestProvider/test-model")

    def test_no_context(self):
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            context=None,
        )
        config = self.model._build_config(request, ["cb"])
        self.assertEqual(config["callbacks"], ["cb"])
        self.assertNotIn("metadata", config)
        self.assertNotIn("tags", config)
