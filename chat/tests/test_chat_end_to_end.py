"""End-to-end chat tests: WebSocket -> consumer -> LLMService -> streamed events -> DB."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional
from unittest.mock import patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from documents.models import Project
from llm.core.registry import get_model_registry
from llm.types.messages import Message, ToolCall
from llm.types.responses import ChatResponse
from llm.types.streaming import StreamEvent

User = get_user_model()


def make_application():
    return URLRouter(websocket_urlpatterns)


@dataclass
class _FakeChatModel:
    """Deterministic fake model for tests (no network)."""

    model_name: str
    tokens: str = "OK"
    tool_call_on_first_generate: bool = False

    # captured requests
    generate_requests: List[object] = None  # type: ignore[assignment]
    stream_requests: List[object] = None  # type: ignore[assignment]
    _generate_calls: int = 0

    def __post_init__(self) -> None:
        self.generate_requests = []
        self.stream_requests = []

    def generate(self, request) -> ChatResponse:
        self._generate_calls += 1
        self.generate_requests.append(request)

        tool_calls: Optional[list[ToolCall]] = None
        if self.tool_call_on_first_generate and self._generate_calls == 1:
            tool_calls = [
                ToolCall(
                    id="tc1",
                    name="search_documents",
                    arguments={"query": "foo", "k": 1},
                )
            ]

        return ChatResponse(
            message=Message(role="assistant", content="", tool_calls=tool_calls),
            model=self.model_name,
            usage=None,
            metadata={},
        )

    def stream(self, request) -> Iterator[StreamEvent]:
        self.stream_requests.append(request)
        run_id = request.context.run_id if request.context else ""
        yield StreamEvent(event_type="message_start", data={}, sequence=1, run_id=run_id)
        yield StreamEvent(event_type="token", data={"text": self.tokens}, sequence=2, run_id=run_id)
        yield StreamEvent(event_type="message_end", data={}, sequence=3, run_id=run_id)


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ChatEndToEndTests(TransactionTestCase):
    """Exercise the whole path without real LLM APIs."""

    def setUp(self):
        # Ensure pipeline/tool registration side effects have happened.
        import llm.pipelines.simple_chat  # noqa: F401
        import chat.tools  # noqa: F401

        self.user = User.objects.create_user(email="owner@example.com", password="pass123")
        self.project = Project.objects.create(
            name="Test", slug="test-project", created_by=self.user
        )

        self._model_registry = get_model_registry()
        self._orig_prefix_factories = dict(self._model_registry._prefix_factories)

    def tearDown(self):
        # Restore global registry (tests run in-process).
        self._model_registry._prefix_factories = dict(self._orig_prefix_factories)

    async def _connect(self):
        app = make_application()
        communicator = WebsocketCommunicator(app, f"/ws/projects/{self.project.uuid}/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        assert connected
        return communicator

    def _register_fake_model(self, fake: _FakeChatModel) -> None:
        # ModelRegistry resolves by prefix match.
        self._model_registry.register_model_prefix("test-", lambda _name: fake)

    async def _db_messages(self):
        return await database_sync_to_async(list)(
            ChatMessage.objects.order_by("created_at").values_list("role", "content")
        )

    @patch.dict(
        os.environ,
        {"LLM_ALLOWED_MODELS": "test-model", "DEFAULT_LLM_MODEL": "test-model"},
        clear=False,
    )
    async def test_user_message_to_llm_tokens_to_persisted_assistant(self):
        fake = _FakeChatModel(model_name="test-model", tokens="Hello")
        self._register_fake_model(fake)

        communicator = await self._connect()
        await communicator.send_json_to({"type": "chat.message", "content": "Hi"})

        # Thread is created first time
        created = await communicator.receive_json_from(timeout=5)
        self.assertEqual(created["event_type"], "thread.created")

        # Then LLM stream events (message_start, token, message_end)
        ev1 = await communicator.receive_json_from(timeout=5)
        ev2 = await communicator.receive_json_from(timeout=5)
        ev3 = await communicator.receive_json_from(timeout=5)
        self.assertEqual(ev1["event_type"], "message_start")
        self.assertEqual(ev2["event_type"], "token")
        self.assertEqual(ev2["data"].get("text"), "Hello")
        self.assertEqual(ev3["event_type"], "message_end")

        await communicator.disconnect()

        # DB should have user + assistant messages (assistant accumulated from token stream)
        msgs = await self._db_messages()
        self.assertEqual([m[0] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[0][1], "Hi")
        self.assertEqual(msgs[1][1], "Hello")

        # LLM requests should include a system prompt and the conversation context
        self.assertGreaterEqual(len(fake.generate_requests), 1)
        req = fake.generate_requests[0]
        roles = [m.role for m in req.messages]
        self.assertEqual(roles[0], "system")
        self.assertIn("helpful assistant", req.messages[0].content.lower())
        self.assertIn("Test", req.messages[0].content)
        self.assertEqual(req.messages[-1].role, "user")
        self.assertEqual(req.messages[-1].content, "Hi")
        self.assertEqual(req.model, "test-model")
        self.assertEqual(req.context.conversation_id, str(self.project.pk))

    @patch.dict(
        os.environ,
        {"LLM_ALLOWED_MODELS": "test-model", "DEFAULT_LLM_MODEL": "test-model"},
        clear=False,
    )
    @patch("documents.services.retrieval.similarity_search_chunks")
    async def test_tool_events_emitted_and_tool_called(self, mock_search):
        mock_search.return_value = []

        fake = _FakeChatModel(
            model_name="test-model",
            tokens="Done",
            tool_call_on_first_generate=True,
        )
        self._register_fake_model(fake)

        communicator = await self._connect()
        await communicator.send_json_to({"type": "chat.message", "content": "Search please"})

        created = await communicator.receive_json_from(timeout=5)
        self.assertEqual(created["event_type"], "thread.created")

        # Tool loop events then message stream (message_start, token, message_end)
        tool_start = await communicator.receive_json_from(timeout=5)
        tool_end = await communicator.receive_json_from(timeout=5)
        msg_start = await communicator.receive_json_from(timeout=5)
        token_ev = await communicator.receive_json_from(timeout=5)
        msg_end = await communicator.receive_json_from(timeout=5)

        self.assertEqual(tool_start["event_type"], "tool_start")
        self.assertEqual(tool_start["data"]["tool_name"], "search_documents")
        self.assertEqual(tool_end["event_type"], "tool_end")
        self.assertEqual(tool_end["data"]["tool_name"], "search_documents")
        self.assertEqual(msg_start["event_type"], "message_start")
        self.assertEqual(token_ev["event_type"], "token")
        self.assertEqual(token_ev["data"].get("text"), "Done")
        self.assertEqual(msg_end["event_type"], "message_end")

        await communicator.disconnect()

        mock_search.assert_called_once_with(project_id=self.project.pk, query="foo", k=1)

    @patch.dict(os.environ, {"LLM_ALLOWED_MODELS": "", "DEFAULT_LLM_MODEL": ""}, clear=False)
    async def test_missing_llm_allowed_models_surfaces_as_error_event(self):
        # Intentionally do not register any model; resolve_model should fail before registry lookup.
        communicator = await self._connect()
        await communicator.send_json_to({"type": "chat.message", "content": "Hi"})

        created = await communicator.receive_json_from(timeout=5)
        self.assertEqual(created["event_type"], "thread.created")

        err = await communicator.receive_json_from(timeout=5)
        self.assertEqual(err["event_type"], "error")
        self.assertIn("Failed to get AI response", err["data"]["message"])

        await communicator.disconnect()

        msgs = await self._db_messages()
        # User message is persisted before LLM call; assistant should not be persisted.
        self.assertEqual([m[0] for m in msgs], ["user"])

