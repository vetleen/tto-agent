"""Consumer wiring for Automatic model choice (chat/auto_model.py).

Same conventions as test_turn_gate_consumer.py: never wait on a socket
timeout (it cancels the app) — read only events known to arrive.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from channels.db import database_sync_to_async
from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings

from chat import turn_gate
from chat.models import ChatMessage, ChatThread
from chat.routing import websocket_urlpatterns
from chat.tests.test_turn_gate_consumer import StreamDispatcher

User = get_user_model()


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class AutoModelConsumerTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="auto-owner@example.com", password="pass123")
        turn_gate.reset()
        self.addCleanup(turn_gate.reset)
        from guardrails.service import GuardrailVerdict

        gr = patch(
            "guardrails.service.run_classifier_pipeline",
            new_callable=AsyncMock,
            return_value=GuardrailVerdict(action="allow"),
        )
        gr.start()
        self.addCleanup(gr.stop)
        # Titles go through the (mocked) service's arun; keep them out of it.
        title = patch("chat.consumers.ChatConsumer._generate_thread_title", new_callable=AsyncMock)
        title.start()
        self.addCleanup(title.stop)

    async def _connect(self):
        communicator = WebsocketCommunicator(URLRouter(websocket_urlpatterns), "/ws/chat/")
        communicator.scope["user"] = self.user
        connected, _ = await communicator.connect()
        self.assertTrue(connected)
        return communicator

    async def _turn(self, get_service, payload):
        """Send one message with a non-blocking stream; return (events, dispatcher)."""
        dispatcher = StreamDispatcher(1)
        dispatcher.release_all()
        service = MagicMock()
        service.astream = dispatcher.astream
        get_service.return_value = service

        communicator = await self._connect()
        await communicator.send_json_to({"type": "chat.message", **payload})
        events = []
        while True:
            event = await communicator.receive_json_from(timeout=10)
            events.append(event)
            if event["event_type"] in ("message_end", "error"):
                break
        await communicator.disconnect()
        return events, dispatcher

    def _expected(self, category):
        from chat.auto_model import resolve_category
        from core.preferences import get_preferences

        return resolve_category(get_preferences(self.user), category)

    @patch("chat.auto_model.decide_category", new_callable=AsyncMock, return_value="writing")
    @patch("llm.get_llm_service")
    async def test_auto_turn_routes_before_streaming(self, get_service, _decide):
        events, dispatcher = await self._turn(get_service, {"content": "Draft the report", "model": "auto"})

        kinds = [e["event_type"] for e in events]
        self.assertIn("turn.auto_routed", kinds)
        self.assertLess(kinds.index("turn.auto_routed"), kinds.index("message_start"))
        routed = events[kinds.index("turn.auto_routed")]["data"]
        model, _level = await database_sync_to_async(self._expected)("writing")
        self.assertEqual((routed["category"], routed["source"], routed["model"]), ("writing", "router", model))

        request = dispatcher.streams[0].requests[0]
        self.assertEqual(request.model, model)
        self.assertEqual(request.params["_log_metadata"]["auto_route"]["category"], "writing")

        thread = await database_sync_to_async(ChatThread.objects.get)(created_by=self.user)
        self.assertEqual(thread.model, model)
        self.assertTrue(thread.auto_route["enabled"])
        self.assertEqual(thread.auto_route["category"], "writing")
        self.assertEqual(thread.auto_route["model"], model)
        self.assertTrue(thread.auto_route["last_turn_at"])

        reply = await database_sync_to_async(
            lambda: ChatMessage.objects.filter(thread=thread, role="assistant").last()
        )()
        self.assertEqual(reply.metadata["auto_route"]["category"], "writing")

    @patch("chat.auto_model.decide_category", new_callable=AsyncMock, return_value="writing")
    @patch("llm.get_llm_service")
    async def test_concrete_model_turns_automatic_off(self, get_service, _decide):
        await self._turn(get_service, {"content": "Draft the report", "model": "auto"})
        thread = await database_sync_to_async(ChatThread.objects.get)(created_by=self.user)
        manual = thread.model

        events, dispatcher = await self._turn(
            get_service, {"content": "and again", "thread_id": str(thread.id), "model": manual},
        )

        self.assertNotIn("turn.auto_routed", [e["event_type"] for e in events])
        self.assertNotIn("_log_metadata", dispatcher.streams[0].requests[0].params)
        await database_sync_to_async(thread.refresh_from_db)()
        self.assertFalse(thread.auto_route["enabled"])
        self.assertIsNone(thread.auto_route["category"])

    @override_settings(AUTO_MODEL_CACHE_TTL_SECONDS={"anthropic": 3600, "openai": 3600, "google_genai": 3600})
    @patch("chat.auto_model.decide_category", new_callable=AsyncMock, return_value="writing")
    @patch("llm.get_llm_service")
    async def test_warm_followup_keeps_model_without_deciding(self, get_service, decide):
        await self._turn(get_service, {"content": "Draft the report", "model": "auto"})
        thread = await database_sync_to_async(ChatThread.objects.get)(created_by=self.user)
        decide.reset_mock()
        decide.return_value = "cheap"

        events, _ = await self._turn(
            get_service, {"content": "thanks", "thread_id": str(thread.id), "model": "auto"},
        )

        routed = next(e for e in events if e["event_type"] == "turn.auto_routed")["data"]
        self.assertEqual(routed["source"], "cached")
        self.assertEqual(routed["model"], thread.model)
        decide.assert_not_awaited()
