"""Tests for headless loop execution (chat.loop_service.execute_loop_run)."""

from unittest.mock import AsyncMock, MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TransactionTestCase, override_settings
from django.utils import timezone

from chat.models import ChatMessage, ChatThread, Loop

User = get_user_model()


def _fake_service(text="Done."):
    from llm.types.streaming import StreamEvent

    events = [
        StreamEvent(event_type="message_start", data={}, sequence=0, run_id="r1"),
        StreamEvent(event_type="token", data={"text": text}, sequence=1, run_id="r1"),
        StreamEvent(event_type="message_end", data={}, sequence=2, run_id="r1"),
    ]
    service = MagicMock()

    async def mock_astream(*args, **kwargs):
        for e in events:
            yield e

    service.astream = mock_astream
    return service


@override_settings(
    CHANNEL_LAYERS={"default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}}
)
class ExecuteLoopRunTests(TransactionTestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="loop@example.com", password="pass123",
        )
        self.thread = ChatThread.objects.create(created_by=self.user, title="My loop")
        self.loop = Loop.objects.create(
            thread=self.thread, created_by=self.user, prompt="Summarize today.",
            history_mode=Loop.HistoryMode.FRESH, cadence_kind=Loop.Cadence.INTERVAL,
            interval_seconds=3600, next_run=timezone.now(), max_runs=10,
            running=True, locked_at=timezone.now(),
        )

    @patch("llm.get_llm_service")
    def test_run_persists_messages_and_reschedules(self, mock_get_service):
        mock_get_service.return_value = _fake_service("All done.")
        from chat.loop_service import execute_loop_run

        before = timezone.now()
        execute_loop_run(self.loop.id)

        msgs = list(ChatMessage.objects.filter(thread=self.thread).order_by("created_at"))
        self.assertTrue(any(m.role == "user" and m.content == "Summarize today." for m in msgs))
        self.assertTrue(any(m.role == "assistant" and "All done." in m.content for m in msgs))

        self.loop.refresh_from_db()
        self.assertEqual(self.loop.runs_completed, 1)
        self.assertFalse(self.loop.running)  # lock released
        self.assertIsNone(self.loop.locked_at)
        self.assertGreater(self.loop.next_run, before)  # rescheduled forward
        self.assertIsNotNone(self.loop.last_result_at)
        self.assertEqual(self.loop.consecutive_errors, 0)

    @patch("llm.get_llm_service")
    def test_auto_pause_at_max_runs(self, mock_get_service):
        mock_get_service.return_value = _fake_service()
        self.loop.max_runs = 1
        self.loop.save(update_fields=["max_runs"])
        from chat.loop_service import execute_loop_run

        execute_loop_run(self.loop.id)
        self.loop.refresh_from_db()
        self.assertEqual(self.loop.runs_completed, 1)
        self.assertEqual(self.loop.status, Loop.Status.PAUSED)
        self.assertFalse(self.loop.running)

    def _capture_service(self, captured):
        service = MagicMock()

        async def astream(mode, request, **kwargs):
            captured["messages"] = request.messages
            return
            yield  # make it an async generator

        service.astream = astream
        return service

    @patch("llm.get_llm_service")
    def test_fresh_mode_excludes_prior_history(self, mock_get_service):
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="Earlier reply.",
        )
        captured = {}
        mock_get_service.return_value = self._capture_service(captured)
        from chat.loop_service import execute_loop_run

        execute_loop_run(self.loop.id)  # loop is FRESH by default
        blob = " ".join(str(m.content) for m in captured["messages"])
        self.assertNotIn("Earlier reply.", blob)
        self.assertIn("Summarize today.", blob)

    @patch("llm.get_llm_service")
    def test_conversational_mode_includes_prior_history(self, mock_get_service):
        ChatMessage.objects.create(
            thread=self.thread, role="assistant", content="Earlier reply.",
        )
        self.loop.history_mode = Loop.HistoryMode.CONVERSATIONAL
        self.loop.save(update_fields=["history_mode"])
        captured = {}
        mock_get_service.return_value = self._capture_service(captured)
        from chat.loop_service import execute_loop_run

        execute_loop_run(self.loop.id)
        blob = " ".join(str(m.content) for m in captured["messages"])
        self.assertIn("Earlier reply.", blob)
        self.assertIn("Summarize today.", blob)

    @patch("llm.get_llm_service")
    def test_loop_framing_rides_on_last_user_message(self, mock_get_service):
        """A loop turn frames its prompt as an unattended standing instruction
        in the last user message — not in the cached system prompt — so the
        model executes instead of pushing back / asking what the real ask is."""
        captured = {}
        mock_get_service.return_value = self._capture_service(captured)
        from chat.loop_service import execute_loop_run

        execute_loop_run(self.loop.id)
        messages = captured["messages"]

        system = next(m for m in messages if m.role == "system")
        self.assertNotIn("Scheduled Loop Task", system.content)

        last_user = [m for m in messages if m.role == "user"][-1]
        self.assertIn("Scheduled Loop Task", last_user.content)
        self.assertIn("# Loop instructions", last_user.content)
        self.assertIn("do not ask for clarification", last_user.content.lower())
        # The actual loop prompt follows the framing.
        self.assertIn("Summarize today.", last_user.content)
        self.assertLess(
            last_user.content.index("# Loop instructions"),
            last_user.content.index("Summarize today."),
        )

    @patch("llm.get_llm_service")
    def test_uses_thread_model(self, mock_get_service):
        from core.preferences import get_preferences

        model = get_preferences(self.user).allowed_models[0]
        self.thread.model = model
        self.thread.save(update_fields=["model"])
        captured = {}

        service = MagicMock()

        async def astream(mode, request, **kwargs):
            captured["model"] = request.model
            return
            yield

        service.astream = astream
        mock_get_service.return_value = service
        from chat.loop_service import execute_loop_run

        execute_loop_run(self.loop.id)
        self.assertEqual(captured["model"], model)

    def test_error_increments_and_releases_lock(self):
        from chat import loop_service

        with patch.object(
            loop_service.HeadlessTurnRunner, "run_loop_turn",
            new_callable=AsyncMock, side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(Exception):
                loop_service.execute_loop_run(self.loop.id)

        self.loop.refresh_from_db()
        self.assertEqual(self.loop.consecutive_errors, 1)
        self.assertFalse(self.loop.running)  # lock released even on error
        self.assertIsNone(self.loop.locked_at)
