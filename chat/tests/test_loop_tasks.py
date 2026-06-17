"""Tests for the tick-and-scan loop scheduler (chat.loop_service.enqueue_due_loops)."""

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from chat.loop_service import LOOP_STALE_LOCK_MINUTES, enqueue_due_loops
from chat.models import ChatThread, Loop

User = get_user_model()


class BeatScheduleTests(SimpleTestCase):
    def test_tick_and_scan_registered_in_beat(self):
        from django.conf import settings

        entry = settings.CELERY_BEAT_SCHEDULE.get("tick-and-scan-loops")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["task"], "chat.tasks.tick_and_scan_loops")


class EnqueueDueLoopsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="loop@example.com", password="pass123",
        )

    def _make_loop(self, **kw):
        thread = ChatThread.objects.create(created_by=self.user)
        defaults = dict(
            thread=thread, created_by=self.user, prompt="x",
            cadence_kind=Loop.Cadence.INTERVAL, interval_seconds=3600,
            next_run=timezone.now() - timedelta(seconds=10),
            status=Loop.Status.ACTIVE,
        )
        defaults.update(kw)
        return Loop.objects.create(**defaults)

    @patch("chat.tasks.run_loop.delay")
    def test_due_loop_is_enqueued_and_claimed(self, mock_delay):
        loop = self._make_loop()
        n = enqueue_due_loops()
        self.assertEqual(n, 1)
        mock_delay.assert_called_once_with(str(loop.id))
        loop.refresh_from_db()
        self.assertTrue(loop.running)
        self.assertIsNotNone(loop.locked_at)

    @patch("chat.tasks.run_loop.delay")
    def test_future_loop_not_enqueued(self, mock_delay):
        self._make_loop(next_run=timezone.now() + timedelta(hours=1))
        self.assertEqual(enqueue_due_loops(), 0)
        mock_delay.assert_not_called()

    @patch("chat.tasks.run_loop.delay")
    def test_paused_loop_not_enqueued(self, mock_delay):
        self._make_loop(status=Loop.Status.PAUSED)
        self.assertEqual(enqueue_due_loops(), 0)
        mock_delay.assert_not_called()

    @patch("chat.tasks.run_loop.delay")
    def test_running_loop_not_re_enqueued(self, mock_delay):
        # Fresh lock — a turn is in flight; reentrancy guard must skip it.
        self._make_loop(running=True, locked_at=timezone.now())
        self.assertEqual(enqueue_due_loops(), 0)
        mock_delay.assert_not_called()

    @patch("chat.tasks.run_loop.delay")
    def test_stale_lock_is_reclaimed_and_enqueued(self, mock_delay):
        stale = timezone.now() - timedelta(minutes=LOOP_STALE_LOCK_MINUTES + 5)
        loop = self._make_loop(running=True, locked_at=stale)
        n = enqueue_due_loops()
        self.assertEqual(n, 1)
        mock_delay.assert_called_once_with(str(loop.id))
        loop.refresh_from_db()
        self.assertTrue(loop.running)  # re-claimed for this run
