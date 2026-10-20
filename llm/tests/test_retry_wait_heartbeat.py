"""The provider retry wait keeps the sub-agent heartbeat alive
(llm/core/providers/base.py: _wait_before_retry / _activity_hook).

A 30/60/120 s backoff is progress, not silence: without a heartbeat during it
the stale-run sweeper (chat.subagent_limits) could expire a run that is merely
waiting out a rate limit.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from llm.core.providers.base import _activity_hook, _wait_before_retry
from llm.types import ChatRequest, Message, RunContext


class ActivityHookTests(SimpleTestCase):
    def _request(self, context):
        return ChatRequest(messages=[Message(role="user", content="x")], context=context)

    def test_hook_is_the_contexts_note_activity(self):
        ctx = RunContext.create()
        beats = []
        ctx.on_activity = lambda: beats.append(1)
        hook = _activity_hook(self._request(ctx))
        hook()
        self.assertEqual(beats, [1])

    def test_no_context_means_no_hook(self):
        self.assertIsNone(_activity_hook(self._request(None)))


class WaitBeforeRetryTickTests(SimpleTestCase):
    @patch("llm.core.providers.base.time.sleep")
    def test_sliced_wait_ticks_every_slice(self, _sleep):
        on_tick = MagicMock()
        ok = _wait_before_retry(3.0, cancel_check=lambda: False, on_tick=on_tick)
        self.assertTrue(ok)
        self.assertEqual(on_tick.call_count, 3)

    @patch("llm.core.providers.base.time.sleep")
    def test_cancel_stops_the_wait_before_ticking(self, _sleep):
        on_tick = MagicMock()
        ok = _wait_before_retry(3.0, cancel_check=lambda: True, on_tick=on_tick)
        self.assertFalse(ok)
        on_tick.assert_not_called()

    @patch("llm.core.providers.base.time.sleep")
    def test_plain_wait_without_cancel_controls_needs_no_tick(self, sleep):
        self.assertTrue(_wait_before_retry(2.0))
        sleep.assert_called_once_with(2.0)
