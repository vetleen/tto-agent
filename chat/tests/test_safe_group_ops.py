"""Tests for the consumer's resilient channel-layer group ops.

``ChatConsumer._safe_group_add`` / ``_safe_group_discard`` must degrade to a
WARNING (not crash the turn) when the shared Redis Mini resets the channels-layer
connection — the thread broadcast group is best-effort (it only routes background
sub-agent completion notifications; the user's own turn streams via direct
``self.send``). Regression guard for WILFRED-6P.
"""

import logging

from django.test import SimpleTestCase
from redis.exceptions import ConnectionError as RedisConnectionError

from chat.consumers import ChatConsumer
from chat.sinks import BroadcastSink


class _RaisingLayer:
    """channel_layer whose group ops raise a redis blip, like the saturated Mini."""

    async def group_add(self, group, channel):
        raise RedisConnectionError("Error connecting to redis:26480")

    async def group_discard(self, group, channel):
        raise RedisConnectionError("Error connecting to redis:26480")


class _RecordingLayer:
    def __init__(self):
        self.calls = []

    async def group_add(self, group, channel):
        self.calls.append(("add", group, channel))

    async def group_discard(self, group, channel):
        self.calls.append(("discard", group, channel))


def _consumer(layer):
    c = ChatConsumer()
    c.channel_name = "chan.1"
    c.channel_layer = layer
    return c


class SafeGroupOpTests(SimpleTestCase):
    async def test_group_add_swallows_redis_blip(self):
        c = _consumer(_RaisingLayer())
        # Must not raise — a transient blip degrades, it does not crash the turn.
        await c._safe_group_add("thread_1")

    async def test_group_discard_swallows_redis_blip(self):
        c = _consumer(_RaisingLayer())
        await c._safe_group_discard("thread_1")

    async def test_group_add_happy_path_calls_through(self):
        layer = _RecordingLayer()
        await _consumer(layer)._safe_group_add("thread_1")
        self.assertEqual(layer.calls, [("add", "thread_1", "chan.1")])

    async def test_group_discard_happy_path_calls_through(self):
        layer = _RecordingLayer()
        await _consumer(layer)._safe_group_discard("thread_9")
        self.assertEqual(layer.calls, [("discard", "thread_9", "chan.1")])


class _SendLayer:
    """channel_layer whose ``group_send`` fails while ``fail`` is set."""

    def __init__(self, exc=None):
        self.exc = exc

    async def group_send(self, group, message):
        if self.exc is not None:
            raise self.exc


def _sink(layer):
    sink = BroadcastSink.__new__(BroadcastSink)  # bypass get_channel_layer()
    sink._group = "thread_1"
    sink._channel_layer = layer
    sink._redis_blip_reported = False
    return sink


class BroadcastSinkLoggingTests(SimpleTestCase):
    """``send_event`` runs per streamed token, so its reporting must be throttled.

    A Redis outage should put ONE event on the Sentry board per sink, not one per
    token — but it must still put one there, which the old blanket DEBUG did not.
    """

    async def test_first_redis_blip_warns(self):
        sink = _sink(_SendLayer(RedisConnectionError("boom")))
        with self.assertLogs("chat.sinks", level="DEBUG") as cm:
            await sink.send_event({"type": "token"})
        self.assertEqual(cm.records[0].levelno, logging.WARNING)

    async def test_repeat_blips_are_throttled_to_debug(self):
        sink = _sink(_SendLayer(RedisConnectionError("boom")))
        with self.assertLogs("chat.sinks", level="DEBUG") as cm:
            for _ in range(5):
                await sink.send_event({"type": "token"})
        levels = [r.levelno for r in cm.records]
        self.assertEqual(levels, [logging.WARNING] + [logging.DEBUG] * 4)

    async def test_recovery_rearms_the_warning(self):
        """A later outage must be reported again, not swallowed forever."""
        layer = _SendLayer(RedisConnectionError("boom"))
        sink = _sink(layer)
        with self.assertLogs("chat.sinks", level="DEBUG") as cm:
            await sink.send_event({"type": "token"})  # WARNING
            layer.exc = None
            await sink.send_event({"type": "token"})  # succeeds, clears the flag
            layer.exc = RedisConnectionError("boom")
            await sink.send_event({"type": "token"})  # WARNING again
        levels = [r.levelno for r in cm.records]
        self.assertEqual(levels, [logging.WARNING, logging.WARNING])

    async def test_non_redis_failure_stays_debug(self):
        """A missing group or bad payload is not an outage — keep it quiet."""
        sink = _sink(_SendLayer(ValueError("bad message")))
        with self.assertLogs("chat.sinks", level="DEBUG") as cm:
            await sink.send_event({"type": "token"})
        self.assertEqual(cm.records[0].levelno, logging.DEBUG)

    async def test_send_event_never_raises(self):
        """Whatever happens, a failed broadcast must not break the turn."""
        sink = _sink(_SendLayer(RedisConnectionError("boom")))
        with self.assertLogs("chat.sinks", level="DEBUG"):
            await sink.send_event({"type": "token"})
