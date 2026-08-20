"""Tests for the consumer's resilient channel-layer group ops.

``ChatConsumer._safe_group_add`` / ``_safe_group_discard`` must degrade to a
WARNING (not crash the turn) when the shared Redis Mini resets the channels-layer
connection — the thread broadcast group is best-effort (it only routes background
sub-agent completion notifications; the user's own turn streams via direct
``self.send``). Regression guard for WILFRED-6P.
"""

from django.test import SimpleTestCase
from redis.exceptions import ConnectionError as RedisConnectionError

from chat.consumers import ChatConsumer


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
