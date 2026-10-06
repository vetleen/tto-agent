"""The interactive sink must never end a turn because the socket died.

A send on a half-dead WebSocket raises (daphne: "Attempt to send on a closed
protocol"). That used to escape ``WebSocketSink.send_event`` and abort the
stream — the reply was never persisted and the model ran on for nobody. The
sink now swallows the failure once, tells the consumer to detach, and drops
every later event.
"""

import logging

from django.test import SimpleTestCase

from chat.sinks import WebSocketSink


class _Consumer:
    """Only what the sink touches: ``send`` and the detach hook."""

    def __init__(self, deliver=0):
        self.deliver = deliver  # sends that succeed before the socket "dies"
        self.attempts = 0
        self.lost = []

    async def send(self, text_data=None):
        self.attempts += 1
        if self.attempts > self.deliver:
            raise RuntimeError("Attempt to send on a closed protocol")

    async def _socket_lost(self, exc):
        self.lost.append(exc)


class _HooklessConsumer:
    async def send(self, text_data=None):
        raise RuntimeError("closed")


class WebSocketSinkTests(SimpleTestCase):
    async def test_failed_send_detaches_once_and_drops_later_events(self):
        consumer = _Consumer(deliver=1)
        sink = WebSocketSink(consumer)

        await sink.send_event({"event_type": "token"})  # delivered
        self.assertEqual(consumer.lost, [])

        with self.assertLogs("chat.sinks", level="INFO") as cm:
            await sink.send_event({"event_type": "token"})  # socket gone: no raise
        self.assertEqual([r.levelno for r in cm.records], [logging.INFO])
        self.assertEqual(len(consumer.lost), 1)
        self.assertIsInstance(consumer.lost[0], RuntimeError)

        # Everything after the loss is dropped without touching the socket again.
        await sink.send_event({"event_type": "message_end"})
        self.assertEqual(consumer.attempts, 2)
        self.assertEqual(len(consumer.lost), 1)

    async def test_consumer_without_hook_still_survives(self):
        sink = WebSocketSink(_HooklessConsumer())
        with self.assertLogs("chat.sinks", level="INFO"):
            await sink.send_event({"event_type": "token"})
        await sink.send_event({"event_type": "token"})  # dropped, no raise
