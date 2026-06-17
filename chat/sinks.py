"""Output sinks for chat turn streaming.

A turn produces a stream of events (tokens, tool_start/tool_end, canvas/task
side-effects, errors). Interactive chat delivers them over the WebSocket; a
headless loop turn broadcasts them to the thread's channel group so a connected
browser can render them, or discards them. The turn-execution code in
``ChatConsumer`` writes to ``self._sink.send_event(...)`` instead of calling
``self.send(...)`` directly, so the same code path serves all three callers.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


class WebSocketSink:
    """Delivers events to the connected browser over the WebSocket.

    This is the interactive default; it preserves the consumer's original
    behaviour (``await self.send(text_data=json.dumps(event))``).
    """

    wants_heartbeats = True

    def __init__(self, consumer):
        self._consumer = consumer

    async def send_event(self, event: dict) -> None:
        await self._consumer.send(text_data=json.dumps(event))


class BroadcastSink:
    """Publishes events to a thread's channel group for any connected viewers.

    Used by headless loop turns. Each event is wrapped as a ``loop.event``
    channel message; ``ChatConsumer.loop_event`` unwraps it and forwards the
    inner event to its own socket — but only if that consumer is currently
    viewing the thread. Best-effort: a missing channel layer or a send failure
    must never break the turn (nobody may be watching).
    """

    wants_heartbeats = False

    def __init__(self, thread_id: str):
        from channels.layers import get_channel_layer

        self._group = f"thread_{thread_id}"
        self._channel_layer = get_channel_layer()

    async def send_event(self, event: dict) -> None:
        if self._channel_layer is None:
            return
        try:
            await self._channel_layer.group_send(
                self._group, {"type": "loop.event", "event": event},
            )
        except Exception:
            logger.debug("BroadcastSink: could not publish event to %s", self._group)


class NullSink:
    """Discards all events — persistence-only turn execution."""

    wants_heartbeats = False

    async def send_event(self, event: dict) -> None:
        return
