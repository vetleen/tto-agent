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

from core.redis_errors import is_redis_blip, log_broadcast_failure

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
        # Whether a Redis blip on this sink has already been reported; see below.
        self._redis_blip_reported = False

    async def send_event(self, event: dict) -> None:
        if self._channel_layer is None:
            return
        try:
            await self._channel_layer.group_send(
                self._group, {"type": "loop.event", "event": event},
            )
        except Exception as exc:
            # This runs once per streamed event — per token — so reporting every
            # Redis blip at WARNING would put thousands of Sentry events on the
            # board for a single outage. Report the first blip of an episode and
            # go quiet until a send succeeds again, which keeps the signal (an
            # outage is visible) without the flood.
            log_broadcast_failure(
                logger,
                exc,
                "BroadcastSink: could not publish event to %s",
                self._group,
                redis_level=(
                    logging.DEBUG if self._redis_blip_reported else logging.WARNING
                ),
            )
            if is_redis_blip(exc):
                self._redis_blip_reported = True
        else:
            self._redis_blip_reported = False


class NullSink:
    """Discards all events — persistence-only turn execution."""

    wants_heartbeats = False

    async def send_event(self, event: dict) -> None:
        return
