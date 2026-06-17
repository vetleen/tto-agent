"""Headless execution and scheduling for Loops.

``enqueue_due_loops`` is the tick-and-scan selector (run from a beat task); it
claims each due loop with an atomic CAS on the ``running`` flag so a turn still
in flight is never double-fired. ``execute_loop_run`` runs one full agent turn
headlessly by driving a ``ChatConsumer`` subclass with no socket — output goes
to a ``BroadcastSink`` so a browser viewing the thread can watch it live.
"""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta

from asgiref.sync import async_to_sync
from django.utils import timezone

from chat.consumers import ChatConsumer
from chat.loop_schedule import next_run_for_loop
from chat.sinks import BroadcastSink

logger = logging.getLogger(__name__)

# A loop whose lock is older than this is assumed orphaned (worker died
# mid-run) and is reclaimed by the next scan.
LOOP_STALE_LOCK_MINUTES = 20
# Auto-pause a loop after this many consecutive failed runs.
LOOP_MAX_CONSECUTIVE_ERRORS = 3


class HeadlessTurnRunner(ChatConsumer):
    """Drives a ``ChatConsumer``'s turn machinery without a WebSocket.

    Reuses the full interactive turn path (prompt assembly, history loading,
    streaming, persistence, canvas/tool side-effects) but routes output to a
    sink. It is never mounted as an ASGI app and ``connect()`` is never called —
    it is instantiated directly, its context attributes populated from the user,
    then ``run_loop_turn`` is awaited.
    """

    def __init__(self, *, user, thread_id, sink):
        from channels.layers import get_channel_layer

        # Mirror the attributes ChatConsumer.connect() would set. We do NOT call
        # super().__init__/connect — there is no socket.
        self.user = user
        self.resolved_prefs = None
        self.data_room_ids = []
        self.active_skill_id = None
        self._cancel_event = None
        self._turn = None
        self._guardrail_task = None
        self._org_id = None
        self._org_name = None
        self._soul = None
        self._org_description = None
        self._user_context = None
        self._current_thread_id = str(thread_id)
        self._active_thread_id = str(thread_id)
        self._stopped = False
        self._stream_task = None
        self._sink = sink
        self.channel_layer = get_channel_layer()
        self.channel_name = None

    async def run_loop_turn(self, thread, content, *, history_mode, data_room_ids, active_skill_id):
        """Populate context, persist the loop prompt, and run one full turn."""
        # Resolve org customization + preferences for this user (same helpers
        # the live consumer uses on connect / per message).
        await self._load_membership()
        self.resolved_prefs = await self._resolve_preferences()
        self.data_room_ids = data_room_ids
        self.active_skill_id = active_skill_id

        # Persist the loop's prompt as a real, visible user message and let any
        # connected viewer render it before the assistant turn begins.
        await self._create_message(
            thread, "user", content, metadata={"loop_run": True},
        )
        await self._sink.send_event({"event_type": "loop.user_message", "content": content})

        await self.run_turn_to_completion(
            thread, content, history_mode=history_mode, is_loop_turn=True,
        )


def enqueue_due_loops() -> int:
    """Find active, due, not-running loops and enqueue a run for each.

    Reclaims stale locks first, then claims each loop with an atomic
    ``running=False → True`` UPDATE so only the winner enqueues — this is the
    reentrancy guard. Returns the number of runs enqueued.
    """
    from chat.models import Loop
    from chat.tasks import run_loop

    now = timezone.now()
    stale_cutoff = now - timedelta(minutes=LOOP_STALE_LOCK_MINUTES)
    Loop.objects.filter(running=True, locked_at__lt=stale_cutoff).update(
        running=False, locked_at=None,
    )

    due_ids = list(
        Loop.objects.filter(
            status=Loop.Status.ACTIVE, next_run__lte=now, running=False,
        ).values_list("id", flat=True)
    )

    count = 0
    for loop_id in due_ids:
        claimed = Loop.objects.filter(
            pk=loop_id, running=False, status=Loop.Status.ACTIVE,
        ).update(running=True, locked_at=now)
        if claimed:
            run_loop.delay(str(loop_id))
            count += 1
    return count


def execute_loop_run(loop_id: uuid.UUID) -> None:
    """Run one scheduled loop turn headlessly, then reschedule / bookkeep.

    Always releases the running lock in ``finally``. Advances ``next_run`` from
    the fire time (so cadence doesn't drift with run duration), auto-pauses at
    the run cap, and auto-pauses after too many consecutive errors.
    """
    from chat.models import Loop

    loop = Loop.objects.select_related("thread", "created_by").get(pk=loop_id)
    fire_time = timezone.now()

    try:
        sink = BroadcastSink(str(loop.thread_id))
        runner = HeadlessTurnRunner(
            user=loop.created_by, thread_id=loop.thread_id, sink=sink,
        )
        data_room_ids = list(loop.thread.data_rooms.values_list("pk", flat=True))
        active_skill_id = str(loop.thread.skill_id) if loop.thread.skill_id else None

        async_to_sync(runner.run_loop_turn)(
            loop.thread, loop.prompt,
            history_mode=loop.history_mode,
            data_room_ids=data_room_ids,
            active_skill_id=active_skill_id,
        )

        loop.runs_completed += 1
        loop.last_result_at = timezone.now()
        loop.consecutive_errors = 0
        loop.next_run = next_run_for_loop(loop, fire_time)
        update_fields = [
            "runs_completed", "last_result_at", "consecutive_errors",
            "next_run", "updated_at",
        ]
        if loop.runs_completed >= loop.max_runs:
            loop.status = Loop.Status.PAUSED
            update_fields.append("status")
        loop.save(update_fields=update_fields)

    except Exception:
        logger.exception("Loop %s run failed", loop_id)
        loop.consecutive_errors += 1
        update_fields = ["consecutive_errors", "updated_at"]
        if loop.consecutive_errors >= LOOP_MAX_CONSECUTIVE_ERRORS:
            loop.status = Loop.Status.PAUSED
            update_fields.append("status")
        else:
            # Reschedule and try again on the next cadence tick.
            loop.next_run = next_run_for_loop(loop, fire_time)
            update_fields.append("next_run")
        loop.save(update_fields=update_fields)
        raise

    finally:
        Loop.objects.filter(pk=loop_id).update(running=False, locked_at=None)
