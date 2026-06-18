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
        self.active_skill_ids = []
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

    async def run_loop_turn(self, thread, content, *, history_mode, data_room_ids, active_skill_ids):
        """Populate context, persist the loop prompt, and run one full turn."""
        # Resolve org customization + preferences for this user (same helpers
        # the live consumer uses on connect / per message).
        await self._load_membership()
        self.resolved_prefs = await self._resolve_preferences()
        self.data_room_ids = data_room_ids
        self.active_skill_ids = active_skill_ids

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
    from chat.models import ChatThreadSkill, Loop

    loop = Loop.objects.select_related("thread", "created_by").get(pk=loop_id)
    fire_time = timezone.now()

    try:
        sink = BroadcastSink(str(loop.thread_id))
        runner = HeadlessTurnRunner(
            user=loop.created_by, thread_id=loop.thread_id, sink=sink,
        )
        data_room_ids = list(loop.thread.data_rooms.values_list("pk", flat=True))
        active_skill_ids = [
            str(sid) for sid in ChatThreadSkill.objects.filter(
                thread=loop.thread
            ).values_list("skill_id", flat=True)
        ]

        async_to_sync(runner.run_loop_turn)(
            loop.thread, loop.prompt,
            history_mode=loop.history_mode,
            data_room_ids=data_room_ids,
            active_skill_ids=active_skill_ids,
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
        if loop.status == Loop.Status.PAUSED:
            _mirror_archive_to_thread(loop, True)

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
        if loop.status == Loop.Status.PAUSED:
            _mirror_archive_to_thread(loop, True)
        raise

    finally:
        Loop.objects.filter(pk=loop_id).update(running=False, locked_at=None)


# ---------------------------------------------------------------------------
# Create / edit / lifecycle orchestration
#
# The payload-validation helpers (``_build_loop_fields`` etc.) and the
# high-level ``create_loop`` / ``update_loop`` / ``pause_loop`` / ``resume_loop``
# / ``restart_loop`` functions are the single source of truth shared by the HTTP views
# (``chat/views.py``) and the agent tools (``chat/tool_loops.py``). They take
# resolved primitives (a user, a parsed ``body`` dict, ``now``, ``tz_name``) and
# never touch an HTTP request.
# ---------------------------------------------------------------------------


def _parse_int(value, *, default, lo, hi):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(n, hi))


def _parse_hhmm(value):
    """Parse 'HH:MM' into a datetime.time, or None."""
    from datetime import time as _time

    try:
        hh, mm = str(value).split(":")
        return _time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return None


def _parse_local_dt(value, tz_name, now):
    """Parse a 'YYYY-MM-DDTHH:MM' datetime-local string in tz into aware UTC."""
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo

    if not value:
        return None
    try:
        naive = _dt.strptime(value[:16], "%Y-%m-%dT%H:%M")
    except (ValueError, TypeError):
        return None
    return naive.replace(tzinfo=ZoneInfo(tz_name)).astimezone(ZoneInfo("UTC"))


def _build_loop_fields(body, *, now, tz_name):
    """Validate + normalize a loop create/edit payload into model field values.

    Returns ``(fields, was_reduced, errors)``. ``fields`` is a dict ready to
    splat onto a Loop; ``errors`` is a list of user-facing strings.
    """
    from chat.loop_schedule import (
        DEFAULT_MAX_RUNS, clamp_max_runs, compute_next_run,
    )
    from chat.models import Loop

    errors = []
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        errors.append("Please enter a prompt for the loop to run.")

    history_mode = body.get("history_mode")
    if history_mode not in (Loop.HistoryMode.FRESH, Loop.HistoryMode.CONVERSATIONAL):
        history_mode = Loop.HistoryMode.FRESH

    cadence_kind = body.get("cadence_kind")
    interval_seconds = None
    clock_time = None
    clock_frequency = ""
    clock_weekday = None

    if cadence_kind == "clock":
        clock_frequency = body.get("clock_frequency")
        if clock_frequency not in ("daily", "weekdays", "weekly"):
            clock_frequency = "daily"
        clock_time = _parse_hhmm(body.get("clock_time"))
        if clock_time is None:
            errors.append("Please choose a valid time of day.")
            from datetime import time as _time
            clock_time = _time(9, 0)
        if clock_frequency == "weekly":
            clock_weekday = _parse_int(body.get("clock_weekday"), default=0, lo=0, hi=6)
    else:
        cadence_kind = "interval"
        unit = body.get("interval_unit", "hours")
        value = _parse_int(body.get("interval_value"), default=1, lo=1, hi=100000)
        mult = {"minutes": 60, "hours": 3600, "days": 86400}.get(unit, 3600)
        interval_seconds = max(60, value * mult)  # floor at 1 minute

    sched = {
        "cadence_kind": cadence_kind,
        "interval_seconds": interval_seconds,
        "clock_time": clock_time,
        "clock_frequency": clock_frequency or None,
        "clock_weekday": clock_weekday,
        "tz": tz_name,
    }

    # First run: keep existing (edit only), Now (default), the next scheduled
    # occurrence, or a custom time.
    first_run_mode = body.get("first_run_mode", "now")
    if first_run_mode == "keep":
        next_run = None  # caller preserves the loop's existing next_run
    elif first_run_mode == "custom":
        next_run = _parse_local_dt(body.get("first_run_at"), tz_name, now) or now
        if next_run < now:
            next_run = now
    elif first_run_mode == "scheduled" and cadence_kind == "clock":
        next_run = compute_next_run(now, **sched)
    else:
        next_run = now

    # Run count is fixed policy (DEFAULT_MAX_RUNS), not taken from the payload —
    # customization was removed. clamp_max_runs still trims it so the last run
    # lands within a year; ``was_reduced`` then reflects only that year-horizon
    # trim.
    effective_max, was_reduced = clamp_max_runs(DEFAULT_MAX_RUNS, now, **sched)

    fields = {
        "prompt": prompt,
        "history_mode": history_mode,
        "cadence_kind": cadence_kind,
        "interval_seconds": interval_seconds,
        "clock_time": clock_time,
        "clock_frequency": clock_frequency or "",
        "clock_weekday": clock_weekday,
        "tz": tz_name,
        "next_run": next_run,
        "max_runs": effective_max,
    }
    return fields, was_reduced, errors


def _link_loop_resources(thread, user, data_room_ids, skill_ids, model=None):
    """Attach validated data rooms + skills + model to a loop's thread (idempotent)."""
    from agent_skills.models import MAX_THREAD_SKILLS
    from chat.models import ChatThreadDataRoom, ChatThreadSkill
    from core.preferences import get_preferences
    from documents.models import DataRoom

    prefs = get_preferences(user)
    valid_pks = list(
        DataRoom.objects.filter(
            created_by=user, pk__in=data_room_ids or [], is_archived=False,
        ).values_list("pk", flat=True)
    )
    ChatThreadDataRoom.objects.filter(thread=thread).delete()
    ChatThreadDataRoom.objects.bulk_create(
        [ChatThreadDataRoom(thread=thread, data_room_id=pk) for pk in valid_pks]
    )

    # Validate each requested skill against the user's allowed set, preserving
    # order and dropping duplicates, then cap. Replace the thread's skill set.
    allowed = {str(s["id"]) for s in prefs.allowed_skills}
    resolved_skill_ids: list[str] = []
    for sid in (skill_ids or []):
        sid = str(sid)
        if sid in allowed and sid not in resolved_skill_ids:
            resolved_skill_ids.append(sid)
    resolved_skill_ids = resolved_skill_ids[:MAX_THREAD_SKILLS]
    ChatThreadSkill.objects.filter(thread=thread).delete()
    ChatThreadSkill.objects.bulk_create(
        [ChatThreadSkill(thread=thread, skill_id=sid) for sid in resolved_skill_ids]
    )

    # Per-thread model: store only an allowed model; otherwise clear so it
    # resolves to the user's preferred chat model.
    thread.model = model if (model and model in prefs.allowed_models) else ""
    thread.save(update_fields=["model"])


def _loop_form_json(loop, prefs):
    """Serialize a loop's editable fields for the create/edit modal prefill."""
    from chat.models import ChatThreadSkill
    from core.preferences import resolve_thread_model

    return {
        "id": str(loop.id),
        "status": loop.status,
        "prompt": loop.prompt,
        "history_mode": loop.history_mode,
        "cadence_kind": loop.cadence_kind,
        "interval_seconds": loop.interval_seconds,
        "clock_time": loop.clock_time.strftime("%H:%M") if loop.clock_time else "",
        "clock_frequency": loop.clock_frequency,
        "clock_weekday": loop.clock_weekday,
        "max_runs": loop.max_runs,
        "data_room_ids": list(loop.thread.data_rooms.values_list("pk", flat=True)),
        "skill_ids": [
            str(sid) for sid in ChatThreadSkill.objects.filter(
                thread=loop.thread
            ).values_list("skill_id", flat=True)
        ],
        "model": resolve_thread_model(loop.thread.model, prefs),
    }


def create_loop(*, user, body, now, tz_name):
    """Validate a payload, create a backing thread + Loop for ``user``.

    Returns ``(loop, was_reduced, errors)``. On validation error ``loop`` is
    ``None`` and ``errors`` is non-empty.
    """
    from django.db import transaction

    from chat.models import ChatThread, Loop

    fields, was_reduced, errors = _build_loop_fields(body, now=now, tz_name=tz_name)
    if errors:
        return None, was_reduced, errors
    if fields["next_run"] is None:  # "keep" has no meaning on create
        fields["next_run"] = now

    with transaction.atomic():
        thread = ChatThread.objects.create(
            created_by=user, title=fields["prompt"][:80] or "Scheduled loop",
        )
        _link_loop_resources(
            thread, user, body.get("data_room_ids"), body.get("skill_ids"),
            model=body.get("model"),
        )
        loop = Loop.objects.create(thread=thread, created_by=user, **fields)
    return loop, was_reduced, []


def update_loop(*, loop, user, body, now, tz_name):
    """Validate a payload and apply it to an existing ``loop``.

    The caller is responsible for fetching ``loop`` and checking ownership.
    Honors ``body['restart']`` (re-activate + reset run count). Returns
    ``(loop, was_reduced, errors)``; ``loop`` is ``None`` on validation error.
    """
    from chat.models import Loop

    fields, was_reduced, errors = _build_loop_fields(body, now=now, tz_name=tz_name)
    if errors:
        return None, was_reduced, errors

    _link_loop_resources(
        loop.thread, user, body.get("data_room_ids"), body.get("skill_ids"),
        model=body.get("model"),
    )
    for key, val in fields.items():
        if key == "next_run" and val is None:
            continue  # "keep" — preserve the existing schedule time
        setattr(loop, key, val)

    # Restarting a paused loop: re-activate, start the run count over, release
    # any stale lock.
    if body.get("restart"):
        loop.status = Loop.Status.ACTIVE
        loop.running = False
        loop.locked_at = None
        loop.runs_completed = 0
        loop.consecutive_errors = 0
        if fields["next_run"] is None:  # "keep" → start now on restart
            loop.next_run = now

    loop.thread.title = fields["prompt"][:80] or loop.thread.title
    loop.thread.save(update_fields=["title"])
    loop.save()
    if body.get("restart"):
        _mirror_archive_to_thread(loop, False)
    return loop, was_reduced, []


def _mirror_archive_to_thread(loop, archived):
    """Keep a loop's backing thread archived iff the loop is paused.

    A loop's paused state and its thread's archived state move together: a
    paused loop reads as archived in the chat sidebar, and reviving it restores
    the thread. Pausing archives; resuming / restarting unarchives. The guard
    makes this idempotent so callers that already toggled the thread (e.g. the
    archive view) don't re-save it.
    """
    thread = loop.thread
    if thread.is_archived != archived:
        thread.is_archived = archived
        thread.save(update_fields=["is_archived"])


def pause_loop(loop):
    """Pause a loop so it stops firing, and archive its thread.

    Reversible via :func:`restart_loop` (the user-facing action — resets the run
    count) or :func:`resume_loop` (the agent's resume-where-it-left-off path),
    both of which unarchive the thread again.
    """
    from chat.models import Loop

    loop.status = Loop.Status.PAUSED
    loop.save(update_fields=["status", "updated_at"])
    _mirror_archive_to_thread(loop, True)
    return loop


def resume_loop(loop, now):
    """Resume a paused loop where it left off: fire on the next tick, keep the
    completed-run count, then continue the cadence.

    This preserves ``runs_completed``; it is the agent-only ``resume=true``
    behaviour. The user-facing revive action is :func:`restart_loop`, which
    starts the run count over.
    """
    from chat.models import Loop

    loop.status = Loop.Status.ACTIVE
    loop.next_run = now
    loop.running = False
    loop.locked_at = None
    # A resumed loop that already hit its cap would re-pause immediately; give it
    # at least one more run.
    if loop.runs_completed >= loop.max_runs:
        loop.max_runs = loop.runs_completed + 1
    loop.save(update_fields=[
        "status", "next_run", "running", "locked_at", "max_runs", "updated_at",
    ])
    _mirror_archive_to_thread(loop, False)
    return loop


def restart_loop(loop, now):
    """Restart a paused loop from scratch: re-activate, reset the run count to 0,
    clear any error streak, and fire on the next tick.

    This is the single user-facing revive action (the Loops page "Restart"
    control) and matches the ``restart`` branch of :func:`update_loop`.
    """
    from chat.models import Loop

    loop.status = Loop.Status.ACTIVE
    loop.next_run = now
    loop.running = False
    loop.locked_at = None
    loop.runs_completed = 0
    loop.consecutive_errors = 0
    loop.save(update_fields=[
        "status", "next_run", "running", "locked_at",
        "runs_completed", "consecutive_errors", "updated_at",
    ])
    _mirror_archive_to_thread(loop, False)
    return loop


def list_loops_for_user(user):
    """Return the user's active + paused loops, most-recent-result first."""
    from django.db.models import F

    from chat.models import Loop

    return list(
        Loop.objects.filter(
            created_by=user,
            status__in=[Loop.Status.ACTIVE, Loop.Status.PAUSED],
        )
        .select_related("thread")
        .order_by(F("last_result_at").desc(nulls_last=True), "-created_at")
    )
