"""Loop-management tools: create / list / stop / edit scheduled chat loops.

These let the in-chat agent manage Loops — recurring, scheduled chat turns bound
1:1 to their own thread. All business logic is shared with the HTTP views via
``chat/loop_service.py``; these tools just translate the agent's typed arguments
into the payload those service functions expect.
"""

from __future__ import annotations

import json
import uuid as uuid_mod

from pydantic import BaseModel, Field

from llm.tools import ContextAwareTool, ReasonBaseModel, get_tool_registry


def _now():
    from django.utils import timezone

    return timezone.now()


def _default_tz() -> str:
    from django.conf import settings

    return settings.TIME_ZONE


def _get_user(context):
    """Resolve the acting user from run context. Returns ``(user, error_json)``."""
    from django.contrib.auth import get_user_model

    user_id = context.user_id if context else None
    if not user_id:
        return None, json.dumps({"status": "error", "message": "No user context available."})
    User = get_user_model()
    try:
        return User.objects.get(pk=user_id), None
    except User.DoesNotExist:
        return None, json.dumps({"status": "error", "message": "User not found."})


def _loop_summary(loop) -> dict:
    """Compact, JSON-serializable view of a loop for tool results.

    Uses ``loop_status`` (not ``status``) so spreading this into a result dict
    never clobbers the tool's own ``status`` field.
    """
    return {
        "loop_id": str(loop.id),
        "prompt": loop.prompt,
        "schedule_label": loop.schedule_label,
        "loop_status": loop.status,
        "history_mode": loop.history_mode,
        "next_run": loop.next_run.isoformat() if loop.next_run else None,
        "runs_completed": loop.runs_completed,
        "max_runs": loop.max_runs,
        "thread_id": str(loop.thread_id),
        "is_unread": loop.is_unread,
    }


def _interval_unit_value(interval_seconds: int | None) -> tuple[str, int]:
    """Express stored interval seconds as the largest whole unit/value pair.

    Loops store only ``interval_seconds``; ``_build_loop_fields`` rebuilds them
    from a unit + value, so an edit must round-trip through this to preserve the
    existing cadence.
    """
    secs = interval_seconds or 3600
    if secs % 86400 == 0:
        return "days", secs // 86400
    if secs % 3600 == 0:
        return "hours", secs // 3600
    return "minutes", max(1, secs // 60)


# ---------------------------------------------------------------------------
# chat_loop_create
# ---------------------------------------------------------------------------

class LoopCreateInput(ReasonBaseModel):
    prompt: str = Field(description="The instruction the loop runs on each scheduled fire.")
    cadence_kind: str = Field(
        default="interval",
        description="'interval' (every N minutes/hours/days) or 'clock' (at a wall-clock time of day).",
    )
    interval_value: int = Field(default=1, description="Interval cadence: units between runs (>= 1).")
    interval_unit: str = Field(default="hours", description="Interval cadence: 'minutes', 'hours', or 'days'.")
    clock_time: str = Field(default="", description="Clock cadence: time of day as 'HH:MM' (24-hour).")
    clock_frequency: str = Field(default="daily", description="Clock cadence: 'daily', 'weekdays', or 'weekly'.")
    clock_weekday: int = Field(default=0, description="Weekly clock cadence: weekday, 0=Mon .. 6=Sun.")
    history_mode: str = Field(
        default="fresh",
        description="'fresh' (each run starts clean) or 'conversational' (keeps the loop thread's prior turns).",
    )
    first_run_mode: str = Field(
        default="now",
        description="When the first run fires: 'now', 'scheduled' (next clock occurrence), or 'custom'.",
    )
    first_run_at: str = Field(
        default="", description="For first_run_mode='custom': local datetime 'YYYY-MM-DDTHH:MM'.",
    )
    tz: str = Field(
        default="",
        description="IANA timezone for clock cadences (e.g. 'Europe/Oslo'). Defaults to the server timezone.",
    )
    data_room_ids: list[int] = Field(
        default_factory=list,
        description="Data room IDs to attach to the loop. Omit to run with no data rooms.",
    )
    skill_ids: list[str] = Field(
        default_factory=list,
        description="Skill IDs to attach to the loop (up to 5). Omit for none.",
    )
    model: str = Field(default="", description="Model ID for the loop's turns. Omit for the preferred chat model.")
    max_runs: int | None = Field(
        default=None,
        description="Auto-pause the loop after this many runs. Omit or 0 for unlimited (runs until paused).",
    )


class LoopCreateTool(ContextAwareTool):
    """Create a scheduled, recurring chat loop."""

    name: str = "chat_loop_create"
    audience: str = "main"
    start_label: str = "Creating loop..."
    end_label: str = "Created loop"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return None
        label = result.get("schedule_label")
        return f"Created loop · {label}" if label else "Created loop"

    description: str = (
        "Create a Loop: a recurring, scheduled task that runs the same prompt on a cadence in its own "
        "chat thread. Use when the user wants something done repeatedly on a schedule (e.g. 'every "
        "morning summarize the data room'). The loop runs in a NEW thread, not this conversation. "
        "Choose interval cadence (every N minutes/hours/days) or clock cadence (a time of day, daily/"
        "weekdays/weekly). Data rooms, skill, and model are NOT inherited from this chat — pass them "
        "explicitly by ID/value or the loop runs without them."
    )
    args_schema: type[BaseModel] = LoopCreateInput
    section: str = "skills"

    def _run(
        self, prompt: str, cadence_kind: str = "interval", interval_value: int = 1,
        interval_unit: str = "hours", clock_time: str = "", clock_frequency: str = "daily",
        clock_weekday: int = 0, history_mode: str = "fresh",
        first_run_mode: str = "now", first_run_at: str = "", tz: str = "",
        data_room_ids: list | None = None, skill_ids: list | None = None, model: str = "",
        max_runs: int | None = None, **kwargs,
    ) -> str:
        from chat.loop_service import create_loop

        user, err = _get_user(self.context)
        if err:
            return err

        body = {
            "prompt": prompt,
            "history_mode": history_mode,
            "cadence_kind": cadence_kind,
            "interval_unit": interval_unit,
            "interval_value": interval_value,
            "clock_time": clock_time,
            "clock_frequency": clock_frequency,
            "clock_weekday": clock_weekday,
            "first_run_mode": first_run_mode,
            "first_run_at": first_run_at,
            "data_room_ids": data_room_ids or [],
            "skill_ids": skill_ids or [],
            "model": model,
            "max_runs": max_runs,
        }
        loop, errors = create_loop(
            user=user, body=body, now=_now(), tz_name=(tz or _default_tz()),
        )
        if errors:
            return json.dumps({"status": "error", "message": " ".join(errors)})

        return json.dumps({"status": "ok", **_loop_summary(loop)})


# ---------------------------------------------------------------------------
# chat_loop_list
# ---------------------------------------------------------------------------

class LoopListInput(ReasonBaseModel):
    pass


class LoopListTool(ContextAwareTool):
    """List the current user's loops."""

    name: str = "chat_loop_list"
    audience: str = "main"
    start_label: str = "Listing loops..."
    end_label: str = "Listed loops"

    def end_label_for_result(self, result: dict) -> str | None:
        n = result.get("count")
        if n is None:
            return None
        return f"Found {n} loop{'' if n == 1 else 's'}"

    description: str = (
        "List the current user's loops (active and paused), with each loop's id, prompt, schedule, status, "
        "next run time, run progress, and backing thread id. Use this to find a loop's id before "
        "stopping or editing it, or to report what automations are set up."
    )
    args_schema: type[BaseModel] = LoopListInput
    section: str = "skills"

    def _run(self, **kwargs) -> str:
        from chat.loop_service import list_loops_for_user

        user, err = _get_user(self.context)
        if err:
            return err

        loops = list_loops_for_user(user)
        return json.dumps({
            "status": "ok",
            "count": len(loops),
            "loops": [_loop_summary(loop) for loop in loops],
        })


# ---------------------------------------------------------------------------
# chat_loop_stop
# ---------------------------------------------------------------------------

class LoopStopInput(ReasonBaseModel):
    loop_id: str = Field(description="ID of the loop to pause (from chat_loop_list).")


class LoopStopTool(ContextAwareTool):
    """Pause a loop so it stops firing (reversible)."""

    name: str = "chat_loop_stop"
    audience: str = "main"
    start_label: str = "Stopping loop..."
    end_label: str = "Stopped loop"
    description: str = (
        "Stop a loop by pausing it: it stops firing but is kept and can be resumed later via "
        "chat_loop_edit(resume=true). Identify the loop by its id (use chat_loop_list to find it)."
    )
    args_schema: type[BaseModel] = LoopStopInput
    section: str = "skills"

    def _run(self, loop_id: str, **kwargs) -> str:
        from chat.loop_service import pause_loop
        from chat.models import Loop

        user, err = _get_user(self.context)
        if err:
            return err

        try:
            uuid_mod.UUID(str(loop_id))
        except (ValueError, AttributeError, TypeError):
            return json.dumps({"status": "error", "message": "Invalid loop_id."})

        loop = (
            Loop.objects.filter(id=loop_id, created_by=user)
            .select_related("thread").first()
        )
        if loop is None:
            return json.dumps({"status": "error", "message": "Loop not found."})

        pause_loop(loop)
        return json.dumps({
            "status": "ok",
            "loop_id": str(loop.id),
            "loop_status": loop.status,
            "schedule_label": loop.schedule_label,
        })


# ---------------------------------------------------------------------------
# chat_loop_edit
# ---------------------------------------------------------------------------

class LoopEditInput(ReasonBaseModel):
    loop_id: str = Field(description="ID of the loop to edit (from chat_loop_list).")
    prompt: str | None = Field(default=None, description="New prompt. Omit to keep the current one.")
    cadence_kind: str | None = Field(default=None, description="'interval' or 'clock'. Omit to keep.")
    interval_value: int | None = Field(default=None, description="Interval cadence: units between runs.")
    interval_unit: str | None = Field(default=None, description="Interval cadence: 'minutes', 'hours', 'days'.")
    clock_time: str | None = Field(default=None, description="Clock cadence: 'HH:MM' (24-hour).")
    clock_frequency: str | None = Field(default=None, description="Clock cadence: 'daily', 'weekdays', 'weekly'.")
    clock_weekday: int | None = Field(default=None, description="Weekly clock cadence: 0=Mon .. 6=Sun.")
    history_mode: str | None = Field(default=None, description="'fresh' or 'conversational'.")
    first_run_mode: str | None = Field(
        default=None,
        description="Reschedule the next run: 'keep' (default), 'now', 'scheduled', or 'custom'.",
    )
    first_run_at: str | None = Field(default=None, description="For first_run_mode='custom': 'YYYY-MM-DDTHH:MM'.")
    data_room_ids: list[int] | None = Field(default=None, description="Replace attached data rooms.")
    skill_ids: list[str] | None = Field(
        default=None,
        description="Replace the attached skills (up to 5; omit to keep, [] to clear).",
    )
    model: str | None = Field(default=None, description="Replace the loop's model ('' for preferred).")
    max_runs: int | None = Field(
        default=None,
        description="Auto-pause after this many runs; 0 for unlimited. Omit to keep the current setting.",
    )
    restart: bool = Field(
        default=False,
        description="Re-activate a paused loop and reset its run count to 0 (start over from now).",
    )
    resume: bool = Field(
        default=False,
        description="Re-activate a paused loop where it left off (keeps the completed run count).",
    )


class LoopEditTool(ContextAwareTool):
    """Edit an existing loop; can also resume or restart a paused one."""

    name: str = "chat_loop_edit"
    audience: str = "main"
    start_label: str = "Updating loop..."
    end_label: str = "Updated loop"

    def end_label_for_result(self, result: dict) -> str | None:
        if result.get("status") != "ok":
            return None
        label = result.get("schedule_label")
        return f"Updated loop · {label}" if label else "Updated loop"

    description: str = (
        "Edit an existing loop identified by loop_id. Only the fields you pass change; everything else "
        "is preserved. Use restart=true to re-activate a paused loop and start its run count over, or "
        "resume=true to re-activate it where it left off. By default the next run time is kept; pass "
        "first_run_mode to reschedule it."
    )
    args_schema: type[BaseModel] = LoopEditInput
    section: str = "skills"

    def _run(
        self, loop_id: str, prompt=None, cadence_kind=None, interval_value=None,
        interval_unit=None, clock_time=None, clock_frequency=None, clock_weekday=None,
        history_mode=None, first_run_mode=None, first_run_at=None,
        data_room_ids=None, skill_ids=None, model=None, max_runs=None,
        restart: bool = False, resume: bool = False, **kwargs,
    ) -> str:
        from chat.loop_service import _loop_form_json, resume_loop, update_loop
        from chat.models import Loop
        from core.preferences import get_preferences

        user, err = _get_user(self.context)
        if err:
            return err

        try:
            uuid_mod.UUID(str(loop_id))
        except (ValueError, AttributeError, TypeError):
            return json.dumps({"status": "error", "message": "Invalid loop_id."})

        loop = (
            Loop.objects.filter(id=loop_id, created_by=user)
            .select_related("thread").first()
        )
        if loop is None:
            return json.dumps({"status": "error", "message": "Loop not found."})

        # Seed the payload from the loop's current values (a partial edit must not
        # reset untouched fields), translating into the keys _build_loop_fields reads.
        prefs = get_preferences(user)
        seed = _loop_form_json(loop, prefs)
        unit, value = _interval_unit_value(seed["interval_seconds"])
        body = {
            "prompt": seed["prompt"],
            "history_mode": seed["history_mode"],
            "cadence_kind": seed["cadence_kind"],
            "interval_unit": unit,
            "interval_value": value,
            "clock_time": seed["clock_time"],
            "clock_frequency": seed["clock_frequency"] or "daily",
            "clock_weekday": seed["clock_weekday"] if seed["clock_weekday"] is not None else 0,
            "data_room_ids": seed["data_room_ids"],
            "skill_ids": seed["skill_ids"],
            "model": seed["model"],
            "max_runs": seed["max_runs"],
            "first_run_mode": "keep",
        }

        overrides = {
            "prompt": prompt, "history_mode": history_mode, "cadence_kind": cadence_kind,
            "interval_value": interval_value, "interval_unit": interval_unit,
            "clock_time": clock_time, "clock_frequency": clock_frequency,
            "clock_weekday": clock_weekday,
            "first_run_mode": first_run_mode, "first_run_at": first_run_at,
            "data_room_ids": data_room_ids, "skill_ids": skill_ids, "model": model,
            "max_runs": max_runs,
        }
        for key, val in overrides.items():
            if val is not None:
                body[key] = val
        if restart:
            body["restart"] = True

        loop, errors = update_loop(
            loop=loop, user=user, body=body, now=_now(),
            tz_name=(loop.tz or _default_tz()),
        )
        if errors:
            return json.dumps({"status": "error", "message": " ".join(errors)})

        if resume and loop.status == Loop.Status.PAUSED:
            resume_loop(loop, _now())

        return json.dumps({"status": "ok", **_loop_summary(loop)})


# Register on import
_registry = get_tool_registry()
for _tool in (LoopCreateTool(), LoopListTool(), LoopStopTool(), LoopEditTool()):
    _registry.register_tool(_tool)
