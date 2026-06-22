"""Pure scheduling helpers for Loops — no DB, fully unit-testable.

A loop stores only ``next_run`` (when it should next fire). After a fire the
service recomputes ``next_run`` from the fire time via :func:`compute_next_run`.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

_UTC = ZoneInfo("UTC")


def _day_allowed(dt: datetime, clock_frequency: str | None, clock_weekday: int | None) -> bool:
    """Whether ``dt``'s weekday is permitted by the clock cadence."""
    wd = dt.weekday()  # Mon=0 … Sun=6
    if clock_frequency == "weekdays":
        return wd < 5
    if clock_frequency == "weekly":
        return wd == clock_weekday
    return True  # daily / unspecified → every day


def _advance_to_allowed_day(
    candidate: datetime, clock_frequency: str | None, clock_weekday: int | None,
    clock_time: time, zone: ZoneInfo,
) -> datetime:
    """Move ``candidate`` forward whole days until it lands on an allowed weekday.

    Recombines the local time each day so the wall-clock time stays exact across
    DST boundaries.
    """
    for _ in range(8):
        if _day_allowed(candidate, clock_frequency, clock_weekday):
            return candidate
        next_date = (candidate + timedelta(days=1)).date()
        candidate = datetime.combine(next_date, clock_time, tzinfo=zone)
    return candidate


def compute_next_run(
    reference_dt: datetime, *,
    cadence_kind: str,
    interval_seconds: int | None = None,
    clock_time: time | None = None,
    clock_frequency: str | None = None,
    clock_weekday: int | None = None,
    tz: str = "UTC",
) -> datetime:
    """Return the next fire time (aware, UTC) strictly after ``reference_dt``.

    - ``interval``: ``reference_dt + interval_seconds``.
    - ``clock``: the next ``clock_time`` (in ``tz``) on a day permitted by
      ``clock_frequency`` (daily / weekdays / weekly+``clock_weekday``).
    """
    if cadence_kind == "interval":
        return reference_dt + timedelta(seconds=int(interval_seconds or 0))

    zone = ZoneInfo(tz or "UTC")
    ref_local = reference_dt.astimezone(zone)
    candidate = datetime.combine(ref_local.date(), clock_time, tzinfo=zone)
    if candidate <= ref_local:
        next_date = (candidate + timedelta(days=1)).date()
        candidate = datetime.combine(next_date, clock_time, tzinfo=zone)
    candidate = _advance_to_allowed_day(
        candidate, clock_frequency, clock_weekday, clock_time, zone,
    )
    return candidate.astimezone(_UTC)


def loop_schedule_kwargs(loop) -> dict:
    """Extract the scheduling kwargs from a ``Loop`` instance."""
    return {
        "cadence_kind": loop.cadence_kind,
        "interval_seconds": loop.interval_seconds,
        "clock_time": loop.clock_time,
        "clock_frequency": loop.clock_frequency or None,
        "clock_weekday": loop.clock_weekday,
        "tz": loop.tz or "UTC",
    }


def next_run_for_loop(loop, reference_dt: datetime) -> datetime:
    """Convenience: next fire time for a ``Loop`` from ``reference_dt``."""
    return compute_next_run(reference_dt, **loop_schedule_kwargs(loop))
