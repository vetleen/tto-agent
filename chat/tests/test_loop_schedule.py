"""Pure scheduling-helper tests (no DB) for chat.loop_schedule."""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from django.test import SimpleTestCase

from chat.loop_schedule import compute_next_run

UTC = ZoneInfo("UTC")


class ComputeNextRunIntervalTests(SimpleTestCase):
    def test_interval_adds_seconds(self):
        ref = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
        nxt = compute_next_run(ref, cadence_kind="interval", interval_seconds=3600)
        self.assertEqual(nxt, ref + timedelta(hours=1))


class ComputeNextRunClockTests(SimpleTestCase):
    def test_daily_same_day_when_before_time(self):
        ref = datetime(2026, 6, 15, 8, 0, tzinfo=UTC)
        nxt = compute_next_run(
            ref, cadence_kind="clock", clock_time=time(9, 0),
            clock_frequency="daily", tz="UTC",
        )
        self.assertEqual(nxt, datetime(2026, 6, 15, 9, 0, tzinfo=UTC))

    def test_daily_next_day_when_past_time(self):
        ref = datetime(2026, 6, 15, 9, 30, tzinfo=UTC)
        nxt = compute_next_run(
            ref, cadence_kind="clock", clock_time=time(9, 0),
            clock_frequency="daily", tz="UTC",
        )
        self.assertEqual(nxt, datetime(2026, 6, 16, 9, 0, tzinfo=UTC))

    def test_weekdays_skips_weekend(self):
        # Walk from a few references; result must always be a weekday at 09:00.
        for day in range(15, 23):  # spans a weekend
            ref = datetime(2026, 6, day, 10, 0, tzinfo=UTC)
            nxt = compute_next_run(
                ref, cadence_kind="clock", clock_time=time(9, 0),
                clock_frequency="weekdays", tz="UTC",
            )
            self.assertLess(nxt.weekday(), 5, f"got weekend for ref {ref}")
            self.assertEqual(nxt.hour, 9)
            self.assertGreater(nxt, ref)

    def test_weekly_lands_on_target_weekday(self):
        ref = datetime(2026, 6, 15, 10, 0, tzinfo=UTC)  # Monday
        nxt = compute_next_run(
            ref, cadence_kind="clock", clock_time=time(9, 0),
            clock_frequency="weekly", clock_weekday=2, tz="UTC",  # Wednesday
        )
        self.assertEqual(nxt.weekday(), 2)
        self.assertEqual(nxt.hour, 9)
        self.assertGreater(nxt, ref)

    def test_timezone_resolves_to_local_clock(self):
        # 09:00 in Europe/Oslo (UTC+2 in June) == 07:00 UTC.
        ref = datetime(2026, 6, 15, 0, 0, tzinfo=UTC)
        nxt = compute_next_run(
            ref, cadence_kind="clock", clock_time=time(9, 0),
            clock_frequency="daily", tz="Europe/Oslo",
        )
        self.assertEqual(nxt, datetime(2026, 6, 15, 7, 0, tzinfo=UTC))
