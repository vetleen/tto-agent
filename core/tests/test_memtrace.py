"""Tests for the opt-in memory-attribution sampler (``core.memtrace``).

Covers the process-gating decision (web-only; never Celery/test/management), the
gc type histogram, and the top-movers diff formatting. The sampler *thread* is
never started here — only the pure, side-effect-free helpers and the declined
path of ``maybe_start`` are exercised, so no daemon lingers past the test.
"""

from collections import Counter

from django.test import SimpleTestCase

from core import memtrace


class ShouldStartGatingTests(SimpleTestCase):
    def test_declines_without_flag(self):
        self.assertFalse(memtrace._should_start({}, ["daphne"]))
        self.assertFalse(memtrace._should_start({"MEM_DEBUG": "0"}, ["daphne"]))
        self.assertFalse(memtrace._should_start({"MEM_DEBUG": "false"}, ["daphne"]))

    def test_starts_on_web_process_with_flag(self):
        self.assertTrue(
            memtrace._should_start({"MEM_DEBUG": "1"}, ["/app/.heroku/python/bin/daphne"])
        )
        self.assertTrue(memtrace._should_start({"MEM_DEBUG": "on"}, ["daphne", "-b", "0"]))

    def test_declines_on_celery_worker(self):
        self.assertFalse(
            memtrace._should_start(
                {"MEM_DEBUG": "1"}, ["/app/.heroku/python/bin/celery", "-A", "config"]
            )
        )

    def test_declines_for_management_and_test_commands(self):
        for cmd in ("test", "migrate", "collectstatic", "shell", "makemigrations"):
            with self.subTest(cmd=cmd):
                self.assertFalse(
                    memtrace._should_start({"MEM_DEBUG": "1"}, ["manage.py", cmd])
                )


class MaybeStartDeclinedPathTests(SimpleTestCase):
    def test_maybe_start_noops_without_flag(self):
        # No flag → returns False and never flips the module-level started guard
        # (so no daemon thread is spawned during the suite).
        self.assertFalse(memtrace.maybe_start(env={}, argv=["daphne"]))
        self.assertFalse(memtrace._started)


class TypeHistogramTests(SimpleTestCase):
    def test_counts_live_instances_by_label(self):
        class _Marker:  # local class → distinctive module-qualified label
            pass

        keep = [_Marker() for _ in range(7)]  # noqa: F841 — hold refs so gc sees them
        hist = memtrace._type_histogram()
        label = memtrace._type_label(keep[0])
        self.assertIn("_Marker", label)
        self.assertGreaterEqual(hist.get(label, 0), 7)

    def test_builtin_label_is_bare_name(self):
        self.assertEqual(memtrace._type_label({}), "dict")
        self.assertEqual(memtrace._type_label([]), "list")


class TopMoversFormattingTests(SimpleTestCase):
    def test_no_baseline_shows_largest_populations(self):
        curr = Counter({"dict": 100, "list": 40, "tuple": 5})
        out = memtrace._format_top_movers(None, curr, top=2)
        self.assertIn("dict 100", out)
        self.assertIn("list 40", out)
        self.assertNotIn("tuple", out)  # trimmed to top=2

    def test_diff_shows_growth_first_and_hides_unchanged(self):
        prev = Counter({"dict": 100, "list": 40, "steady": 10})
        curr = Counter({"dict": 130, "list": 25, "steady": 10})
        out = memtrace._format_top_movers(prev, curr, top=10)
        # Growth first, sign-formatted; the unchanged type is omitted.
        self.assertTrue(out.startswith("dict +30"), out)
        self.assertIn("list -15", out)
        self.assertNotIn("steady", out)

    def test_diff_all_unchanged_reports_no_change(self):
        c = Counter({"dict": 5})
        self.assertEqual(memtrace._format_top_movers(c, Counter({"dict": 5}), top=5), "(no change)")
