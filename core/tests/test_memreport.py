"""Tests for the worker-capable memory report (``core.memreport``).

Exercises the gate, the ``sys._debugmallocstats`` parser, the readers'
never-raise contract, the task-boundary hooks (no-op when disabled, skip the
sweepers when enabled) and the on-demand task/command. No sampler thread is
started and no Linux-only reading is assumed — everything must pass on Windows.
"""

from __future__ import annotations

from unittest import mock

from django.core.management import call_command
from django.test import SimpleTestCase

from core import memreport


_DMS_SAMPLE = """Small block threshold = 512, in 32 size classes.

class   size   num pools   blocks in use  avail blocks
-----   ----   ---------   -------------  ------------
    0     16           3             300           300

# arenas allocated total           =                  1,234
# arenas reclaimed                 =                    567
# arenas highwater mark            =                    700
# arenas allocated current         =                    667
667 arenas * 1048576 bytes/arena   =            699,400,192

# bytes in allocated blocks        =            600,000,000
# bytes in available blocks        =             50,000,000
123 unused pools * 16384 bytes     =              2,015,232
# bytes lost to pool headers       =              1,000,000
# bytes lost to quantization       =              2,000,000
# bytes lost to arena alignment    =                      0
Total                              =            699,400,192
"""


class GateTests(SimpleTestCase):
    def test_disabled_without_flag(self):
        self.assertFalse(memreport.worker_debug_enabled({}))
        self.assertFalse(memreport.worker_debug_enabled({"MEM_DEBUG_WORKER": "0"}))
        self.assertFalse(memreport.worker_debug_enabled({"MEM_DEBUG": "1"}))

    def test_enabled_with_flag(self):
        for v in ("1", "true", "on", "YES"):
            with self.subTest(v=v):
                self.assertTrue(memreport.worker_debug_enabled({"MEM_DEBUG_WORKER": v}))

    def test_sweepers_are_skipped(self):
        self.assertTrue(memreport._skip_task("chat.tasks.expire_stale_subagent_runs"))
        self.assertTrue(memreport._skip_task("chat.tasks.tick_and_scan_loops"))
        self.assertTrue(memreport._skip_task("core.tasks.memory_report_task"))
        self.assertFalse(memreport._skip_task("chat.tasks.run_subagent_task"))
        self.assertFalse(memreport._skip_task("documents.tasks.process_document"))

    def test_report_is_noop_when_disabled(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MEM_DEBUG_WORKER", None)
            with self.assertNoLogs("core.memreport", level="INFO"):
                self.assertEqual(memreport.memory_report("x"), {})


class ParserTests(SimpleTestCase):
    def test_parse_debugmallocstats(self):
        stats = memreport.parse_debugmallocstats(_DMS_SAMPLE)
        self.assertEqual(stats["arenas"], 667)
        self.assertEqual(stats["arena_bytes"], 699_400_192)
        self.assertEqual(stats["arenas_hwm"], 700)
        self.assertEqual(stats["allocated_bytes"], 600_000_000)
        self.assertEqual(stats["available_bytes"], 50_000_000)

    def test_parse_empty(self):
        self.assertEqual(memreport.parse_debugmallocstats(""), {})


class ReaderTests(SimpleTestCase):
    def test_readers_never_raise(self):
        # Linux-only files may be absent; the readers must degrade, not raise.
        self.assertIsInstance(memreport.read_proc_status(), dict)
        self.assertIsInstance(memreport.read_smaps_rollup(), dict)
        self.assertIsInstance(memreport.largest_mappings(), list)
        mi = memreport.mallinfo2()
        self.assertTrue(mi is None or isinstance(mi, dict))

    def test_pymalloc_stats_reads_real_numbers(self):
        stats = memreport.pymalloc_stats()
        # sys._debugmallocstats exists on every CPython build we run.
        self.assertIsNotNone(stats)
        self.assertGreater(stats.get("arenas", 0), 0)
        self.assertGreater(stats.get("allocated_bytes", 0), 0)

    def test_thread_dump_shape(self):
        n, lines = memreport.thread_dump()
        self.assertGreaterEqual(n, 1)
        self.assertIsInstance(lines, list)
        for line in lines:
            self.assertIn(":", line)


class ReportTests(SimpleTestCase):
    def test_forced_report_logs_headline_and_returns_numbers(self):
        with self.assertLogs("core.memreport", level="INFO") as logs:
            result = memreport.memory_report("unit", force=True, full=True, collect=True)
        joined = "\n".join(logs.output)
        self.assertIn("MEMREPORT unit rss=", joined)
        self.assertIn("gc_objects=", joined)
        self.assertIn("threads=", joined)
        self.assertEqual(result["tag"], "unit")
        self.assertIn("pymalloc", result)
        self.assertIsNotNone(result.get("collected"))

    def test_task_hooks_noop_when_disabled(self):
        with mock.patch.object(memreport, "worker_debug_enabled", return_value=False):
            with self.assertNoLogs("core.memreport", level="INFO"):
                memreport.task_prerun_report("t1", "chat.tasks.run_subagent_task")
                memreport.task_postrun_report("t1", "chat.tasks.run_subagent_task")
        self.assertNotIn("t1", memreport._task_state)

    def test_task_hooks_report_when_enabled_and_skip_sweepers(self):
        with mock.patch.object(memreport, "worker_debug_enabled", return_value=True):
            with self.assertNoLogs("core.memreport", level="INFO"):
                memreport.task_prerun_report("s1", "chat.tasks.expire_stale_subagent_runs")
            with self.assertLogs("core.memreport", level="INFO") as logs:
                memreport.task_prerun_report("t2", "chat.tasks.run_subagent_task")
                self.assertIn("t2", memreport._task_state)
                memreport.task_postrun_report("t2", "chat.tasks.run_subagent_task")
        joined = "\n".join(logs.output)
        self.assertIn("MEMREPORT task_start name=chat.tasks.run_subagent_task id=t2", joined)
        self.assertIn("MEMREPORT task_end name=chat.tasks.run_subagent_task id=t2", joined)
        self.assertNotIn("t2", memreport._task_state)

    def test_sampler_declines_without_flag(self):
        self.assertFalse(memreport.maybe_start_worker_sampler({}))
        self.assertFalse(memreport._sampler_started)


class TracemallocTests(SimpleTestCase):
    def test_line_sizes_none_when_not_tracing(self):
        import tracemalloc

        if tracemalloc.is_tracing():
            self.skipTest("tracemalloc already tracing in this process")
        self.assertIsNone(memreport.tracemalloc_line_sizes())

    def test_line_sizes_and_growth_when_tracing(self):
        import tracemalloc

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start(3)
        try:
            before = memreport.tracemalloc_line_sizes()
            keep = [bytearray(256 * 1024) for _ in range(8)]  # ~2 MB attributed to this line
            after = memreport.tracemalloc_line_sizes()
            self.assertIsInstance(before, dict)
            self.assertIsInstance(after, dict)
            growth = memreport._fmt_growth(after, before, 5, memreport._app_root())
            self.assertIn("test_memreport.py", growth)
            self.assertIn("+", growth)
            del keep
        finally:
            if not was_tracing:
                tracemalloc.stop()

    def test_growth_formatting_empty(self):
        self.assertEqual(memreport._fmt_growth({}, {}, 5, "/x"), "(none)")
        self.assertEqual(memreport._fmt_sizes({}, 5, "/x"), "(none)")


class OnDemandTests(SimpleTestCase):
    def test_task_forces_a_report(self):
        from core.tasks import memory_report_task

        with mock.patch.object(memreport, "memory_report", return_value={"rss_kb": 1024, "threads": 3, "busy_threads": []}) as rep:
            out = memory_report_task.run("t")
        rep.assert_called_once()
        self.assertTrue(rep.call_args.kwargs.get("force"))
        self.assertEqual(out["rss_kb"], 1024)

    def test_command_enqueues_or_runs_locally(self):
        with mock.patch("core.tasks.memory_report_task.delay") as delay:
            delay.return_value = mock.Mock(id="abc")
            call_command("memreport", "--tag", "x")
        delay.assert_called_once_with("x")
        with mock.patch.object(memreport, "memory_report", return_value={"rss_kb": 2048}) as rep:
            call_command("memreport", "--local")
        self.assertTrue(rep.call_args.kwargs.get("force"))
