"""Opt-in, web-process memory-attribution logging for leak hunting.

Gated entirely behind the ``MEM_DEBUG`` env var. When enabled, a single daemon
thread logs — every ``MEM_DEBUG_INTERVAL`` seconds — the process RSS/swap, a few
``gc`` vitals, and the **top movers** in a ``gc`` object-type histogram (the diff
versus the previous sample). A growing object type (``dict``, ``chat.consumers.
Turn``, an SDK client, …) therefore shows up directly in the logs, which is often
enough to name a leak by *what* is accumulating.

``gc.get_objects()`` only sees container/instance objects (not atomic ``bytes`` /
``str`` buffers), so for a size-based / native-buffer leak set
``MEM_DEBUG_TRACEMALLOC=1`` (intended for **staging**, where the tracker's own
overhead is acceptable): it additionally logs ``tracemalloc``'s top allocation
sites by ``file:line`` and their growth between samples — the precise culprit.

Design constraints:
- **Web only.** Started from :meth:`core.apps.CoreConfig.ready`, but
  :func:`_should_start` refuses to run under Celery, the test runner, or a
  management/release command — so an app-wide ``MEM_DEBUG`` config var only ever
  activates on the ``daphne`` web process. The worker (1 GB, healthy) is left be.
- **Never crashes the app.** Every sample iteration is wrapped; a failure logs and
  the loop continues. Startup failures are swallowed too.
- **Cheap when off.** ``maybe_start()`` returns immediately unless the flag is set.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from collections import Counter

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Management/release commands that run ``AppConfig.ready`` but must never start the
# sampler (they are short-lived one-off processes, not the long-lived web server).
_SKIP_ARGV1 = {
    "test", "shell", "shell_plus", "migrate", "makemigrations", "collectstatic",
    "check", "createsuperuser", "dbshell", "showmigrations", "loaddata",
    "dumpdata", "sqlmigrate", "runscript",
}

# Module-level guard so a second ready() (autoreload, repeated setup) can't spawn
# a second sampler thread.
_started = False
_start_lock = threading.Lock()


def _is_truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


def _should_start(env: dict[str, str], argv: list[str]) -> bool:
    """Decide whether the sampler should run in *this* process.

    Pure (no side effects) so the gating is unit-testable. Requires ``MEM_DEBUG``
    truthy AND that this is not a Celery worker, test run, or management command.
    A ``daphne`` web process passes; everything else is refused.
    """
    if not _is_truthy(env.get("MEM_DEBUG")):
        return False
    argv0 = os.path.basename(argv[0]) if argv else ""
    # Celery worker/beat — the worker has its own 1 GB budget and is not the leak.
    if "celery" in argv0 or "celery" in " ".join(argv[1:2]):
        return False
    # Management / release-phase commands (manage.py <cmd> ...).
    if len(argv) > 1 and argv[1] in _SKIP_ARGV1:
        return False
    return True


def _read_proc_status_kb() -> dict[str, int]:
    """Return selected ``/proc/self/status`` sizes in kB (Linux). Empty off-Linux."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmSwap:", "VmHWM:", "RssAnon:")):
                    key, val = line.split(":", 1)
                    out[key.strip()] = int(val.strip().split()[0])
    except (OSError, ValueError):
        pass
    return out


def _fmt_mb(kb: int | None) -> str:
    return f"{kb / 1024:.1f}MB" if kb is not None else "n/a"


def _type_label(obj: object) -> str:
    """A stable, readable name for an object's type — ``module.Qualname`` except
    for builtins, which use the bare name (``dict``, ``list``, ``function``)."""
    cls = type(obj)
    mod = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", "?")
    return name if mod in ("builtins", "") else f"{mod}.{name}"


def _type_histogram() -> Counter:
    """Count gc-tracked objects by type label. ``bytes``/``str`` are not tracked
    by the collector, so this catches object-*count* growth, not raw buffers."""
    counts: Counter = Counter()
    for obj in gc.get_objects():
        try:
            counts[_type_label(obj)] += 1
        except Exception:  # noqa: BLE001 — a broken __class__ must not abort the walk
            counts["<unknown>"] += 1
    return counts


def _format_top_movers(prev: Counter | None, curr: Counter, top: int) -> str:
    """Render the ``top`` largest count deltas (growth first) as a compact string."""
    if prev is None:
        # No baseline yet — show the largest absolute populations instead.
        items = curr.most_common(top)
        return " | ".join(f"{name} {count:,}" for name, count in items)
    deltas = {name: curr.get(name, 0) - prev.get(name, 0) for name in set(curr) | set(prev)}
    movers = sorted(deltas.items(), key=lambda kv: kv[1], reverse=True)
    movers = [m for m in movers if m[1] != 0][:top]
    if not movers:
        return "(no change)"
    return " | ".join(f"{name} {delta:+,}" for name, delta in movers)


class _Sampler:
    def __init__(self, interval: float, top: int, use_tracemalloc: bool, frames: int):
        self.interval = interval
        self.top = top
        self.use_tracemalloc = use_tracemalloc
        self.frames = frames
        self._prev_hist: Counter | None = None
        self._prev_snapshot = None  # tracemalloc.Snapshot
        self._started_at = time.monotonic()

    def _log_tracemalloc(self) -> None:
        import tracemalloc

        snapshot = tracemalloc.take_snapshot()
        if self._prev_snapshot is not None:
            stats = snapshot.compare_to(self._prev_snapshot, "lineno")
            top = stats[: self.top]
            parts = [
                f"{s.traceback[0]} {s.size_diff / 1048576:+.1f}MB"
                f" (now {s.size / 1048576:.1f}MB)"
                for s in top
            ]
            logger.info("MEMTRACE tracemalloc growth: %s", " | ".join(parts) or "(none)")
        else:
            stats = snapshot.statistics("lineno")[: self.top]
            parts = [f"{s.traceback[0]} {s.size / 1048576:.1f}MB" for s in stats]
            logger.info("MEMTRACE tracemalloc top (baseline): %s", " | ".join(parts))
        self._prev_snapshot = snapshot

    def _sample_once(self) -> None:
        status = _read_proc_status_kb()
        gc_counts = gc.get_count()
        n_objects = len(gc.get_objects())
        uptime = int(time.monotonic() - self._started_at)
        logger.info(
            "MEMTRACE rss=%s swap=%s hwm=%s gc_objects=%s gc_garbage=%s "
            "gc_counts=%s uptime=%ss",
            _fmt_mb(status.get("VmRSS")),
            _fmt_mb(status.get("VmSwap")),
            _fmt_mb(status.get("VmHWM")),
            f"{n_objects:,}",
            len(gc.garbage),
            gc_counts,
            uptime,
        )

        hist = _type_histogram()
        logger.info(
            "MEMTRACE top type growth since last sample: %s",
            _format_top_movers(self._prev_hist, hist, self.top),
        )
        self._prev_hist = hist

        if self.use_tracemalloc:
            self._log_tracemalloc()

    def run(self) -> None:
        logger.info(
            "MEMTRACE sampler started (interval=%ss top=%s tracemalloc=%s)",
            self.interval, self.top, self.use_tracemalloc,
        )
        while True:
            try:
                self._sample_once()
            except Exception:  # noqa: BLE001 — diagnostics must never kill the web process
                logger.exception("MEMTRACE sample failed")
            time.sleep(self.interval)


def maybe_start(env: dict[str, str] | None = None, argv: list[str] | None = None) -> bool:
    """Start the memory sampler iff ``MEM_DEBUG`` is set and this is the web process.

    Idempotent and exception-safe: returns ``True`` if it started the thread (or one
    was already running), ``False`` if gating declined. Safe to call from
    ``AppConfig.ready`` in every process — it no-ops everywhere but ``daphne``.
    """
    global _started
    env = env if env is not None else dict(os.environ)
    argv = argv if argv is not None else sys.argv

    try:
        if not _should_start(env, argv):
            return False
        with _start_lock:
            if _started:
                return True

            def _pos_int(name: str, default: int, lo: int, hi: int) -> int:
                try:
                    return max(lo, min(hi, int(env.get(name, default))))
                except (TypeError, ValueError):
                    return default

            interval = _pos_int("MEM_DEBUG_INTERVAL", 120, 15, 3600)
            top = _pos_int("MEM_DEBUG_TOP", 12, 1, 50)
            use_tm = _is_truthy(env.get("MEM_DEBUG_TRACEMALLOC"))
            frames = _pos_int("MEM_DEBUG_TRACEMALLOC_FRAMES", 8, 1, 30)

            if use_tm:
                import tracemalloc

                if not tracemalloc.is_tracing():
                    tracemalloc.start(frames)

            sampler = _Sampler(interval, top, use_tm, frames)
            thread = threading.Thread(
                target=sampler.run, name="memtrace", daemon=True,
            )
            thread.start()
            _started = True
            return True
    except Exception:  # noqa: BLE001 — never let diagnostics break startup
        logger.exception("MEMTRACE failed to start")
        return False
