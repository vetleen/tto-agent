"""Periodic glibc ``malloc_trim`` to return freed heap arenas to the OS.

glibc keeps freed memory in its per-thread arenas rather than handing it back to
the OS, so a process's RSS stays at its high-water mark after a large *transient*
allocation. Concretely (prod incident 2026-08-20): a single long LLM streaming
turn briefly holds thousands of langchain message chunks (~280 MB); the chunks are
freed the moment the turn ends, but RSS stayed pinned at ~577 MB (over the 512 MB
web-dyno cap → sustained R14) because glibc never released the arenas.

A small daemon reads RSS every ``MALLOC_TRIM_INTERVAL`` seconds and, when it's
above ``MALLOC_TRIM_THRESHOLD_MB``, calls ``malloc_trim(0)`` to release free arena
memory back to the OS — dropping RSS back toward baseline after each peak. The
threshold means idle/lean periods pay nothing (no lock contention when there's
nothing to release).

Linux/glibc only (no-op elsewhere). Runs on the long-lived web (daphne) process;
skips Celery, the test runner, and management/release commands. Disable by setting
``MALLOC_TRIM_INTERVAL=0``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time

logger = logging.getLogger(__name__)

# Management/release commands that run AppConfig.ready but must not start the
# daemon (short-lived one-off processes). Mirrors core.memtrace._SKIP_ARGV1.
_SKIP_ARGV1 = {
    "test", "shell", "shell_plus", "migrate", "makemigrations", "collectstatic",
    "check", "createsuperuser", "dbshell", "showmigrations", "loaddata",
    "dumpdata", "sqlmigrate", "runscript",
}

_started = False
_start_lock = threading.Lock()
_libc = None
_libc_loaded = False


def _get_libc():
    """Return the glibc handle on Linux (cached), else ``None``."""
    global _libc, _libc_loaded
    if not _libc_loaded:
        _libc_loaded = True
        if sys.platform.startswith("linux"):
            try:
                _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            except OSError:
                _libc = None
    return _libc


def trim_malloc() -> bool:
    """Call ``malloc_trim(0)`` to return free arenas to the OS. Best-effort.

    Returns ``True`` if the call was made (Linux/glibc), ``False`` otherwise.
    Never raises — a diagnostics/ops helper must not break the caller.
    """
    libc = _get_libc()
    if libc is None or not hasattr(libc, "malloc_trim"):
        return False
    try:
        libc.malloc_trim(0)
        return True
    except Exception:  # noqa: BLE001 — best-effort, never propagate
        logger.debug("malloc_trim failed", exc_info=True)
        return False


def _read_rss_kb() -> int | None:
    """Return VmRSS in kB from ``/proc/self/status`` (Linux), else ``None``."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split(":", 1)[1].strip().split()[0])
    except (OSError, ValueError):
        pass
    return None


def _should_start(env: dict[str, str], argv: list[str]) -> bool:
    """Decide whether the trimmer should run in *this* process (pure/testable).

    Requires Linux, a positive interval, and that this is not a Celery worker,
    test run, or management command — so it only activates on the daphne web
    process (the memory-constrained one).
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        if int(env.get("MALLOC_TRIM_INTERVAL", "60")) <= 0:
            return False
    except (TypeError, ValueError):
        # Typo/garbage (e.g. "60s") is not a disable request — the documented
        # off switch is a valid "0". Fall through and keep the mitigation on;
        # maybe_start's _pos_int clamps the bad value back to the default.
        pass
    argv0 = os.path.basename(argv[0]) if argv else ""
    if "celery" in argv0 or "celery" in " ".join(argv[1:2]):
        return False
    if len(argv) > 1 and argv[1] in _SKIP_ARGV1:
        return False
    return True


class _Trimmer:
    def __init__(self, interval: float, threshold_kb: int):
        self.interval = interval
        self.threshold_kb = threshold_kb

    def run(self) -> None:
        logger.info(
            "malloc_trim daemon started (interval=%ss threshold=%.0fMB)",
            self.interval, self.threshold_kb / 1024,
        )
        while True:
            time.sleep(self.interval)
            try:
                rss = _read_rss_kb()
                # If RSS is unreadable, trim unconditionally (safe); otherwise only
                # when elevated, so lean/idle periods incur no trim.
                if rss is not None and rss < self.threshold_kb:
                    continue
                if trim_malloc() and rss is not None:
                    after = _read_rss_kb()
                    if after is not None and after < rss:
                        logger.info(
                            "malloc_trim released RSS %.0f->%.0fMB",
                            rss / 1024, after / 1024,
                        )
            except Exception:  # noqa: BLE001 — never let the daemon die
                logger.exception("malloc_trim tick failed")


def maybe_start(env: dict[str, str] | None = None, argv: list[str] | None = None) -> bool:
    """Start the periodic trimmer iff this is the Linux web process. Idempotent."""
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

            interval = _pos_int("MALLOC_TRIM_INTERVAL", 60, 10, 3600)
            threshold_mb = _pos_int("MALLOC_TRIM_THRESHOLD_MB", 450, 64, 4096)
            thread = threading.Thread(
                target=_Trimmer(interval, threshold_mb * 1024).run,
                name="malloc-trim", daemon=True,
            )
            thread.start()
            _started = True
            return True
    except Exception:  # noqa: BLE001 — never break startup
        logger.exception("malloc_trim failed to start")
        return False
