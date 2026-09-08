"""In-process memory attribution report — usable on the Celery worker.

Why this exists: the worker's RSS ratchets upward after memory-heavy tasks and
never comes back down, and neither Heroku's ``sample#memory`` lines nor
:mod:`core.memtrace` (web-only by design) can say *where* that memory lives. One
sample taken from inside the process splits RSS into buckets that each point at
a different fix:

- live Python objects still referenced   -> tracemalloc live total + gc type deltas
- glibc heap freed but not returned      -> ``mallinfo2``: ``free`` (fordblks) vs
                                            ``inuse`` (uordblks)
- pymalloc arenas pinned by fragmentation -> ``sys._debugmallocstats``: arenas held
                                            vs bytes actually in allocated blocks
- native allocator state (onnxruntime,    -> RSS far above glibc + pymalloc, large
  lxml, numpy, SSL buffers)                  anonymous mappings in ``smaps``
- threads still holding a finished task's -> live-thread stack dump (app frames)
  working set

Gated by ``MEM_DEBUG_WORKER`` (truthy), so production pays nothing unless asked.
When enabled, :mod:`config.celery` logs a report before and after every
non-trivial task (with the RSS delta and, when tracemalloc is tracing, the top
allocation *growth* by ``file:line`` across that task), and a sampler thread logs
a compact report every ``MEM_DEBUG_INTERVAL`` seconds. Independently of the flag,
``core.tasks.memory_report_task`` logs a full report on demand (enqueue it with
``manage.py memreport``) so a live worker can be inspected without a restart.

Linux-specific readings degrade to ``None`` elsewhere. Nothing here may raise
into a task: every reader is wrapped, and a failed reader just leaves its field
blank.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import logging
import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Periodic sweepers fire every 60-300 s; reporting around them would drown the
# signal. Matched as substrings of the Celery task name.
_SKIP_TASK_SUBSTRINGS = (
    "expire_stale",
    "tick_and_scan",
    "requeue_stale",
    "prune_",
    "memory_report",
    "debug_task",
)

# Frames kept per tracemalloc allocation on the worker. Each traced block costs
# roughly (frames x 16 B + ~60 B); 5 keeps the tracker's own footprint at a
# fraction of the process while still naming the caller two levels up.
_DEFAULT_TRACEMALLOC_FRAMES = 5

_MB = 1024 * 1024


def _is_truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


def worker_debug_enabled(env: dict[str, str] | None = None) -> bool:
    """True when the worker-side memory reporting is switched on."""
    env = env if env is not None else os.environ
    return _is_truthy(env.get("MEM_DEBUG_WORKER"))


def _skip_task(task_name: str | None) -> bool:
    name = task_name or ""
    return any(s in name for s in _SKIP_TASK_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Readers (each returns None / empty on failure — never raises)
# ---------------------------------------------------------------------------


def read_proc_status() -> dict[str, int]:
    """Selected ``/proc/self/status`` fields in kB (``Threads`` is a count)."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith(("VmRSS:", "VmSwap:", "VmHWM:", "RssAnon:", "RssFile:", "Threads:")):
                    key, val = line.split(":", 1)
                    out[key.strip()] = int(val.strip().split()[0])
    except (OSError, ValueError):
        pass
    return out


def read_smaps_rollup() -> dict[str, int]:
    """Selected ``/proc/self/smaps_rollup`` fields in kB."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/smaps_rollup", "r") as fh:
            for line in fh:
                if line.startswith(("Rss:", "Pss:", "Anonymous:", "Swap:", "Private_Dirty:")):
                    key, val = line.split(":", 1)
                    out[key.strip()] = int(val.strip().split()[0])
    except (OSError, ValueError):
        pass
    return out


_MAP_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+\s+\S+\s+\S+\s+\S+\s+\S+\s*(.*)$")


def largest_mappings(limit: int = 8) -> list[str]:
    """The ``limit`` mappings with the most resident memory, as ``name rssMB/sizeMB``.

    Distinguishes the brk heap (``[heap]``), anonymous mmap regions (glibc
    non-main arenas, pymalloc arenas, onnxruntime/numpy buffers — the kernel
    merges adjacent anonymous maps, so one big ``anon`` entry is normal) and
    file-backed maps (shared libraries, the tiktoken/onnx model files).
    """
    entries: list[tuple[int, int, str]] = []
    try:
        with open("/proc/self/smaps", "r") as fh:
            name = None
            size_kb = 0
            for line in fh:
                m = _MAP_HEADER.match(line)
                if m:
                    name = m.group(1).strip() or "anon"
                    size_kb = 0
                    continue
                if line.startswith("Size:"):
                    size_kb = int(line.split()[1])
                elif line.startswith("Rss:") and name is not None:
                    rss_kb = int(line.split()[1])
                    if rss_kb:
                        entries.append((rss_kb, size_kb, name))
                    name = None
    except (OSError, ValueError):
        return []
    entries.sort(reverse=True)
    return [
        f"{os.path.basename(n) if n.startswith('/') else n} {r / 1024:.0f}/{s / 1024:.0f}MB"
        for r, s, n in entries[:limit]
    ]


class _MallInfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost",
    )]


_libc = None
_libc_loaded = False


def _get_libc():
    global _libc, _libc_loaded
    if not _libc_loaded:
        _libc_loaded = True
        if sys.platform.startswith("linux"):
            try:
                _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            except OSError:
                _libc = None
    return _libc


def mallinfo2() -> dict[str, int] | None:
    """glibc ``mallinfo2`` totals across all arenas, in bytes (glibc >= 2.33)."""
    libc = _get_libc()
    if libc is None or not hasattr(libc, "mallinfo2"):
        return None
    try:
        fn = libc.mallinfo2
        fn.restype = _MallInfo2
        fn.argtypes = []
        mi = fn()
        return {
            "arena": mi.arena,        # bytes obtained via brk (main heap)
            "hblks": mi.hblks,        # number of mmapped chunks
            "hblkhd": mi.hblkhd,      # bytes in mmapped chunks
            "uordblks": mi.uordblks,  # bytes in use
            "fordblks": mi.fordblks,  # bytes free inside the arenas
            "keepcost": mi.keepcost,  # releasable top-of-heap bytes
        }
    except Exception:  # noqa: BLE001 — diagnostics must not raise
        return None


_DMS_LINE = re.compile(r"^#?\s*(?P<key>[^=]+?)\s*=\s*(?P<val>[\d,]+)")
_DMS_ARENA_LINE = re.compile(r"^(?P<n>[\d,]+) arenas \* (?P<size>[\d,]+) bytes/arena\s*=\s*(?P<total>[\d,]+)")


def parse_debugmallocstats(text: str) -> dict[str, int]:
    """Pull the arena-level totals out of ``sys._debugmallocstats`` output."""
    out: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        m = _DMS_ARENA_LINE.match(line)
        if m:
            out["arenas"] = int(m.group("n").replace(",", ""))
            out["arena_bytes"] = int(m.group("total").replace(",", ""))
            continue
        m = _DMS_LINE.match(line)
        if not m:
            continue
        key = m.group("key").strip().lower()
        val = int(m.group("val").replace(",", ""))
        if key == "arenas allocated current":
            out.setdefault("arenas", val)
        elif key == "arenas highwater mark":
            out["arenas_hwm"] = val
        elif key == "bytes in allocated blocks":
            out["allocated_bytes"] = val
        elif key == "bytes in available blocks":
            out["available_bytes"] = val
    return out


_dms_lock = threading.Lock()


def pymalloc_stats() -> dict[str, int] | None:
    """Arena-level pymalloc numbers via ``sys._debugmallocstats``.

    The function prints to the C-level stderr, so fd 2 is pointed at a temp
    file for the few milliseconds of the call (another thread's stderr write in
    that window lands in the file instead of the log — acceptable for an
    opt-in diagnostic).
    """
    dms = getattr(sys, "_debugmallocstats", None)
    if dms is None:
        return None
    with _dms_lock:
        try:
            with tempfile.TemporaryFile(mode="w+b") as tmp:
                try:
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001
                    pass
                saved = os.dup(2)
                try:
                    os.dup2(tmp.fileno(), 2)
                    dms()
                finally:
                    os.dup2(saved, 2)
                    os.close(saved)
                tmp.seek(0)
                text = tmp.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
    stats = parse_debugmallocstats(text)
    return stats or None


_gc_baseline: Counter | None = None
_gc_lock = threading.Lock()


def gc_summary(top: int = 10) -> tuple[int, str]:
    """(gc object count, top type-count movers since the first report)."""
    from core.memtrace import _format_top_movers, _type_histogram

    global _gc_baseline
    hist = _type_histogram()
    n_objects = sum(hist.values())
    with _gc_lock:
        if _gc_baseline is None:
            _gc_baseline = hist
            return n_objects, "(baseline set) " + _format_top_movers(None, hist, top)
        return n_objects, _format_top_movers(_gc_baseline, hist, top)


def _app_root() -> str:
    # core/memreport.py -> project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_app_frame(filename: str, root: str) -> bool:
    if "site-packages" in filename or "/lib/python" in filename or "\\lib\\python" in filename.lower():
        return False
    return filename.startswith(root)


def thread_dump(limit: int = 12) -> tuple[int, list[str]]:
    """(thread count, per-thread ``name: innermost <- outermost`` app frames).

    Threads with no application frame on their stack (idle pool threads, the
    Celery consumer, samplers) are omitted — what matters is which tasks are
    still executing and where, especially after their DB row says otherwise.
    """
    root = _app_root()
    names = {t.ident: t.name for t in threading.enumerate()}
    lines: list[str] = []
    try:
        frames = sys._current_frames()
    except Exception:  # noqa: BLE001
        return len(names), lines
    me = threading.get_ident()
    for ident, frame in frames.items():
        if ident == me:
            continue
        app: list[str] = []
        f = frame
        depth = 0
        while f is not None and depth < 200:
            fn = f.f_code.co_filename
            if _is_app_frame(fn, root):
                rel = os.path.relpath(fn, root).replace("\\", "/")
                app.append(f"{rel}:{f.f_lineno} {f.f_code.co_name}")
            f = f.f_back
            depth += 1
        if not app:
            continue
        desc = app[0] if len(app) == 1 else f"{app[0]} <- {app[-1]}"
        lines.append(f"{names.get(ident, ident)}: {desc}")
        if len(lines) >= limit:
            break
    return len(names), lines


def ensure_tracemalloc(env: dict[str, str] | None = None) -> bool:
    """Start tracemalloc if ``MEM_DEBUG_TRACEMALLOC`` asks for it. Idempotent."""
    env = env if env is not None else os.environ
    if not _is_truthy(env.get("MEM_DEBUG_TRACEMALLOC")):
        return False
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            try:
                frames = int(env.get("MEM_DEBUG_TRACEMALLOC_FRAMES", _DEFAULT_TRACEMALLOC_FRAMES))
            except (TypeError, ValueError):
                frames = _DEFAULT_TRACEMALLOC_FRAMES
            tracemalloc.start(max(1, min(frames, 30)))
        return True
    except Exception:  # noqa: BLE001
        return False


def tracemalloc_snapshot():
    """Current tracemalloc snapshot, or None when not tracing."""
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            return None
        return tracemalloc.take_snapshot()
    except Exception:  # noqa: BLE001
        return None


def _fmt_stats(stats, top: int, root: str, growth: bool) -> str:
    parts = []
    for s in stats[:top]:
        fr = s.traceback[0]
        fn = fr.filename
        if fn.startswith(root):
            fn = os.path.relpath(fn, root).replace("\\", "/")
        else:
            fn = fn.split("site-packages/")[-1].split("site-packages\\")[-1]
        if growth:
            parts.append(f"{fn}:{fr.lineno} {s.size_diff / _MB:+.1f}MB (now {s.size / _MB:.1f})")
        else:
            parts.append(f"{fn}:{fr.lineno} {s.size / _MB:.1f}MB")
    return " | ".join(parts) or "(none)"


def _caches_summary() -> str:
    bits = []
    try:
        tk = sys.modules.get("tiktoken")
        if tk is not None:
            from tiktoken import registry as _reg  # type: ignore

            bits.append("tiktoken=" + ",".join(sorted(getattr(_reg, "ENCODINGS", {}).keys())) or "tiktoken=-")
    except Exception:  # noqa: BLE001
        pass
    try:
        ret = sys.modules.get("documents.services.retrieval")
        if ret is not None:
            bits.append(f"ranker={'yes' if getattr(ret, '_ranker_cache', None) is not None else 'no'}")
    except Exception:  # noqa: BLE001
        pass
    try:
        mf = sys.modules.get("llm.core.model_factory")
        if mf is not None:
            bits.append(f"llm_clients={len(getattr(mf, '_client_cache', {}))}")
    except Exception:  # noqa: BLE001
        pass
    loaded = [m for m in ("onnxruntime", "lxml.etree", "numpy", "weasyprint", "playwright", "pypdfium2") if m in sys.modules]
    bits.append("native=" + (",".join(loaded) or "-"))
    bits.append(f"modules={len(sys.modules)}")
    return " ".join(bits)


def _mb(kb: int | None) -> str:
    return f"{kb / 1024:.0f}MB" if kb is not None else "n/a"


def _mbb(b: int | None) -> str:
    return f"{b / _MB:.0f}MB" if b is not None else "n/a"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def memory_report(
    tag: str,
    *,
    full: bool = False,
    prev_snapshot=None,
    prev_rss_kb: int | None = None,
    top: int = 10,
    force: bool = False,
    collect: bool = False,
) -> dict:
    """Log a memory report and return the numbers (for tests / callers).

    ``full`` adds the largest mappings and a bigger tracemalloc top list;
    ``prev_snapshot`` (a tracemalloc snapshot) adds allocation growth since it;
    ``collect`` runs ``gc.collect()`` first and reports what it freed (so a
    post-task report shows whether reference cycles were holding the memory).
    ``force`` bypasses the ``MEM_DEBUG_WORKER`` gate (on-demand task).
    """
    if not force and not worker_debug_enabled():
        return {}
    result: dict = {"tag": tag}
    try:
        collected = None
        if collect:
            try:
                collected = gc.collect()
            except Exception:  # noqa: BLE001
                collected = None

        status = read_proc_status()
        rollup = read_smaps_rollup()
        mi = mallinfo2()
        pm = pymalloc_stats()
        try:
            py_blocks = sys.getallocatedblocks()
        except Exception:  # noqa: BLE001
            py_blocks = None

        traced = None
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                traced = tracemalloc.get_traced_memory()
        except Exception:  # noqa: BLE001
            traced = None

        rss_kb = status.get("VmRSS")
        delta = ""
        if rss_kb is not None and prev_rss_kb is not None:
            delta = f" delta_rss={(rss_kb - prev_rss_kb) / 1024:+.0f}MB"

        glibc = "n/a"
        if mi:
            glibc = (
                f"heap={_mbb(mi['arena'])} mmap={_mbb(mi['hblkhd'])}({mi['hblks']}) "
                f"inuse={_mbb(mi['uordblks'])} free={_mbb(mi['fordblks'])} top={_mbb(mi['keepcost'])}"
            )
        pym = "n/a"
        if pm:
            pym = (
                f"arenas={pm.get('arenas', '?')}({_mbb(pm.get('arena_bytes'))}) "
                f"used={_mbb(pm.get('allocated_bytes'))} avail={_mbb(pm.get('available_bytes'))} "
                f"hwm={pm.get('arenas_hwm', '?')}"
            )
        traced_s = "off"
        if traced:
            traced_s = f"{traced[0] / _MB:.0f}MB peak={traced[1] / _MB:.0f}MB"

        result.update({
            "rss_kb": rss_kb, "swap_kb": status.get("VmSwap"), "hwm_kb": status.get("VmHWM"),
            "mallinfo2": mi, "pymalloc": pm, "traced": traced, "py_blocks": py_blocks,
            "collected": collected,
        })

        logger.info(
            "MEMREPORT %s rss=%s swap=%s hwm=%s anon=%s file=%s pss=%s threads=%s%s"
            " | glibc: %s | pymalloc: %s | py_blocks=%s traced=%s%s | %s",
            tag, _mb(rss_kb), _mb(status.get("VmSwap")), _mb(status.get("VmHWM")),
            _mb(status.get("RssAnon")), _mb(status.get("RssFile")), _mb(rollup.get("Pss")),
            status.get("Threads", threading.active_count()), delta,
            glibc, pym,
            f"{py_blocks:,}" if py_blocks is not None else "n/a", traced_s,
            f" gc_collected={collected}" if collected is not None else "",
            _caches_summary(),
        )

        try:
            n_objects, movers = gc_summary(top)
            result["gc_objects"] = n_objects
            logger.info("MEMREPORT %s gc_objects=%s movers: %s", tag, f"{n_objects:,}", movers)
        except Exception:  # noqa: BLE001
            logger.debug("MEMREPORT gc summary failed", exc_info=True)

        try:
            n_threads, lines = thread_dump()
            result["threads"] = n_threads
            result["busy_threads"] = lines
            logger.info(
                "MEMREPORT %s threads=%d busy=%d: %s", tag, n_threads, len(lines),
                " || ".join(lines) or "(none)",
            )
        except Exception:  # noqa: BLE001
            logger.debug("MEMREPORT thread dump failed", exc_info=True)

        snap = tracemalloc_snapshot()
        if snap is not None:
            root = _app_root()
            try:
                n = top * 2 if full else top
                logger.info(
                    "MEMREPORT %s tracemalloc top: %s", tag,
                    _fmt_stats(snap.statistics("lineno"), n, root, growth=False),
                )
                if prev_snapshot is not None:
                    logger.info(
                        "MEMREPORT %s tracemalloc growth since start: %s", tag,
                        _fmt_stats(snap.compare_to(prev_snapshot, "lineno"), n, root, growth=True),
                    )
                if full:
                    logger.info(
                        "MEMREPORT %s tracemalloc by file: %s", tag,
                        _fmt_stats(snap.statistics("filename"), top, root, growth=False),
                    )
            except Exception:  # noqa: BLE001
                logger.debug("MEMREPORT tracemalloc summary failed", exc_info=True)

        if full:
            maps = largest_mappings()
            result["largest_mappings"] = maps
            if maps:
                logger.info("MEMREPORT %s largest maps: %s", tag, " | ".join(maps))
    except Exception:  # noqa: BLE001 — never let diagnostics break a task
        logger.debug("MEMREPORT failed", exc_info=True)
    return result


# ---------------------------------------------------------------------------
# Celery task boundaries (wired in config.celery)
# ---------------------------------------------------------------------------

_task_state: dict[str, tuple[int | None, object, float]] = {}
_task_state_lock = threading.Lock()


def task_prerun_report(task_id: str, task_name: str | None) -> None:
    """Record RSS (and a tracemalloc snapshot) when a non-trivial task starts."""
    if not worker_debug_enabled() or _skip_task(task_name):
        return
    try:
        status = read_proc_status()
        rss_kb = status.get("VmRSS")
        snap = tracemalloc_snapshot()
        with _task_state_lock:
            _task_state[task_id] = (rss_kb, snap, time.monotonic())
        logger.info(
            "MEMREPORT task_start name=%s id=%s rss=%s swap=%s threads=%s",
            task_name, task_id, _mb(rss_kb), _mb(status.get("VmSwap")),
            status.get("Threads", threading.active_count()),
        )
    except Exception:  # noqa: BLE001
        logger.debug("MEMREPORT prerun failed", exc_info=True)


def task_postrun_report(task_id: str, task_name: str | None) -> None:
    """Full report when a non-trivial task ends: delta RSS, what grew, what's held."""
    if not worker_debug_enabled() or _skip_task(task_name):
        return
    try:
        with _task_state_lock:
            prev = _task_state.pop(task_id, None)
        prev_rss, prev_snap, t0 = prev if prev else (None, None, None)
        dur = f" took={time.monotonic() - t0:.0f}s" if t0 else ""
        memory_report(
            f"task_end name={task_name} id={task_id}{dur}",
            full=True, prev_snapshot=prev_snap, prev_rss_kb=prev_rss, collect=True,
        )
    except Exception:  # noqa: BLE001
        logger.debug("MEMREPORT postrun failed", exc_info=True)


# ---------------------------------------------------------------------------
# Periodic sampler (worker)
# ---------------------------------------------------------------------------

_sampler_started = False
_sampler_lock = threading.Lock()


def _sampler_loop(interval: float) -> None:
    logger.info("MEMREPORT sampler started (interval=%ss)", interval)
    while True:
        time.sleep(interval)
        try:
            memory_report("sample")
        except Exception:  # noqa: BLE001
            logger.debug("MEMREPORT sample failed", exc_info=True)


def maybe_start_worker_sampler(env: dict[str, str] | None = None) -> bool:
    """Start the periodic worker sampler iff ``MEM_DEBUG_WORKER`` is set. Idempotent."""
    global _sampler_started
    env = env if env is not None else dict(os.environ)
    if not worker_debug_enabled(env):
        return False
    with _sampler_lock:
        if _sampler_started:
            return True
        try:
            interval = max(15, min(3600, int(env.get("MEM_DEBUG_INTERVAL", 120))))
        except (TypeError, ValueError):
            interval = 120
        ensure_tracemalloc(env)
        thread = threading.Thread(
            target=_sampler_loop, args=(interval,), name="memreport", daemon=True,
        )
        thread.start()
        _sampler_started = True
        # A first report right away gives the post-boot baseline.
        memory_report("boot", full=True)
        return True


__all__ = [
    "memory_report",
    "worker_debug_enabled",
    "task_prerun_report",
    "task_postrun_report",
    "maybe_start_worker_sampler",
    "ensure_tracemalloc",
    "parse_debugmallocstats",
    "pymalloc_stats",
    "mallinfo2",
    "thread_dump",
    "largest_mappings",
]
