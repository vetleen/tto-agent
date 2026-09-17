"""Best-effort, per-version processing-progress store (stage + current/total).

Backs the granular progress the document list shows during ingestion — a stage
label ("Extracting…", "Reading images 12/46", "Indexing…") instead of an opaque
spinner. State lives in the Redis cache (``core.cache.ResilientRedisCache``, DB 1),
NOT the database: it is high-churn (updated per described image), ephemeral, and a
lost write is harmless. The cache is fail-open, so a Redis blip silently drops a
write and reads back as "no progress" — the poll endpoint then just falls back to
the coarse document status. Never let a progress call raise into the pipeline.

Keyed by *version* id (the ``current_version`` the poll endpoint resolves per
document). Same TTL-keyed pattern as ``core.spend.get_budget_status_cached``.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Bump the version suffix if the stored dict's shape ever changes.
_KEY = "docprogress:v1:{version_id}"


def _key(version_id: int) -> str:
    return _KEY.format(version_id=version_id)


def _ttl() -> int:
    return getattr(settings, "DOCPROGRESS_CACHE_TTL", 600)


def set_stage(version_id: int, stage: str, *, current: int = 0, total: int = 0) -> None:
    """Record the current processing *stage* (and optional counter) for a version.

    ``stage`` is a short machine token the frontend maps to display copy
    (e.g. ``"extracting"``, ``"describing_images"``, ``"chunking"``,
    ``"embedding"``, ``"scanning"``). ``current``/``total`` drive the
    "N of M" counter and are 0 for stages without a natural count.
    """
    try:
        cache.set(
            _key(version_id),
            {"stage": stage, "current": int(current), "total": int(total)},
            _ttl(),
        )
    except Exception:  # pragma: no cover - cache is already fail-open
        logger.debug("progress.set_stage failed for version %s", version_id, exc_info=True)


def bump_progress(version_id: int, stage: str, current: int, total: int) -> None:
    """Overwrite the progress dict with a new counter value.

    Called from the main thread as each embedded-image description completes, so
    it just rewrites the whole (self-consistent) dict and refreshes the TTL — no
    read-modify-write, no reliance on Redis INCR.
    """
    set_stage(version_id, stage, current=current, total=total)


def read_many(version_ids) -> dict[int, dict]:
    """Return ``{version_id: {stage, current, total}}`` for the ids that have a
    live progress dict. One cache round-trip; missing ids are simply absent.
    """
    version_ids = [v for v in version_ids if v is not None]
    if not version_ids:
        return {}
    try:
        found = cache.get_many([_key(v) for v in version_ids])
    except Exception:  # pragma: no cover - cache is already fail-open
        logger.debug("progress.read_many failed", exc_info=True)
        return {}
    out: dict[int, dict] = {}
    for v in version_ids:
        val = found.get(_key(v))
        if isinstance(val, dict) and val.get("stage"):
            out[v] = val
    return out


def clear(version_id: int) -> None:
    """Drop a version's progress dict (on terminal status). The TTL also reaps it,
    so this is only a promptness optimization.
    """
    try:
        cache.delete(_key(version_id))
    except Exception:  # pragma: no cover - cache is already fail-open
        logger.debug("progress.clear failed for version %s", version_id, exc_info=True)
