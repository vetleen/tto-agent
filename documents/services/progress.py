"""Best-effort processing-progress store (stage + current/total).

Backs the granular progress the document list shows during ingestion — a stage
label ("Reading the file…", "Viewing images 12 of 46…", "Indexing…") instead of an
opaque spinner — and, through the ``attachments`` store, the same thing for chat
/ meeting attachments (composer pill + the held turn's bubble). State lives in
the Redis cache (``core.cache.ResilientRedisCache``, DB 1), NOT the database: it
is high-churn (updated per described image), ephemeral, and a lost write is
harmless. The cache is fail-open, so a Redis blip silently drops a write and
reads back as "no progress" — the poll endpoints then just fall back to the
coarse status. Never let a progress call raise into a pipeline.

Two stores share one implementation, differing only in the key namespace:

* ``versions`` — keyed by ``DataRoomDocumentVersion`` id (the ``current_version``
  the document poll endpoint resolves per document). The module-level
  ``set_stage`` / ``bump_progress`` / ``read_many`` / ``clear`` are its aliases.
* ``attachments`` — keyed by ``str(ChatAttachment.id)`` (a UUID); stages
  ``extracting``, ``describing_images``, ``rendering``.

Same TTL-keyed pattern as ``core.spend.get_budget_status_cached``.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


def _ttl() -> int:
    return getattr(settings, "DOCPROGRESS_CACHE_TTL", 600)


class ProgressStore:
    """A ``{stage, current, total}`` dict per owner id under one key namespace."""

    def __init__(self, key_template: str):
        # Bump the version suffix in the template if the stored dict's shape changes.
        self._key_template = key_template

    def key(self, owner_id) -> str:
        return self._key_template.format(id=owner_id)

    def set_stage(self, owner_id, stage: str, *, current: int = 0, total: int = 0) -> None:
        """Record the current processing *stage* (and optional counter).

        ``stage`` is a short machine token the frontend maps to display copy
        (e.g. ``"extracting"``, ``"describing_images"``, ``"chunking"``,
        ``"embedding"``, ``"rendering"``). ``current``/``total`` drive the
        "N of M" counter and are 0 for stages without a natural count.
        """
        try:
            cache.set(
                self.key(owner_id),
                {"stage": stage, "current": int(current), "total": int(total)},
                _ttl(),
            )
        except Exception:  # pragma: no cover - cache is already fail-open
            logger.debug("progress.set_stage failed for %s", owner_id, exc_info=True)

    def bump(self, owner_id, stage: str, current: int, total: int) -> None:
        """Overwrite the progress dict with a new counter value.

        Called from the main thread as each embedded-image description (or
        rendered slide) completes, so it just rewrites the whole (self-consistent)
        dict and refreshes the TTL — no read-modify-write, no reliance on INCR.
        """
        self.set_stage(owner_id, stage, current=current, total=total)

    def read(self, owner_id) -> dict | None:
        """The live ``{stage, current, total}`` dict for one owner, or ``None``."""
        return self.read_many([owner_id]).get(owner_id)

    def read_many(self, owner_ids) -> dict:
        """Return ``{owner_id: {stage, current, total}}`` for the ids that have a
        live progress dict (keys as passed). One cache round-trip; missing ids
        are simply absent.
        """
        owner_ids = [v for v in owner_ids if v is not None]
        if not owner_ids:
            return {}
        try:
            found = cache.get_many([self.key(v) for v in owner_ids])
        except Exception:  # pragma: no cover - cache is already fail-open
            logger.debug("progress.read_many failed", exc_info=True)
            return {}
        out: dict = {}
        for v in owner_ids:
            val = found.get(self.key(v))
            if isinstance(val, dict) and val.get("stage"):
                out[v] = val
        return out

    def clear(self, owner_id) -> None:
        """Drop an owner's progress dict (on terminal status). The TTL also reaps
        it, so this is only a promptness optimization.
        """
        try:
            cache.delete(self.key(owner_id))
        except Exception:  # pragma: no cover - cache is already fail-open
            logger.debug("progress.clear failed for %s", owner_id, exc_info=True)


versions = ProgressStore("docprogress:v1:{id}")
attachments = ProgressStore("attprogress:v1:{id}")


# ── Data-room aliases (the original module API) ───────────────────────────
def _key(version_id: int) -> str:
    return versions.key(version_id)


def set_stage(version_id: int, stage: str, *, current: int = 0, total: int = 0) -> None:
    versions.set_stage(version_id, stage, current=current, total=total)


def bump_progress(version_id: int, stage: str, current: int, total: int) -> None:
    versions.bump(version_id, stage, current, total)


def read_many(version_ids) -> dict[int, dict]:
    return versions.read_many(version_ids)


def clear(version_id: int) -> None:
    versions.clear(version_id)
