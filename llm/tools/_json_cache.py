"""Compressed, size-bounded JSON entries in the shared Django cache.

The cache shares the production Redis (50 MB, noeviction) with the Celery broker,
so an unbounded entry can starve the broker and crash the worker (WILFRED-99/9A/9B).
Values are stored as zlib-compressed JSON bytes; an entry whose compressed size is
over ``max_bytes`` is not cached at all, and one over ``large_bytes`` is kept only
for ``large_ttl``. Every cache error is swallowed: the cache is an optimisation.

Callers switching an existing key to this format must bump the key's version so a
legacy (uncompressed) entry is never read; an unreadable entry reads as a miss.
"""

from __future__ import annotations

import json
import logging
import zlib

logger = logging.getLogger(__name__)


def cache_get_json(cache, key: str):
    """Return the decoded value for *key*, or None on a miss / cache error /
    unreadable entry."""
    try:
        blob = cache.get(key)
    except Exception:
        logger.debug("json cache: read failed for key=%s", key)
        return None
    if blob is None:
        return None
    try:
        return json.loads(zlib.decompress(blob))
    except (zlib.error, TypeError, ValueError):
        logger.debug("json cache: unreadable entry for key=%s", key)
        return None


def cache_set_json(
    cache,
    key: str,
    value,
    *,
    ttl: int,
    max_bytes: int,
    large_bytes: int | None = None,
    large_ttl: int | None = None,
) -> bool:
    """Store *value* compressed under *key*. Returns True if it was cached."""
    try:
        blob = zlib.compress(json.dumps(value).encode("utf-8"), 6)
        if len(blob) > max_bytes:
            logger.info("json cache: not caching key=%s (compressed=%d bytes > cap)", key, len(blob))
            return False
        timeout = ttl
        if large_bytes is not None and large_ttl is not None and len(blob) > large_bytes:
            timeout = large_ttl
        cache.set(key, blob, timeout=timeout)
        return True
    except Exception:
        logger.debug("json cache: write failed for key=%s", key)
        return False
