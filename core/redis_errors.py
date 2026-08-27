"""Shared classification of transient Redis failures on best-effort paths.

Channel-layer broadcasts (``group_send``) are best-effort by design: nobody may be
watching the thread, so a failed push must never break the turn. Swallowing *every*
exception at DEBUG achieved that, but it also meant a genuine Redis capacity problem
stayed invisible until a user reported the symptom (WILFRED-6P) — prod ``LOG_LEVEL``
is INFO, so those DEBUG lines were never emitted at all, not even as Sentry
breadcrumbs.

These helpers keep the swallowing while separating the two causes:

  - "nobody is listening" / a programming error -> DEBUG, as before;
  - "Redis is failing" -> WARNING, which the Sentry LoggingIntegration
    (``event_level=WARNING``) turns into an event.

Callers on hot paths should throttle: see ``BroadcastSink`` in ``chat/sinks.py``,
which streams one event per token and reports only the first blip of an episode.
"""

from __future__ import annotations

import logging

try:
    from redis.exceptions import (
        ConnectionError as _RedisConnectionError,
        TimeoutError as _RedisTimeoutError,
    )

    # MaxConnectionsError — a bounded pool at its cap — subclasses ConnectionError,
    # so pool exhaustion is covered here too. Bare OSError catches the builtin
    # ConnectionResetError raised out of the SSL-handshake path when Heroku drops an
    # over-limit connection. See CHANNEL_LAYERS in config/settings.py.
    REDIS_BLIP: tuple[type[BaseException], ...] = (
        _RedisConnectionError,
        _RedisTimeoutError,
        OSError,
    )
except Exception:  # pragma: no cover - redis is always installed in this app
    REDIS_BLIP = (OSError,)


def is_redis_blip(exc: BaseException) -> bool:
    """Whether ``exc`` is a transient Redis connectivity failure."""
    return isinstance(exc, REDIS_BLIP)


def log_broadcast_failure(
    logger: logging.Logger,
    exc: BaseException,
    msg: str,
    *args,
    redis_level: int = logging.WARNING,
) -> None:
    """Log a swallowed broadcast failure at a level reflecting its cause.

    Redis blips are logged at ``redis_level`` (WARNING by default, so they reach
    Sentry); anything else stays at DEBUG. Pass ``redis_level=logging.DEBUG`` to
    suppress a repeat report while an outage is already known.
    """
    level = redis_level if is_redis_blip(exc) else logging.DEBUG
    logger.log(level, msg, *args, exc_info=exc)
