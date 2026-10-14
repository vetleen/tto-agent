"""Tiny Sentry helpers: never raise, no-op without the SDK or without init."""

from __future__ import annotations

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _apply_tags(set_tag, tags: dict) -> None:
    for key, value in tags.items():
        if value is None:
            continue
        try:
            set_tag(key, str(value))
        except Exception:
            logger.debug("Could not set Sentry tag %s", key, exc_info=True)


def set_sentry_tags(**tags) -> None:
    """Tag the current Sentry isolation scope with ``tags``.

    For code that already runs inside Sentry's own isolation — a Celery task, a
    Django request. Values are stringified; ``None`` values are skipped. Safe to
    call anywhere: without ``sentry_sdk`` (or before ``sentry_sdk.init``) it does
    nothing, and a failing SDK call is logged at DEBUG rather than raised, so
    observability can never break the code it observes.
    """
    try:
        import sentry_sdk
    except ImportError:
        return
    _apply_tags(sentry_sdk.set_tag, tags)


@contextmanager
def sentry_scope(**tags):
    """Run the block under its own forked Sentry isolation scope carrying ``tags``.

    For code Sentry does not isolate by itself — a Channels consumer's turn task,
    where a bare ``set_tag`` would land on the scope the whole process shares and
    leak into every other connection's events. The fork inherits the parent's
    breadcrumbs and tags, takes the new ones, and is dropped on exit; asyncio
    tasks created inside the block and (with ``ThreadingIntegration``) threads
    started inside it inherit the fork. No-op without the SDK; never raises on
    its own account.
    """
    try:
        import sentry_sdk

        cm = sentry_sdk.isolation_scope()
        scope = cm.__enter__()
    except Exception:
        yield
        return
    try:
        _apply_tags(scope.set_tag, tags)
        yield
    except BaseException:
        import sys

        if not cm.__exit__(*sys.exc_info()):
            raise
    else:
        cm.__exit__(None, None, None)
