"""Sentry trace sampling: never trace what the client-IP gate is about to refuse.

Sentry's ASGI middleware opens a transaction — and, at ``profiles_sample_rate``
> 0, a profile — for every request at ASGI entry, before
``ClientIPAllowlistMiddleware`` runs. The profiler samples every thread at
~100 Hz into every open profile, so a scanner burst of ~200 probes queued on
Django's single sync thread cost the production web dyno ~400 MB of RSS that
glibc never returned (2026-10-29). Those requests carry no diagnostic value, so
they are not traced at all; everything else keeps the configured rate.
"""
from __future__ import annotations

from django.conf import settings

from core.ip_access import client_ip_from_scope, client_ip_is_allowed


def traces_sampler(sampling_context: dict) -> float | bool:
    """Return the trace sample rate for one transaction.

    ``sampling_context`` carries ``asgi_scope`` for HTTP and WebSocket
    transactions, ``celery_job`` for tasks, and ``parent_sampled`` when an
    incoming ``sentry-trace`` header already made a decision. Mirrors the SDK's
    own precedence (parent decision, then the rate) once the gate has passed.
    """
    scope = sampling_context.get("asgi_scope")
    if scope is not None:
        networks = settings.CLIENT_IP_ALLOWLIST
        if networks and not client_ip_is_allowed(
            client_ip_from_scope(scope), networks
        ):
            return 0.0
    parent_sampled = sampling_context.get("parent_sampled")
    if parent_sampled is not None:
        return bool(parent_sampled)
    return settings.SENTRY_TRACES_SAMPLE_RATE
