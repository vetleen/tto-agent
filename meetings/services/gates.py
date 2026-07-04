"""Shared access/concurrency gates for the meetings app.

Keeping these in one service module means the HTTP views and the WebSocket
consumer enforce the *same* rules rather than hand-copying the queries.
"""
from __future__ import annotations


def user_has_active_transcription(user, *, exclude_pk=None) -> bool:
    """True when *user* already has a meeting actively transcribing.

    "Actively transcribing" = ``status == LIVE_TRANSCRIBING``, which covers both
    the live-recording path and the audio-upload path (the upload view/orchestrator
    hold the meeting in that state until the Celery job finishes). This is the
    single source of truth for the "one transcription at a time per user" cap;
    both ``meetings.views`` and ``meetings.consumers`` call it.

    Pass ``exclude_pk`` to ignore a specific meeting (e.g. the one being resumed)
    so reconnecting to an already-live meeting isn't treated as a second session.
    """
    from meetings.models import Meeting

    qs = Meeting.objects.filter(
        created_by=user, status=Meeting.Status.LIVE_TRANSCRIBING
    )
    if exclude_pk is not None:
        qs = qs.exclude(pk=exclude_pk)
    return qs.exists()
