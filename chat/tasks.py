"""Celery tasks for the chat app."""

from __future__ import annotations

import uuid

from celery import shared_task


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
    time_limit=300,
    soft_time_limit=270,
)
def run_subagent_task(run_id: str) -> None:
    """Execute a sub-agent run asynchronously via Celery."""
    from chat.subagent_service import run_subagent

    run_subagent(uuid.UUID(run_id))
