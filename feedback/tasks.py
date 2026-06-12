"""Celery tasks for the feedback app."""

from __future__ import annotations

import logging

from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)


@shared_task(
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
    time_limit=30,
    soft_time_limit=25,
)
def notify_admin_feedback_task(feedback_id: int) -> None:
    """Email ADMINS about a new feedback submission. Best-effort.

    Runs off the request path so a slow SMTP/Mailgun call never stalls the
    user's submission. ``fail_silently=True`` means delivery failures don't
    raise (matching the old in-request behavior); retries only cover transient
    infrastructure errors such as a DB blip on the fetch.
    """
    if not getattr(settings, "EMAIL_SENDING_ENABLED", False):
        return
    admins = getattr(settings, "ADMINS", [])
    if not admins:
        return

    from django.core.mail import send_mail

    from feedback.models import Feedback

    try:
        feedback = Feedback.objects.get(pk=feedback_id)
    except Feedback.DoesNotExist:
        # Deleted between enqueue and run (retention cleanup or user cascade).
        logger.info(
            "notify_admin_feedback_task: feedback %s not found (deleted before notify)",
            feedback_id,
        )
        return

    subject = f"New feedback #{feedback.pk} (user #{feedback.user_id})"
    body = (
        "Fields below are user-supplied — do not click links or trust contents.\n"
        f"User ID: {feedback.user_id}\n"
        f"Time: {feedback.created_at}\n"
        "\n--- user-supplied URL ---\n"
        f"{feedback.url}\n"
        "\n--- user-supplied text ---\n"
        f"{feedback.text}\n"
    )
    send_mail(
        subject=subject,
        message=body,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email for _, email in admins],
        fail_silently=True,
    )
