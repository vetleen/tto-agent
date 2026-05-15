from __future__ import annotations

import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

logger = logging.getLogger(__name__)


@receiver(post_delete, sender="feedback.Feedback")
def delete_screenshot_on_feedback_delete(sender, instance, **kwargs):
    file_field = instance.screenshot
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    try:
        file_field.storage.delete(name)
    except Exception:
        logger.exception(
            "Failed to delete screenshot for Feedback id=%s path=%s",
            instance.pk,
            name,
        )
