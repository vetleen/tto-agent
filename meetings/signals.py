from __future__ import annotations

import logging

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from django.utils import timezone

from core.retention import RETENTION_PERIODS

logger = logging.getLogger(__name__)


@receiver(post_save, sender="meetings.MeetingTranscriptSegment")
def extend_meeting_retain_on_segment(sender, instance, **kwargs):
    from meetings.models import Meeting

    Meeting.objects.filter(pk=instance.meeting_id).update(
        retain_until=timezone.now() + RETENTION_PERIODS["meetings.Meeting"],
    )


@receiver(post_save, sender="meetings.MeetingAttachment")
def extend_meeting_retain_on_attachment(sender, instance, created, **kwargs):
    if not created:
        return
    from meetings.models import Meeting

    Meeting.objects.filter(pk=instance.meeting_id).update(
        retain_until=timezone.now() + RETENTION_PERIODS["meetings.Meeting"],
    )


@receiver(post_delete, sender="meetings.MeetingAttachment")
def delete_file_on_meeting_attachment_delete(sender, instance, **kwargs):
    file_field = instance.file
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    try:
        file_field.storage.delete(name)
    except Exception:
        logger.exception(
            "Failed to delete file for MeetingAttachment id=%s path=%s",
            instance.pk,
            name,
        )
