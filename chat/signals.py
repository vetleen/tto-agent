from __future__ import annotations

import logging

from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver
from django.utils import timezone

from core.retention import RETENTION_PERIODS

logger = logging.getLogger(__name__)


@receiver(post_save, sender="chat.ChatMessage")
def extend_retention_on_message(sender, instance, **kwargs):
    from chat.models import ChatThread
    from documents.models import DataRoom

    now = timezone.now()
    ChatThread.objects.filter(pk=instance.thread_id).update(
        retain_until=now + RETENTION_PERIODS["chat.ChatThread"],
    )
    DataRoom.objects.filter(
        thread_links__thread_id=instance.thread_id,
    ).update(retain_until=now + RETENTION_PERIODS["documents.DataRoom"])


@receiver(post_save, sender="chat.ChatThreadDataRoom")
def extend_dataroom_retain_on_attach(sender, instance, created, **kwargs):
    if not created:
        return
    from documents.models import DataRoom

    DataRoom.objects.filter(pk=instance.data_room_id).update(
        retain_until=timezone.now() + RETENTION_PERIODS["documents.DataRoom"],
    )


@receiver(post_delete, sender="chat.ChatAttachment")
def delete_file_on_chat_attachment_delete(sender, instance, **kwargs):
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
            "Failed to delete file for ChatAttachment id=%s path=%s",
            instance.pk,
            name,
        )
