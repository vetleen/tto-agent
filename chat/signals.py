from __future__ import annotations

import logging

from django.db import transaction
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
    # updated_at drives the sidebar's Today/Yesterday/Earlier grouping and
    # ordering — bump it on activity (a .update() bypasses auto_now).
    ChatThread.objects.filter(pk=instance.thread_id).update(
        retain_until=now + RETENTION_PERIODS["chat.ChatThread"],
        updated_at=now,
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


def _delete_file_on_commit(storage, name, label, pk):
    # post_delete fires inside the delete's transaction: removing the file only
    # once the row deletion commits means a concurrent serve sees a clean 404
    # (not a row whose blob is gone), and a rolled-back delete keeps its file.
    def _cleanup():
        try:
            storage.delete(name)
        except Exception:
            logger.exception("Failed to delete %s id=%s path=%s", label, pk, name)

    transaction.on_commit(_cleanup)


@receiver(post_delete, sender="chat.ChatAttachment")
def delete_file_on_chat_attachment_delete(sender, instance, **kwargs):
    file_field = instance.file
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    _delete_file_on_commit(file_field.storage, name, "file for ChatAttachment", instance.pk)


@receiver(post_delete, sender="chat.Asset")
def delete_blob_on_asset_delete(sender, instance, **kwargs):
    # A *reference* (version-owned) asset has a blank blob — its bytes live on the
    # data-room version and must NOT be deleted here; only blob-owning assets
    # (generated / attached images) carry a file to clean up (the guards below
    # skip the blank case). Without this, deleting a thread/canvas/message — or the
    # enforce_retention GDPR purge — cascades the Asset rows but orphans the image
    # blobs in storage (personal data surviving the purge + unbounded growth).
    file_field = instance.blob
    if not file_field:
        return
    name = file_field.name
    if not name:
        return
    _delete_file_on_commit(file_field.storage, name, "blob for Asset", instance.pk)
