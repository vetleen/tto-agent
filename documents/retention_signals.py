from __future__ import annotations

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

from core.retention import RETENTION_PERIODS

logger = logging.getLogger(__name__)


@receiver(post_save, sender="documents.DataRoomDocument")
def extend_dataroom_retain_on_document_change(sender, instance, **kwargs):
    from documents.models import DataRoom

    DataRoom.objects.filter(pk=instance.data_room_id).update(
        retain_until=timezone.now() + RETENTION_PERIODS["documents.DataRoom"],
    )
